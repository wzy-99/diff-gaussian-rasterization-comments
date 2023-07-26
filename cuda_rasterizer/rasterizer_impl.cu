/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		/**
			对于每个guass
   			会根据其覆盖的tile
      			初始化多个实例

    			对于每个实例
       			设置一个key为 [tile的位置 |  深度]
			设置一个value为 在P个guass中的索引
   
                        通过把key设置这种形式有利于按深度排序
  		**/
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
		if (idx == L - 1)
			ranges[currtile].y = L;
	}
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii)
/**
输入参数

geometryBuffer：返回几何缓冲区的函数
binningBuffer：返回分箱缓冲区的函数
imageBuffer：返回图片缓冲区的函数
P：3D点的数量
D：3D点的维度
M：球谐函数的次数
background：背景图像数据
width：图像的宽度
height：图像的高度
means3D：3D点的坐标数组
shs：球谐函数数组
colors_precomp：预计算的颜色数组
opacities：不透明度数组
scales：缩放因子数组
scale_modifier：缩放因子的修改值
rotations：旋转角度数组
cov3D_precomp：预计算的3D协方差数组
viewmatrix：视图变换矩阵
projmatrix：投影矩阵
cam_pos：相机位置数组
tan_fovx：x方向的焦距
tan_fovy：y方向的焦距
prefiltered：是否应用预过滤
out_color：渲染后的颜色数组
radii：点的半径数组（可选参数，默认为nullptr）

该函数执行以下操作：

1. 计算焦距 `focal_y` 和 `focal_x`。
2. 分配并初始化几何状态对象 `geomState` 和图像状态对象 `imgState`。
3. 对每个高斯进行预处理（转换、边界、SH转RGB等）。
4. 计算每个 tile 的高斯数量，并进行前缀和运算。
5. 根据排序键对高斯索引进行排序。
6. 标识排序后高斯索引在每个 tile 中的范围。
7. 并行处理每个 tile 的高斯融合。
8. 返回渲染的高斯数量。
**/
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 这段代码用于确定线程块和线程网格的维度
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	int img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);
	/**
	这段代码使用CUB库的`DeviceScan`函数来计算被高斯点触碰到的所有tile计数的前缀和。
	
	具体地说，它接受以下参数：
	
	- `geomState.scanning_space`：一个用于临时存储扫描结果的缓冲区。
	- `geomState.scan_size`：缓冲区的大小（以字节为单位）。
	- `geomState.tiles_touched`：一个存储每个高斯点触碰到的tile数量的数组。
	- `geomState.point_offsets`：用于存储计算后的前缀和结果的数组。
	- `P`：高斯点的数量。
	
	函数通过对`geomState.tiles_touched`数组进行前缀和操作，计算每个高斯点之前触碰到的tile的总数。例如，如果传入的`tiles_touched`数组为`[2, 3, 0, 2, 1]`，则计算后的结果将为`[2, 5, 5, 7, 8]`。这个前缀和数组存储在`geomState.point_offsets`中，可以在后续步骤中使用。
 	**/

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
	/**
	这段代码用于获取要渲染的高斯点的总数量，并调整辅助缓冲区的大小。
	
	具体而言，代码通过使用`cudaMemcpy`函数从设备内存复制数据到主机内存，将`geomState.point_offsets`数组中的值复制到`num_rendered`变量中。`geomState.point_offsets`是一个存储了经过前缀和计算后的每个高斯点所触碰到的tile的累计数量的数组。通过将`P - 1`作为偏移量，我们可以获取到最后一个tile的累计数量，从而得到要渲染的高斯点的总数量。
	
	获取到总数量后，可以使用这个值来调整辅助缓冲区的大小，以确保它足够容纳所有要渲染的高斯点。这样可以避免缓冲区溢出或浪费存储空间。
 	**/

	int binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);
	/**
	这段代码用于根据要渲染的高斯点的总数量，分配并初始化`BinningState`对象所需的内存空间。
	
	首先，代码调用`required`函数，传入`num_rendered`作为参数，以确定分配内存所需的大小。这个函数返回一个值，表示为了满足所需的内存大小，需要的字节数。这个值存储在`binning_chunk_size`变量中。
	
	接下来，代码使用`binningBuffer`函数，传入`binning_chunk_size`作为参数，从`binningBuffer`对象中分配一块大小为`binning_chunk_size`的内存空间，并将其返回给`binning_chunkptr`指针。
	
	然后，代码使用`BinningState::fromChunk`静态函数，传入`binning_chunkptr`和`num_rendered`作为参数，将之前分配的内存空间初始化为`BinningState`对象。这个函数会创建一个`BinningState`对象，将内存块的首地址和要渲染的高斯点数量传入，并返回初始化后的对象。
	
	最终，通过这段代码，我们分配了足够的内存来容纳`BinningState`对象，并将其初始化为具有指定大小的内存块。这将为之后的处理步骤提供必要的空间和状态。
 	**/

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid
		);
	/**
	这段代码用于为每个要渲染的高斯点生成适当的`[ tile | depth ]`键和相应的重复高斯点索引，以便进行排序。
	
	代码调用了名为`duplicateWithKeys`的CUDA内核函数，使用了一组参数来执行操作。具体而言，使用了以下参数：
	
	- `(P + 255) / 256`：用于计算启动线程块的数量。这个表达式保证了足够数量的线程块来处理高斯点的数量（`P`）。
	- `P`：高斯点的数量。
	- `geomState.means2D`：二维均值点的数组。
	- `geomState.depths`：深度值的数组。
	- `geomState.point_offsets`：每个高斯点所触碰到的tile的累计数量的数组。
	- `binningState.point_list_keys_unsorted`：未排序的高斯点键的数组。
	- `binningState.point_list_unsorted`：未排序的高斯点索引的数组。
	- `radii`：高斯点的半径的数组。
	- `tile_grid`：存储tile的数据结构。
	
	这段代码的目标是为每个高斯点生成适当的`[ tile | depth ]`键（用于排序），并在排序过程中将高斯点的索引进行重复。通过这个操作，每个高斯点将在列表中出现多次，每次使用不同的`[ tile | depth ]`键。这样可以确保高斯点在排序后以适当的位置进行渲染。
	
	调用CUDA内核函数时，启动足够数量的线程块，并使用不同的线程处理不同的高斯点。这个内核函数执行每个高斯点的处理，并根据给定的参数生成相应的键和索引。所有线程块的处理将保证所有高斯点的键和索引都得到正确生成，并准备进行排序操作。
 	**/

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit);

	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
		num_rendered,
		binningState.point_list_keys,
		imgState.ranges
		);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot);
}
