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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
/**
mean: The mean of the distribution.
focal_x: The focal length of the camera in the x-direction.
focal_y: The focal length of the camera in the y-direction.
tan_fovx: The tangent of the field of view in the x-direction.
tan_fovy: The tangent of the field of view in the y-direction.
cov3D: The 3D covariance matrix.
viewmatrix: The view matrix.

The function first transforms the mean of the distribution from world space to screen space. 
Then, it computes the Jacobian matrix, which is a matrix that describes how the change in the screen space coordinates is related to the change in the world space coordinates. 
The function then uses the Jacobian matrix to compute the 2D covariance matrix.
**/
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	/**
	transforms the mean of the distribution from world space to screen space using the view matrix.
  	**/
	float3 t = transformPoint4x3(mean, viewmatrix);

	/**
 	clamps the screen space coordinates of the mean to the frustum
  	**/
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	/**
	creates the Jacobian matrix
  	**/
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);
	/**
 	he first row describes how the change in the screen space x-coordinate is related to the change in the world space x-coordinatet
  	the second row describes how the change in the screen space y-coordinate is related to the change in the world space y-coordinate
   	the third row is all zeros because the z-coordinate does not affect the screen space coordinates.
 	**/

	/**
 	This block of code multiplies the view matrix and the Jacobian matrix. 
 	**/
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	glm::mat3 T = W * J;

	/**
 	This block of code computes the 2D covariance matrix. 
  	The 2D covariance matrix is computed by multiplying the transpose of the Jacobian matrix, the transpose of the 3D covariance matrix, and the Jacobian matrix.
  	**/
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	/**
	This block of code applies a low-pass filter to the covariance matrix. 
 	The low-pass filter ensures that the covariance matrix is at least one pixel wide and high. 
  	This is done to prevent the Gaussian distribution from becoming too narrow and causing artifacts.
  	**/
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
/**
scale: The scale of the distribution.
mod: The modulation factor.
rot: The rotation of the distribution.
cov3D: The 3D covariance matrix.

\begin{align*}
\text{computeCov3D}(scale, mod, rot, cov3D) &= \begin{aligned}
&\text{Create scaling matrix} \\
&S = \text{diag}(mod \cdot scale) \\
&\text{Normalize quaternion to get valid rotation} \\
&q = \frac{rot}{\left\| rot \right\|} \\
&r = q_x, x = q_y, y = q_z, z = q_w \\
&\text{Compute rotation matrix from quaternion} \\
&R = \begin{pmatrix}
1 - 2(y^2 + z^2) & 2(x \cdot y - r \cdot z) & 2(x \cdot z + r \cdot y) \\
2(x \cdot y + r \cdot z) & 1 - 2(x^2 + z^2) & 2(y \cdot z - r \cdot x) \\
2(x \cdot z - r \cdot y) & 2(y \cdot z + r \cdot x) & 1 - 2(x^2 + y^2)
\end{pmatrix} \\
&\text{Compute 3D world covariance matrix Sigma} \\
&M = S \cdot R \\
&\text{Covariance is symmetric, only store upper right} \\
&cov3D = \text{diag}(M)
\end{aligned}
\end{align*}
**/
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
/**
这是一个名为`preprocessCUDA`的CUDA内核函数，用于对每个高斯点在光栅化之前执行初始处理步骤。

该函数使用了一组参数来执行操作，包括：

- `P`：高斯点的数量。
- `D`：SH系数的数量。
- `M`：SH的阶数。
- `orig_points`：高斯点的原始坐标数组。
- `scales`：高斯点的缩放因子数组。
- `scale_modifier`：缩放修正因子。
- `rotations`：高斯点的旋转数组。
- `opacities`：高斯点的透明度数组。
- `shs`：球谐系数数组。
- `clamped`：指示高斯点是否被裁剪的布尔数组。
- `cov3D_precomp`：预计算的3D协方差矩阵数组。
- `colors_precomp`：预计算的颜色数组。
- `viewmatrix`：视图矩阵。
- `projmatrix`：投影矩阵。
- `cam_pos`：摄像机位置。
- `W`：输出屏幕的宽度。
- `H`：输出屏幕的高度。
- `tan_fovx`、`tan_fovy`：水平和垂直方向的视场角的正切值。
- `focal_x`、`focal_y`：投影矩阵的焦点坐标。
- `radii`：高斯点的半径数组。
- `points_xy_image`：高斯点在图像空间的XY坐标数组。
- `depths`：高斯点在视图空间的深度数组。
- `cov3Ds`：3D协方差矩阵数组。
- `rgb`：高斯点的颜色数组。
- `conic_opacity`：高斯点的锥体参数和不透明度数组。
- `grid`：线程块的网格大小。
- `tiles_touched`：记录每个高斯点触碰到的tile数量的数组。
- `prefiltered`：指示是否对高斯点进行预过滤的布尔值。

该函数执行了对每个高斯点的初始处理步骤，包括以下操作：

1. 对于每个高斯点，首先将其半径和触碰到的tile数量初始化为0。
2. 执行近裁剪操作，检查该高斯点是否在视锥体内，如果不在，则退出处理。具体而言，将高斯点投影到像素坐标系下，查看z值（深度）是否小于0.2。
3. 通过投影变换将计算高斯点在2D屏幕空间的坐标。
4. 根据放缩参数 scale 和 旋转四元数参数 q 参数计算3D协方差矩阵。
5. 计算2D屏幕空间协方差矩阵，并进行逆运算。
6. 使用协方差矩阵计算2D屏幕空间的范围，即高斯点所覆盖的屏幕空间矩形区域。如果范围为0，则退出处理。
7. 如果没有预计算的颜色，根据球谐系数计算颜色，并将结果存储在rgb数组中。
8. 存储一些有用的数据，如高斯点的深度、半径、在图像空间的坐标、锥形参数和不透明度。
9. 记录高斯点触碰到的tile数量。

这个函数对每个高斯点进行单个处理，并根据其属性和屏幕空间位置，为后续的光栅化步骤准备必要的数据。
**/
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// 从 scale 和 旋转四元数 计算 3D 高斯的协方差矩阵
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
/**
这是一个名为`renderCUDA`的CUDA内核函数，用于对每个像素进行光栅化和渲染操作。

该函数使用了一组参数来执行操作，包括：

- `CHANNELS`：颜色通道的数量。
- `ranges`：每个线程块处理的像素ID范围的数组。
- `point_list`：处理的高斯点的索引列表。
- `W`、`H`：输出屏幕的宽度和高度。
- `points_xy_image`：高斯点在图像空间的XY坐标数组。
- `features`：高斯点的特征数组。
- `conic_opacity`：高斯点的锥体参数和不透明度数组。
- `final_T`：最终的T值数组。
- `n_contrib`：每个像素的贡献者数量数组。
- `bg_color`：背景颜色数组。
- `out_color`：输出颜色数组。

该函数实现了光栅化和渲染的主要方法，并在一个线程块中协作地处理一个tile，并且每个线程处理一个像素。函数的主要执行流程如下：

1. 根据线程块的索引计算当前tile的像素范围。
2. 检查当前线程是否与有效像素相关联，如果不是，则退出处理。
3. 加载处理范围的起始和结束ID，并计算需要处理的高斯点数量。
4. 分配存储批处理数据的共享存储器。
5. 初始化辅助变量。
6. 迭代处理批次，直到所有像素完成或处理范围完成。
   - 如果整个线程块均标记为完成，则退出处理。
   - 从全局内存中共同获取每个高斯点的数据到共享存储器。
   - 迭代当前批次的每个高斯点，进行光栅化和计算渲染数据。
   - 更新渲染数据。
7. 所有处理有效像素的线程将最终的渲染数据写入帧和辅助缓冲区。

这个函数在并行处理光栅化和渲染操作上非常高效，并使用共享存储器进行数据共享和协作计算。
**/
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();  // 获取当前线程块
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;  // 计算水平方向的线程块数量
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };  // 计算当前tile的最小像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };  // 计算当前tile的最大像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };  // 计算当前像素的坐标
	uint32_t pix_id = W * pix.y + pix.x;  // 计算当前像素在一维数组中的索引
	float2 pixf = { (float)pix.x, (float)pix.y };  // 当前像素的浮点坐标
	
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;  // 检查当前像素是否有效（在图像范围内）
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;  // 如果像素无效，则将其标记为已完成
	
	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];  // 获取当前线程块处理的像素ID范围
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);  // 计算需要进行处理的批次数
	int toDo = range.y - range.x;  // 当前线程块需要处理的总像素数量
	
	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];  // 共享内存用于存储收集到的高斯点索引
	__shared__ float2 collected_xy[BLOCK_SIZE];  // 共享内存用于存储收集到的高斯点XY坐标
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];  // 共享内存用于存储收集到的高斯点锥体参数和不透明度
	
	// Initialize helper variables
	float T = 1.0f;  // 初始化深度值T
	uint32_t contributor = 0;  // 初始化贡献者计数器
	uint32_t last_contributor = 0;  // 初始化上一个贡献者索引
	float C[CHANNELS] = { 0 };  // 初始化颜色数组

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// 如果整个块都标记为已完成光栅化操作，则结束循环。
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// 从全局内存中共同获取每个高斯点的数据到共享内存。progress是根据循环和线程块内的线程索引计算出的偏移量。然后，将高斯点的索引、位置和锥体参数和不透明度存储到共享内存中。
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];  // 获取当前处理的高斯点索引
			collected_id[block.thread_rank()] = coll_id;  // 存储高斯点索引到共享内存
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];  // 存储高斯点XY坐标到共享内存
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];  // 存储高斯点锥体参数和不透明度到共享内存
		}
		block.sync();  // 同步线程块内的线程
	
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
	
			// Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];  // 获取高斯点的XY坐标
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };  // 计算位置差矢量
			float4 con_o = collected_conic_opacity[j];  // 获取高斯点的锥体参数xyz和不透明度opacities
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;  // 计算权重值
			if (power > 0.0f)
				continue;
	
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));  // 计算颜色的不透明度
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);  // 计算测试的深度值T
			if (test_T < 0.0001f)
			{
				done = true;  // 标记像素已完成处理
				continue;
			}
	
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;  // 计算颜色值的加权和
	
			T = test_T;  // 更新深度值T
	
			// Keep track of last range entry to update this pixel.
			last_contributor = contributor;  // 记录最后一个贡献者索引
		}
	}
	
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
