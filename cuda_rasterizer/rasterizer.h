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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		// 标记在视锥体内可见的点
		static void markVisible(
			int P,                        // 3D点的数量
			float* means3D,               // 3D点的坐标数组
			float* viewmatrix,            // 视图变换矩阵
			float* projmatrix,            // 投影矩阵
			bool* present                 // 表示点是否可见的布尔数组
		);

		// 执行渲染操作
		static int forward(
			std::function<char* (size_t)> geometryBuffer, // 用于渲染计算的几何缓冲区函数
			std::function<char* (size_t)> binningBuffer,  // 用于渲染计算的分箱缓冲区函数
			std::function<char* (size_t)> imageBuffer,    // 用于渲染计算的图片缓冲区函数
			const int P,                   // 3D点的数量
			int D,                         // 3D点的维度（通常为3）
			int M,                         // 球谐函数的次数
			const float* background,       // 背景图像数据
			const int width,               // 图像的宽度
			int height,                    // 图像的高度
			const float* means3D,          // 3D点的坐标数组
			const float* shs,              // 球谐函数数组
			const float* colors_precomp,   // 预计算的颜色数组
			const float* opacities,        // 不透明度数组
			const float* scales,           // 缩放因子数组
			const float scale_modifier,    // 缩放因子的修改值
			const float* rotations,        // 旋转角度数组
			const float* cov3D_precomp,    // 预计算的3D协方差数组
			const float* viewmatrix,       // 视图变换矩阵
			const float* projmatrix,       // 投影矩阵
			const float* cam_pos,          // 相机位置数组
			const float tan_fovx,          // x方向的焦距
			float tan_fovy,                // y方向的焦距
			const bool prefiltered,        // 是否应用预过滤
			float* out_color,              // 渲染后的颜色数组
			int* radii = nullptr           // 点的半径数组，默认为空指针
		);

		// 执行反向传播操作
		static void backward(
			const int P,                   // 3D点的数量
			int D,                         // 3D点的维度（通常为3）
			int M,                         // 球谐函数的次数
			int R,                         // 渲染点的数量
			const float* background,       // 背景图像数据
			const int width,               // 图像的宽度
			int height,                    // 图像的高度
			const float* means3D,          // 3D点的坐标数组
			const float* shs,              // 球谐函数数组
			const float* colors_precomp,   // 预计算的颜色数组
			const float* scales,           // 缩放因子数组
			const float scale_modifier,    // 缩放因子的修改值
			const float* rotations,        // 旋转角度数组
			const float* cov3D_precomp,    // 预计算的3D协方差数组
			const float* viewmatrix,       // 视图变换矩阵
			const float* projmatrix,       // 投影矩阵
			const float* campos,           // 相机位置数组
			const float tan_fovx,          // x方向的焦距
			const float tan_fovy,          // y方向的焦距
			const int* radii,                  // 点的半径数组
			char* geom_buffer,                 // 几何缓冲区
			char* binning_buffer,              // 分箱缓冲区
			char* image_buffer,                // 图片缓冲区
			const float* dL_dpix,              // 损失函数对像素的梯度
			float* dL_dmean2D,                 // 损失函数对2D点坐标的梯度
			float* dL_dconic,                  // 损失函数对投影圆锥的梯度
			float* dL_dopacity,                // 损失函数对不透明度的梯度
			float* dL_dcolor,                  // 损失函数对颜色的梯度
			float* dL_dmean3D,                 // 损失函数对3D点坐标的梯度
			float* dL_dcov3D,                  // 损失函数对3D协方差的梯度
			float* dL_dsh,                     // 损失函数对球谐函数的梯度
			float* dL_dscale,                  // 损失函数对缩放因子的梯度
			float* dL_drot                     // 损失函数对旋转角度的梯度
		);
	};
};

#endif
