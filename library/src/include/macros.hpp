/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

/*******************************************************************************
 * Macros
 ******************************************************************************/

#ifdef WIN32
#define ROCBLAS_KERNEL(lb_) __global__ __launch_bounds__((lb_)) static void
#define ROCBLAS_KERNEL_NO_BOUNDS __global__ static void
#else
#define ROCBLAS_KERNEL(lb_) __global__ __launch_bounds__((lb_)) void
#define ROCBLAS_KERNEL_NO_BOUNDS __global__ void
#endif

// A storage-class-specifier other than thread_local shall not be specified in an explicit specialization.
// ROCBLAS_KERNEL_INSTANTIATE should be used where kernels are instantiated to avoid use of static.
#ifdef WIN32
#define ROCBLAS_KERNEL_INSTANTIATE __global__
#else
#define ROCBLAS_KERNEL_INSTANTIATE __global__
#endif

#ifdef WIN32
#define ROCBLAS_KERNEL_ILF __device__ __attribute__((always_inline))
#else
#define ROCBLAS_KERNEL_ILF __device__ __attribute__((always_inline))
#endif
