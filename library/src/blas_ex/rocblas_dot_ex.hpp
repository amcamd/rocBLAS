/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas1/rocblas_dot.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <rocblas_int NB,
          bool        ISBATCHED,
          bool        CONJ,
          typename Tx,
          typename Ty  = Tx,
          typename Tr  = Ty,
          typename Tex = Tr>
rocblas_status dot_ex_typecasting(rocblas_handle __restrict__ handle,
                                  rocblas_int n,
                                  const void* __restrict__ x,
                                  rocblas_int    incx,
                                  rocblas_stride stride_x,
                                  const void* __restrict__ y,
                                  rocblas_int    incy,
                                  rocblas_stride stride_y,
                                  rocblas_int    batch_count,
                                  void* __restrict__ results,
                                  void* __restrict__ workspace)
{
    static constexpr rocblas_stride offset_0 = 0;
    if(ISBATCHED)
    {
        return rocblas_internal_dot_template<NB, CONJ>(handle,
                                                       n,
                                                       (const Tx* const*)x,
                                                       offset_0,
                                                       incx,
                                                       stride_x,
                                                       (const Ty* const*)y,
                                                       offset_0,
                                                       incy,
                                                       stride_y,
                                                       batch_count,
                                                       (Tr*)results,
                                                       (Tex*)workspace);
    }
    else
    {
        return rocblas_internal_dot_template<NB, CONJ>(handle,
                                                       n,
                                                       (const Tx*)x,
                                                       offset_0,
                                                       incx,
                                                       stride_x,
                                                       (const Ty*)y,
                                                       offset_0,
                                                       incy,
                                                       stride_y,
                                                       batch_count,
                                                       (Tr*)results,
                                                       (Tex*)workspace);
    }
}

template <rocblas_int NB, bool ISBATCHED, bool CONJ>
rocblas_status rocblas_dot_ex_template(rocblas_handle __restrict__ handle,
                                       rocblas_int n,
                                       const void* __restrict__ x,
                                       rocblas_datatype x_type,
                                       rocblas_int      incx,
                                       rocblas_stride   stride_x,
                                       const void* __restrict__ y,
                                       rocblas_datatype y_type,
                                       rocblas_int      incy,
                                       rocblas_stride   stride_y,
                                       rocblas_int      batch_count,
                                       void* __restrict__ results,
                                       rocblas_datatype result_type,
                                       rocblas_datatype execution_type,
                                       void* __restrict__ workspace)
{
#define DOT_EX_TYPECASTING_PARAM \
    handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, results, workspace

    if(x_type == rocblas_datatype_f16_r && y_type == rocblas_datatype_f16_r
       && result_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f16_r)
    {
        return dot_ex_typecasting<NB, ISBATCHED, CONJ, rocblas_half>(DOT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_bf16_r && y_type == rocblas_datatype_bf16_r
            && result_type == rocblas_datatype_bf16_r && execution_type == rocblas_datatype_f32_r)
    {
        return dot_ex_typecasting<NB,
                                  ISBATCHED,
                                  CONJ,
                                  rocblas_bfloat16,
                                  rocblas_bfloat16,
                                  rocblas_bfloat16,
                                  float>(DOT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f16_r && y_type == rocblas_datatype_f16_r
            && result_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f32_r)
    {
        return dot_ex_typecasting<NB,
                                  ISBATCHED,
                                  CONJ,
                                  rocblas_half,
                                  rocblas_half,
                                  rocblas_half,
                                  float>(DOT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_r && y_type == rocblas_datatype_f32_r
            && result_type == rocblas_datatype_f32_r && execution_type == rocblas_datatype_f32_r)
    {
        return dot_ex_typecasting<NB, ISBATCHED, CONJ, float>(DOT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_r && y_type == rocblas_datatype_f64_r
            && result_type == rocblas_datatype_f64_r && execution_type == rocblas_datatype_f64_r)
    {
        return dot_ex_typecasting<NB, ISBATCHED, CONJ, double>(DOT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_c && y_type == rocblas_datatype_f32_c
            && result_type == rocblas_datatype_f32_c && execution_type == rocblas_datatype_f32_c)
    {
        return dot_ex_typecasting<NB, ISBATCHED, CONJ, rocblas_float_complex>(
            DOT_EX_TYPECASTING_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_c && y_type == rocblas_datatype_f64_c
            && result_type == rocblas_datatype_f64_c && execution_type == rocblas_datatype_f64_c)
    {
        return dot_ex_typecasting<NB, ISBATCHED, CONJ, rocblas_double_complex>(
            DOT_EX_TYPECASTING_PARAM);
    }

    return rocblas_status_not_implemented;
}
