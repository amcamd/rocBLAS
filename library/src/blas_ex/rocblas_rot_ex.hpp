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

#include "../blas1/rocblas_rot.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <rocblas_int NB, bool ISBATCHED = false>
rocblas_status rocblas_rot_ex_template(rocblas_handle   handle,
                                       rocblas_int      n,
                                       void*            x,
                                       rocblas_datatype x_type,
                                       rocblas_int      incx,
                                       rocblas_stride   stride_x,
                                       void*            y,
                                       rocblas_datatype y_type,
                                       rocblas_int      incy,
                                       rocblas_stride   stride_y,
                                       const void*      c,
                                       const void*      s,
                                       rocblas_datatype cs_type,
                                       rocblas_int      batch_count,
                                       rocblas_datatype execution_type);
