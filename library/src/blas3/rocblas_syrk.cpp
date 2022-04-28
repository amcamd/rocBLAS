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
#include "logging.hpp"
#include "rocblas_syrk_herk.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_syrk_name[] = "unknown";
    template <>
    constexpr char rocblas_syrk_name<float>[] = "rocblas_ssyrk";
    template <>
    constexpr char rocblas_syrk_name<double>[] = "rocblas_dsyrk";
    template <>
    constexpr char rocblas_syrk_name<rocblas_float_complex>[] = "rocblas_csyrk";
    template <>
    constexpr char rocblas_syrk_name<rocblas_double_complex>[] = "rocblas_zsyrk";

    template <typename T, typename U>
    rocblas_status rocblas_syrk_impl(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_int       n,
                                     rocblas_int       k,
                                     const U*          alpha,
                                     const T*          A,
                                     rocblas_int       lda,
                                     const U*          beta,
                                     T*                C,
                                     rocblas_int       ldc)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);

            if(layer_mode & rocblas_layer_mode_log_trace)
                log_trace(handle,
                          rocblas_syrk_name<T>,
                          uplo,
                          transA,
                          n,
                          k,
                          LOG_TRACE_SCALAR_VALUE(handle, alpha),
                          A,
                          lda,
                          LOG_TRACE_SCALAR_VALUE(handle, beta),
                          C,
                          ldc);

            if(layer_mode & rocblas_layer_mode_log_bench)
                log_bench(handle,
                          "./rocblas-bench -f syrk -r",
                          rocblas_precision_string<T>,
                          "--uplo",
                          uplo_letter,
                          "--transposeA",
                          transA_letter,
                          "-n",
                          n,
                          "-k",
                          k,
                          LOG_BENCH_SCALAR_VALUE(handle, alpha),
                          "--lda",
                          lda,
                          LOG_BENCH_SCALAR_VALUE(handle, beta),
                          "--ldc",
                          ldc);

            if(layer_mode & rocblas_layer_mode_log_profile)
                log_profile(handle,
                            rocblas_syrk_name<T>,
                            "uplo",
                            uplo_letter,
                            "transA",
                            transA_letter,
                            "N",
                            n,
                            "K",
                            k,
                            "lda",
                            lda,
                            "ldc",
                            ldc);
        }

        static constexpr rocblas_stride offset_C = 0, offset_A = 0;
        static constexpr rocblas_int    batch_count = 1;
        static constexpr rocblas_stride stride_C = 0, stride_A = 0;

        rocblas_status arg_status = rocblas_syrk_arg_check(handle,
                                                           uplo,
                                                           transA,
                                                           n,
                                                           k,
                                                           alpha,
                                                           A,
                                                           offset_A,
                                                           lda,
                                                           stride_A,
                                                           beta,
                                                           C,
                                                           offset_C,
                                                           ldc,
                                                           stride_C,
                                                           batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        return rocblas_internal_syrk_template(handle,
                                              uplo,
                                              transA,
                                              n,
                                              k,
                                              alpha,
                                              A,
                                              offset_A,
                                              lda,
                                              stride_A,
                                              beta,
                                              C,
                                              offset_C,
                                              ldc,
                                              stride_C,
                                              batch_count);
    }

}
/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, T_)                                                            \
    rocblas_status routine_name_(rocblas_handle    handle,                                 \
                                 rocblas_fill      uplo,                                   \
                                 rocblas_operation transA,                                 \
                                 rocblas_int       n,                                      \
                                 rocblas_int       k,                                      \
                                 const T_*         alpha,                                  \
                                 const T_*         A,                                      \
                                 rocblas_int       lda,                                    \
                                 const T_*         beta,                                   \
                                 T_*               C,                                      \
                                 rocblas_int       ldc)                                    \
    try                                                                                    \
    {                                                                                      \
        return rocblas_syrk_impl(handle, uplo, transA, n, k, alpha, A, lda, beta, C, ldc); \
    }                                                                                      \
    catch(...)                                                                             \
    {                                                                                      \
        return exception_to_rocblas_status();                                              \
    }

IMPL(rocblas_ssyrk, float);
IMPL(rocblas_dsyrk, double);
IMPL(rocblas_csyrk, rocblas_float_complex);
IMPL(rocblas_zsyrk, rocblas_double_complex);

#undef IMPL

} // extern "C"
