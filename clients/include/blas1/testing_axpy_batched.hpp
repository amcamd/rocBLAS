/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "bytes.hpp"
#include "cblas_interface.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"

/* ============================================================================================ */
template <typename T>
void testing_axpy_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_axpy_batched_fn
        = arg.fortran ? rocblas_axpy_batched<T, true> : rocblas_axpy_batched<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_int N = 100, incx = 1, incy = 1, batch_count = 2;

        device_vector<T> alpha_d(1), zero_d(1);

        const T alpha_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        // Allocate device memory
        device_batch_vector<T> dx(N, incx, batch_count);
        device_batch_vector<T> dy(N, incy, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(dx.memcheck());
        CHECK_DEVICE_ALLOCATION(dy.memcheck());

        EXPECT_ROCBLAS_STATUS(
            rocblas_axpy_batched_fn(
                nullptr, N, alpha, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count),
            rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_axpy_batched_fn(handle,
                                                      N,
                                                      nullptr,
                                                      dx.ptr_on_device(),
                                                      incx,
                                                      dy.ptr_on_device(),
                                                      incy,
                                                      batch_count),
                              rocblas_status_invalid_pointer);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            EXPECT_ROCBLAS_STATUS(
                rocblas_axpy_batched_fn(
                    handle, N, alpha, nullptr, incx, dy.ptr_on_device(), incy, batch_count),
                rocblas_status_invalid_pointer);

            EXPECT_ROCBLAS_STATUS(
                rocblas_axpy_batched_fn(
                    handle, N, alpha, dx.ptr_on_device(), incx, nullptr, incy, batch_count),
                rocblas_status_invalid_pointer);
        }

        // When N==0, alpha, X and Y can be nullptr without error
        EXPECT_ROCBLAS_STATUS(
            rocblas_axpy_batched_fn(handle, 0, nullptr, nullptr, incx, nullptr, incy, batch_count),
            rocblas_status_success);

        // When alpha==0, X and Y can be nullptr without error
        EXPECT_ROCBLAS_STATUS(
            rocblas_axpy_batched_fn(handle, N, zero, nullptr, incx, nullptr, incy, batch_count),
            rocblas_status_success);

        // When batch_count==0, alpha, X and Y can be nullptr without error
        EXPECT_ROCBLAS_STATUS(
            rocblas_axpy_batched_fn(handle, N, nullptr, nullptr, incx, nullptr, incy, 0),
            rocblas_status_success);
    }
}

template <typename T>
void testing_axpy_batched(const Arguments& arg)
{
    auto rocblas_axpy_batched_fn
        = arg.fortran ? rocblas_axpy_batched<T, true> : rocblas_axpy_batched<T, false>;

    rocblas_local_handle handle{arg};
    rocblas_int          N = arg.N, incx = arg.incx, incy = arg.incy, batch_count = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(
            rocblas_axpy_batched_fn(handle, N, nullptr, nullptr, incx, nullptr, incy, batch_count),
            rocblas_status_success);
        return;
    }

    rocblas_int abs_incy = incy > 0 ? incy : -incy;

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    host_batch_vector<T> hx(N, incx ? incx : 1, batch_count);
    host_batch_vector<T> hy_1(N, incy ? incy : 1, batch_count);
    host_batch_vector<T> hy_2(N, incy ? incy : 1, batch_count);
    host_batch_vector<T> hy_gold(N, incy ? incy : 1, batch_count);
    host_vector<T>       halpha(1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hx.memcheck());
    CHECK_HIP_ERROR(hy_1.memcheck());
    CHECK_HIP_ERROR(hy_2.memcheck());
    CHECK_HIP_ERROR(hy_gold.memcheck());

    // Allocate device memory
    device_batch_vector<T> dx(N, incx ? incx : 1, batch_count);
    device_batch_vector<T> dy(N, incy ? incy : 1, batch_count);
    device_vector<T>       dalpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(dalpha.memcheck());

    // Assign host alpha.
    halpha[0] = h_alpha;

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy_1, arg, rocblas_client_alpha_sets_nan, false, true);

    hy_gold.copy_from(hy_1);
    hy_2.copy_from(hy_1);

    double gpu_time_used, cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // Transfer host to device
        CHECK_HIP_ERROR(dx.transfer_from(hx));

        // Call routine with pointer mode on host.
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        // Call routine.
        CHECK_HIP_ERROR(dy.transfer_from(hy_1));

        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_axpy_batched_fn(
            handle, N, halpha, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count));
        handle.post_test(arg);

        // Transfer from device to host.
        CHECK_HIP_ERROR(hy_1.transfer_from(dy));

        // Pointer mode device.
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        // Call routine.
        CHECK_HIP_ERROR(dalpha.transfer_from(halpha));
        CHECK_HIP_ERROR(dy.transfer_from(hy_2));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_axpy_batched_fn(
            handle, N, dalpha, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count));
        handle.post_test(arg);

        // Transfer from device to host.
        CHECK_HIP_ERROR(hy_2.transfer_from(dy));

        // CPU BLAS
        {
            cpu_time_used = get_time_us_no_sync();

            // Compute the host solution.
            for(rocblas_int batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                cblas_axpy<T>(N, h_alpha, hx[batch_index], incx, hy_gold[batch_index], incy);
            }
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        //
        // Compare with with hsolution.
        //
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_1, batch_count);

            unit_check_general<T>(1, N, abs_incy, hy_gold, hy_2, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error_1
                = norm_check_general<T>('I', 1, N, abs_incy, hy_gold, hy_1, batch_count);
            rocblas_error_2
                = norm_check_general<T>('I', 1, N, abs_incy, hy_gold, hy_2, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        // Transfer from host to device.
        CHECK_HIP_ERROR(dy.transfer_from(hy_gold));

        // Cold.
        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocblas_axpy_batched_fn(handle,
                                    N,
                                    &h_alpha,
                                    dx.ptr_on_device(),
                                    incx,
                                    dy.ptr_on_device(),
                                    incy,
                                    batch_count);
        }

        // Transfer from host to device.
        CHECK_HIP_ERROR(dy.transfer_from(hy_gold));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocblas_axpy_batched_fn(handle,
                                    N,
                                    &h_alpha,
                                    dx.ptr_on_device(),
                                    incx,
                                    dy.ptr_on_device(),
                                    incy,
                                    batch_count);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_incy, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            axpy_gflop_count<T>(N),
            axpy_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}
