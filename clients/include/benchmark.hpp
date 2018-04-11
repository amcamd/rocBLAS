/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#pragma once
#ifndef _ROCBLAS_BENCHMARK_HPP_
#define _ROCBLAS_BENCHMARK_HPP_

#include "rocblas.h"
#include "handle.h"

#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>

#include <iostream>
#include <memory>
#include <string>
#include <array>
#include <algorithm>

/*

This header-only library can be used to run an experiment, which calls a function and returns a
runtime, and to accumulate statistics over many experiments.

Benchmarking modern CPUs and GPUs may be difficult. Results may be noisy due to:
 - cpu frequency scaling
     sudo cpupower frequency-set --governor performance		#before test
     sudo cpupower frequency-set --governor powersave		#after test
 - gpu frequency scaling
 - competing for cache and bandwidth with other processes
 - interrupts, kernel time, and context switches by the scheduler
 - the use of C++ lambdas for these benchmarks has an overhead, which may be negligible relative to
memory transfer times

Because of these difficulties, the numbers may not be reproducable every time on every piece of
hardware.


Data members:
  my_lambda: the function to be benchmarked wrapped in a function delegate.
  num_experiments: number of run-times to record, >=1.
  num_iters_per_experiment: number of times to call the function in a loop. >=1.
  sleep_time_between_experiment: amount of time to sleep between experiments.
  num_iters_before_experiment: number of times to call the function just before performing the
experiment.



TODO
 - stabilize the api
   then update each testing_*.hpp
 - what is flops.h and flops.cpp for?
 - how about clients/banchmarks/perf_script/

*/

std::string get_cmd_output(std::string cmd)
{

// This may only work on posix-based systems
#ifdef __linux__
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    // std::cout<<pipe<<std::endl;
    while(!feof(pipe.get()))
    {
        if(fgets(buffer.data(), 128, pipe.get()) != nullptr)
        {
            result += buffer.data();
        }
    }
    return result;
#endif
}

// Unfortunately, it seems that there is no universal way to get system info. The format of system
// calls can change at any time.
std::string get_system_info()
{

// the following may only work on debian-based systems, which support apt-cache. Also, they require
// git to be run in the repository.
#ifdef __linux__
    std::string rocm_version = get_cmd_output("apt-cache show rocm-dev | grep Version");
    rocm_version             = rocm_version.substr(
        rocm_version.find(" ") + 1,
        std::string::npos); // should be like "Version: 1.7.137", so get part after space
    rocm_version.erase(std::remove(rocm_version.begin(), rocm_version.end(), '\n'),
                       rocm_version.end()); // remove newline

    std::string compiler_version = get_cmd_output("apt-cache show hcc | grep Version");
    compiler_version             = compiler_version.substr(
        compiler_version.find(" ") + 1,
        std::string::npos); // should be like "Version: 1.7.137", so get part after space
    compiler_version.erase(std::remove(compiler_version.begin(), compiler_version.end(), '\n'),
                           compiler_version.end()); // remove newline

    std::string hip_version = get_cmd_output("apt-cache show hip_base | grep Version");
    hip_version             = hip_version.substr(
        hip_version.find(" ") + 1,
        std::string::npos); // should be like "Version: 1.7.137", so get part after space
    hip_version.erase(std::remove(hip_version.begin(), hip_version.end(), '\n'),
                      hip_version.end()); // remove newline

    std::string git_commit_hash = get_cmd_output("git show | grep commit");
    git_commit_hash =
        git_commit_hash.substr(git_commit_hash.find(" ") + 1,
                               std::string::npos); // should be like "commit
                                                   // 1381e0c0c445bc5d055d20be7b0198b99242d169", so
                                                   // get part after space
    git_commit_hash.erase(std::remove(git_commit_hash.begin(), git_commit_hash.end(), '\n'),
                          git_commit_hash.end()); // remove newline

    stringstream ss;
    if(rocm_version != "")
        ss << "rocm-dev" << rocm_version;
    if(compiler_version != "")
        ss << "hcc" << compiler_version;
    if(hip_version != "")
        ss << "hip_base" << hip_version;
    if(git_commit_hash != "")
        ss << "git_commit_hash" << git_commit_hash;
    return ss.str();
#endif

    return "";
}

template <typename F, typename TIME>
struct Benchmark
{

    // init values
    F my_lambda;
    int num_iters_per_experiment;
    int num_iters_before_experiment;
    int num_experiments;
    int sleep_time_between_experiments;

    // statistical values
    std::vector<double> run_times;
    double run_time_avg;
    double run_time_min;
    double run_time_max;
    double run_time_std_dev;

    // methods
    Benchmark(F my_lambda,
              int num_iters_per_experiment,
              int num_experiments,
              int num_iters_before_experiment,
              int sleep_time_between_experiments)
        : my_lambda(my_lambda),
          num_iters_per_experiment(num_iters_per_experiment),
          num_experiments(num_experiments),
          num_iters_before_experiment(num_iters_before_experiment),
          sleep_time_between_experiments(sleep_time_between_experiments){};
    double timer();
    double timer_once();
    void accumulate_statistics();
    void approximate_num_iters(int target_run_time, bool verbose);
    void print_cout();
    void print_to_file(
        string filename, string labels, string data, Benchmark stats, hipDeviceProp_t props);
};

template <typename F, typename TIME>
void Benchmark<F, TIME>::print_to_file(
    string filename, string labels, string data, Benchmark stats, hipDeviceProp_t props)
{

    string device_name(props.name);
    // make name alpha-numeric
    device_name.resize(
        remove_if(device_name.begin(), device_name.end(), [](char x) { return !isalnum(x); }) -
        device_name.begin());
    // add extra data to filename
    std::string filename_full =
        "benchmarks_" + filename + "_" + device_name + "_" + get_system_info() + ".txt";

    // if file doesn't exist, create it and print header to it
    ifstream file_if_exists(filename_full);
    if(!file_if_exists.good())
    {
        ofstream of;
        of.open(filename_full, ios::out);
        of << labels << endl;
        of << data << endl;
        of.close();
    }
    else
    {
        file_if_exists.close();
        ofstream of;
        of.open(filename_full, ios::app);
        of << data << endl;
        of.close();
    }
}

template <typename F, typename TIME>
void Benchmark<F, TIME>::print_cout()
{
    std::cout << "BENCHMARK RESULTS" << std::endl;
    std::cout
        << "Ran " << num_experiments << " experiment(s). Each experiment called the function "
        << num_iters_per_experiment << " time(s). There were " << sleep_time_between_experiments
        << "ms of sleep between each experiment, and " << num_iters_before_experiment
        << " iterations just before each experiment. Computed statistics over all experiments:"
        << std::endl;
    std::cout << "  time per experiment:   min " << run_time_min << "  avg " << run_time_avg
              << "  max " << run_time_max << "   stddev " << run_time_std_dev << std::endl;
    std::cout << "  time per function call:   min " << run_time_min / num_iters_per_experiment
              << "  avg " << run_time_avg / num_iters_per_experiment << "  max "
              << run_time_max / num_iters_per_experiment << "   stddev "
              << run_time_std_dev / num_iters_per_experiment << std::endl;
}

/*
The function is called in a loop num_iteration times, and runtime is returned.
*/
template <typename F, typename TIME>
double Benchmark<F, TIME>::timer()
{
    for(int i = 0; i < num_iters_before_experiment; i++)
        my_lambda();
    hipDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    for(long n = 0; n < num_iters_per_experiment; n++)
        my_lambda();
    hipDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();
    return (double)std::chrono::duration_cast<TIME>(t2 - t1).count();
}

/*
The function is called once, and runtime is returned.
*/
template <typename F, typename TIME>
double Benchmark<F, TIME>::timer_once()
{
    for(int i = 0; i < num_iters_before_experiment; i++)
        my_lambda();
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    my_lambda();
    hipDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    return (double)std::chrono::duration_cast<TIME>(t2 - t1).count();
}

/*
For each experiment, recurds a runtime. Then computes statistics over all experiments.
*/
template <typename F, typename TIME>
void Benchmark<F, TIME>::accumulate_statistics()
{

    // init statistical values
    run_times.clear();
    run_time_avg     = -1;
    run_time_min     = -1;
    run_time_max     = -1;
    run_time_std_dev = -1;

    double total_run_time = 0;
    for(int n = 0; n < num_experiments; n++)
    {
        double cur_run_time;
        if(num_iters_per_experiment == 1)
            cur_run_time = timer_once();
        else
            cur_run_time = timer();
        run_times.push_back(cur_run_time);
        total_run_time += cur_run_time;
        if(run_time_min > cur_run_time || run_time_min == -1)
            run_time_min = cur_run_time;
        if(run_time_max < cur_run_time || run_time_max == -1)
            run_time_max = cur_run_time;
        if(sleep_time_between_experiments)
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_between_experiments));
    }
    run_time_avg     = total_run_time / num_experiments;
    run_time_std_dev = 0;
    for(double e : run_times)
        run_time_std_dev += (e - run_time_avg) * (e - run_time_avg);
    run_time_std_dev /= num_experiments;
}

/*
Estimates the number of iterations required to run a given time period. This is a naive
approximation and should be evaluated on each invocation, especially for short experiments when
hardware is "cold" or in "boost".
*/
template <typename F, typename TIME>
void Benchmark<F, TIME>::approximate_num_iters(int target_run_time, bool verbose)
{

    // if we have a target_run_time, then approximate number of iterations which run in
    // approximately target_run_time
    if(target_run_time > 0)
    {
        int growth_factor        = 2;
        num_iters_per_experiment = 1;
        if(verbose)
            std::cout << "----growing #iterations until exceeds " << target_run_time << " -----"
                      << std::endl;
        double current_run_time  = 0;
        double previous_run_time = 0;
        while(current_run_time < target_run_time)
        {
            previous_run_time = current_run_time;
            current_run_time  = timer();
            if(verbose)
                std::cout << current_run_time << " ms   " << num_iters_per_experiment << " iters"
                          << std::endl;
            num_iters_per_experiment *= growth_factor;
        }
        num_iters_per_experiment /= growth_factor;
        // interpolate overshoot to get reasonable num_iters
        if(num_iters_per_experiment > 2)
        {
            num_iters_per_experiment /= growth_factor;
            if(verbose)
                std::cout << "----interpolating overshoot----" << std::endl;
            if(verbose)
                std::cout << "num_iters_per_experiment:" << num_iters_per_experiment
                          << "  "
                             "num_iters_per_experiment*(target_run_time-previous_run_time)/"
                             "(current_run_time-previous_run_time):"
                          << num_iters_per_experiment * (target_run_time - previous_run_time) /
                                 (current_run_time - previous_run_time)
                          << std::endl;
            num_iters_per_experiment += num_iters_per_experiment *
                                        (target_run_time - previous_run_time) /
                                        (current_run_time - previous_run_time);
            if(verbose)
                std::cout << "To match " << target_run_time
                          << " runtime per experiment, the heuristic estimated "
                          << num_iters_per_experiment << " iterations per experiment." << std::endl;
        }
        // estimate milliseconds
        if(verbose)
            std::cout << "----realized runtime----" << std::endl;
        if(verbose)
            std::cout << timer() << std::endl;
    }
}

#endif /* _ROCBLAS_BENCHMARK_HPP_ */
