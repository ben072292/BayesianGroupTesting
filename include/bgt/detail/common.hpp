#ifndef BGT_CORE_H_
#define BGT_CORE_H_

#include "bgt/detail/internal_types.hpp"
#include "bgt/detail/logging_macros.hpp"
#include "bgt/detail/mpi_checks.hpp"
#include "bgt/detail/state_encoding.hpp"
#include "bgt/detail/simd.hpp"
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <mpi.h>
#ifdef ENABLE_OMP
#include <omp.h>
#endif
#include <sstream>
#include <string>

inline MPI_Datatype bgt_state_mpi_type()
{
#if BGT_STATE_BITS == 8
    return MPI_UINT8_T;
#elif BGT_STATE_BITS == 16
    return MPI_UINT16_T;
#elif BGT_STATE_BITS == 32
    return MPI_UINT32_T;
#elif BGT_STATE_BITS == 64
    return MPI_UINT64_T;
#else
#error "Unsupported BGT_STATE_BITS value."
#endif
}

inline MPI_Datatype bgt_posterior_mpi_type()
{
#if BGT_POSTERIOR_BITS == 32
    return MPI_FLOAT;
#elif BGT_POSTERIOR_BITS == 64
    return MPI_DOUBLE;
#else
#error "Unsupported BGT_POSTERIOR_BITS value."
#endif
}

inline MPI_Datatype bgt_accumulator_mpi_type()
{
#if BGT_ACCUMULATOR_BITS == 32
    return MPI_FLOAT;
#elif BGT_ACCUMULATOR_BITS == 64
    return MPI_DOUBLE;
#else
#error "Unsupported BGT_ACCUMULATOR_BITS value."
#endif
}

inline MPI_Datatype bgt_statistic_mpi_type()
{
    return MPI_DOUBLE;
}

inline std::string get_curr_time()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
    return oss.str();
}

inline std::string hardware_config_summary()
{
    std::ostringstream oss;
    oss << "OpenMP is,";
#ifdef ENABLE_OMP
    char* omp_threads = getenv("OMP_NUM_THREADS");
    oss << "Enabled,omp_num_threads is set to," << std::stoi(omp_threads) << std::endl;
#else
    oss << "Disabled" << std::endl;
#endif

	oss << "SIMD is,";
#ifdef ENABLE_SIMD
	oss << "Enabled, Architecture support is " << bgt::simd::architecture_name();
#else
	oss << "Disabled";
#endif
    return oss.str();
}

#endif
