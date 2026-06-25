cmake_minimum_required(VERSION 3.15)
project(BGT VERSION 1.0 DESCRIPTION "Unified Bayesian Group Testing Framework")
set(CMAKE_VERBOSE_MAKEFILE ON)
include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

set(BGT_ENABLE_CUDA "AUTO" CACHE STRING "Build the CUDA backend: AUTO, ON, or OFF")
set(BGT_ENABLE_NCCL "AUTO" CACHE STRING "Build NCCL-backed CUDA collectives: AUTO, ON, or OFF")
set(BGT_ENABLE_NCCL_GIN "AUTO" CACHE STRING "Build NCCL GIN device-initiated CUDA kernels: AUTO, ON, or OFF")
option(BGT_ENABLE_SIMD "Enable CPU SIMD kernels" ON)
option(BGT_ENABLE_OPENMP "Enable OpenMP CPU kernels" OFF)
option(BGT_ENABLE_NATIVE_CPU "Build native CPU kernels with -march=native when supported" ON)
option(BGT_ENABLE_FAST_MATH "Build kernels with fast floating-point math flags" ON)
option(BGT_MINIMAL_LOGGING "Compile out debug and trace logging macros" OFF)
set(BGT_PROFILE_BACKEND "none" CACHE STRING "Profiling backend: none or caliper")
set(BGT_OPTIMIZATION_LEVEL "release" CACHE STRING "CPU optimization level: debug, release, or aggressive")
option(BGT_BUILD_PYTHON "Build the nanobind Python extension" OFF)
set(BGT_BUILD_EXAMPLES_DEFAULT ON)
if(BGT_BUILD_PYTHON)
    set(BGT_BUILD_EXAMPLES_DEFAULT OFF)
endif()
option(BGT_BUILD_EXAMPLES "Build command-line examples and benchmark executables" ${BGT_BUILD_EXAMPLES_DEFAULT})
set_property(CACHE BGT_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE BGT_ENABLE_NCCL PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE BGT_ENABLE_NCCL_GIN PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE BGT_OPTIMIZATION_LEVEL PROPERTY STRINGS debug release aggressive)
set_property(CACHE BGT_PROFILE_BACKEND PROPERTY STRINGS none caliper)
string(TOLOWER "${BGT_OPTIMIZATION_LEVEL}" BGT_OPTIMIZATION_LEVEL)
if(NOT BGT_OPTIMIZATION_LEVEL STREQUAL "debug" AND
   NOT BGT_OPTIMIZATION_LEVEL STREQUAL "release" AND
   NOT BGT_OPTIMIZATION_LEVEL STREQUAL "aggressive")
    message(FATAL_ERROR "BGT_OPTIMIZATION_LEVEL must be debug, release, or aggressive.")
endif()
string(TOLOWER "${BGT_PROFILE_BACKEND}" BGT_PROFILE_BACKEND)
if(NOT BGT_PROFILE_BACKEND STREQUAL "none" AND
   NOT BGT_PROFILE_BACKEND STREQUAL "caliper")
    message(FATAL_ERROR "BGT_PROFILE_BACKEND must be none or caliper.")
endif()
string(TOUPPER "${BGT_ENABLE_CUDA}" BGT_ENABLE_CUDA_REQUEST)
if(NOT BGT_ENABLE_CUDA_REQUEST STREQUAL "AUTO" AND
   NOT BGT_ENABLE_CUDA_REQUEST STREQUAL "ON" AND
   NOT BGT_ENABLE_CUDA_REQUEST STREQUAL "OFF")
    message(FATAL_ERROR "BGT_ENABLE_CUDA must be AUTO, ON, or OFF.")
endif()
string(TOUPPER "${BGT_ENABLE_NCCL}" BGT_ENABLE_NCCL_REQUEST)
if(NOT BGT_ENABLE_NCCL_REQUEST STREQUAL "AUTO" AND
   NOT BGT_ENABLE_NCCL_REQUEST STREQUAL "ON" AND
   NOT BGT_ENABLE_NCCL_REQUEST STREQUAL "OFF")
    message(FATAL_ERROR "BGT_ENABLE_NCCL must be AUTO, ON, or OFF.")
endif()
string(TOUPPER "${BGT_ENABLE_NCCL_GIN}" BGT_ENABLE_NCCL_GIN_REQUEST)
if(NOT BGT_ENABLE_NCCL_GIN_REQUEST STREQUAL "AUTO" AND
   NOT BGT_ENABLE_NCCL_GIN_REQUEST STREQUAL "ON" AND
   NOT BGT_ENABLE_NCCL_GIN_REQUEST STREQUAL "OFF")
    message(FATAL_ERROR "BGT_ENABLE_NCCL_GIN must be AUTO, ON, or OFF.")
endif()

set(BGT_STATE_BITS "32" CACHE STRING "State encoding width in bits: 8, 16, 32, or 64")
set_property(CACHE BGT_STATE_BITS PROPERTY STRINGS 8 16 32 64)
if(NOT BGT_STATE_BITS STREQUAL "8" AND
   NOT BGT_STATE_BITS STREQUAL "16" AND
   NOT BGT_STATE_BITS STREQUAL "32" AND
   NOT BGT_STATE_BITS STREQUAL "64")
    message(FATAL_ERROR "BGT_STATE_BITS must be 8, 16, 32, or 64.")
endif()

set(BGT_POSTERIOR_PRECISION "float64" CACHE STRING "Posterior probability storage precision: float32 or float64")
set_property(CACHE BGT_POSTERIOR_PRECISION PROPERTY STRINGS float32 float64)
string(TOLOWER "${BGT_POSTERIOR_PRECISION}" BGT_POSTERIOR_PRECISION)
if(BGT_POSTERIOR_PRECISION STREQUAL "float32")
    set(BGT_POSTERIOR_BITS 32)
elseif(BGT_POSTERIOR_PRECISION STREQUAL "float64")
    set(BGT_POSTERIOR_BITS 64)
else()
    message(FATAL_ERROR "BGT_POSTERIOR_PRECISION must be float32 or float64.")
endif()

set(BGT_ACCUMULATOR_PRECISION "float64" CACHE STRING "Internal probability reduction precision: float32 or float64")
set_property(CACHE BGT_ACCUMULATOR_PRECISION PROPERTY STRINGS float32 float64)
string(TOLOWER "${BGT_ACCUMULATOR_PRECISION}" BGT_ACCUMULATOR_PRECISION)
if(BGT_ACCUMULATOR_PRECISION STREQUAL "float32")
    set(BGT_ACCUMULATOR_BITS 32)
elseif(BGT_ACCUMULATOR_PRECISION STREQUAL "float64")
    set(BGT_ACCUMULATOR_BITS 64)
else()
    message(FATAL_ERROR "BGT_ACCUMULATOR_PRECISION must be float32 or float64.")
endif()

set(BGT_MAJOR 0)
set(BGT_MINOR 0)
set(BGT_PATCH 1)
set(BGT_SUFFIX "alpha")
set(BGT_VERSION_CODE ${BGT_MAJOR}${BGT_MINOR}${BGT_PATCH})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
find_package(MPI REQUIRED COMPONENTS C CXX)
set(BGT_MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS} ${MPI_CXX_COMPILER_INCLUDE_DIRS})
if(MPI_CXX_COMPILER)
    get_filename_component(BGT_MPI_COMPILER_DIR "${MPI_CXX_COMPILER}" DIRECTORY)
    get_filename_component(BGT_MPI_PREFIX "${BGT_MPI_COMPILER_DIR}" DIRECTORY)
    if(EXISTS "${BGT_MPI_PREFIX}/include/mpi.h")
        list(APPEND BGT_MPI_INCLUDE_DIRS "${BGT_MPI_PREFIX}/include")
    endif()
endif()
list(REMOVE_DUPLICATES BGT_MPI_INCLUDE_DIRS)
find_package(OpenMP)
if(APPLE AND NOT OpenMP_CXX_FOUND)
    foreach(BGT_LIBOMP_PREFIX /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if(EXISTS "${BGT_LIBOMP_PREFIX}/include/omp.h" AND EXISTS "${BGT_LIBOMP_PREFIX}/lib/libomp.dylib")
            add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
            target_compile_options(OpenMP::OpenMP_CXX INTERFACE -Xpreprocessor -fopenmp)
            target_include_directories(OpenMP::OpenMP_CXX INTERFACE "${BGT_LIBOMP_PREFIX}/include")
            target_link_libraries(OpenMP::OpenMP_CXX INTERFACE "${BGT_LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_CXX_FOUND TRUE)
            message(STATUS "Using Homebrew libomp from ${BGT_LIBOMP_PREFIX}")
            break()
        endif()
    endforeach()
endif()
if(BGT_ENABLE_OPENMP AND NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "BGT_ENABLE_OPENMP=ON requested, but OpenMP CXX support was not found.")
endif()
set(BGT_CUDA_ENABLED OFF)
if(BGT_ENABLE_CUDA_REQUEST STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(BGT_CUDA_ENABLED ON)
    else()
        message(STATUS "CUDA compiler not found; building CPU/parallel fallback only.")
    endif()
elseif(BGT_ENABLE_CUDA_REQUEST STREQUAL "ON")
    set(BGT_CUDA_ENABLED ON)
endif()

if(BGT_CUDA_ENABLED)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(BGT_ENABLE_CUDA ON)
else()
    set(BGT_ENABLE_CUDA OFF)
endif()

set(BGT_NCCL_ENABLED OFF)
set(BGT_NCCL_GIN_ENABLED OFF)
if(BGT_ENABLE_NCCL_REQUEST STREQUAL "ON" AND NOT BGT_ENABLE_CUDA)
    message(FATAL_ERROR "BGT_ENABLE_NCCL=ON requires BGT_ENABLE_CUDA=ON or AUTO with a CUDA compiler.")
endif()
if(BGT_ENABLE_CUDA AND NOT BGT_ENABLE_NCCL_REQUEST STREQUAL "OFF")
    find_path(BGT_NCCL_INCLUDE_DIR NAMES nccl.h)
    find_library(BGT_NCCL_LIBRARY NAMES nccl)
    if(BGT_NCCL_INCLUDE_DIR AND BGT_NCCL_LIBRARY)
        set(BGT_NCCL_ENABLED ON)
        set(BGT_ENABLE_NCCL ON)
        message(STATUS "NCCL collectives enabled with ${BGT_NCCL_LIBRARY}")
    elseif(BGT_ENABLE_NCCL_REQUEST STREQUAL "ON")
        message(FATAL_ERROR "BGT_ENABLE_NCCL=ON requires nccl.h and libnccl.")
    else()
        set(BGT_ENABLE_NCCL OFF)
        message(STATUS "NCCL headers/libraries not found; building CUDA provider without NCCL collectives.")
    endif()
else()
    set(BGT_ENABLE_NCCL OFF)
endif()

if(BGT_ENABLE_NCCL_GIN_REQUEST STREQUAL "ON" AND NOT BGT_NCCL_ENABLED)
    message(FATAL_ERROR "BGT_ENABLE_NCCL_GIN=ON requires BGT_ENABLE_NCCL=ON or AUTO with NCCL found.")
endif()
if(BGT_NCCL_ENABLED AND NOT BGT_ENABLE_NCCL_GIN_REQUEST STREQUAL "OFF")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.2)
        if(BGT_ENABLE_NCCL_GIN_REQUEST STREQUAL "ON")
            message(FATAL_ERROR "BGT_ENABLE_NCCL_GIN=ON requires CUDA 12.2 or newer.")
        endif()
    else()
        find_path(BGT_NCCL_DEVICE_INCLUDE_DIR NAMES nccl_device.h HINTS ${BGT_NCCL_INCLUDE_DIR})
        if(BGT_NCCL_DEVICE_INCLUDE_DIR)
            file(READ "${BGT_NCCL_DEVICE_INCLUDE_DIR}/nccl_device.h" BGT_NCCL_DEVICE_HEADER)
            if(BGT_NCCL_DEVICE_HEADER MATCHES "NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER")
                set(BGT_NCCL_GIN_ENABLED ON)
                set(BGT_ENABLE_NCCL_GIN ON)
                message(STATUS "NCCL GIN support enabled")
            elseif(BGT_ENABLE_NCCL_GIN_REQUEST STREQUAL "ON")
                message(FATAL_ERROR "BGT_ENABLE_NCCL_GIN=ON requires the NCCL Device API variant currently used by this implementation.")
            else()
                set(BGT_ENABLE_NCCL_GIN OFF)
                message(STATUS "NCCL device API headers found, but this GIN implementation targets a different NCCL Device API variant; building without NCCL GIN.")
            endif()
        elseif(BGT_ENABLE_NCCL_GIN_REQUEST STREQUAL "ON")
            message(FATAL_ERROR "BGT_ENABLE_NCCL_GIN=ON requires nccl_device.h.")
        else()
            set(BGT_ENABLE_NCCL_GIN OFF)
            message(STATUS "NCCL device API headers not found; building CUDA provider without NCCL GIN.")
        endif()
    endif()
else()
    set(BGT_ENABLE_NCCL_GIN OFF)
endif()

if(BGT_OPTIMIZATION_LEVEL STREQUAL "debug")
    set(BGT_OPTIMIZATION_LEVEL_CODE 0)
elseif(BGT_OPTIMIZATION_LEVEL STREQUAL "release")
    set(BGT_OPTIMIZATION_LEVEL_CODE 1)
else()
    set(BGT_OPTIMIZATION_LEVEL_CODE 2)
endif()
set(BGT_ENABLE_CALIPER_PROFILING OFF)
set(BGT_ENABLE_CALIPER OFF)
set(BGT_PROFILE_BACKEND_CODE 0)
if(BGT_PROFILE_BACKEND STREQUAL "caliper")
    set(BGT_PROFILE_BACKEND_CODE 1)
    set(BGT_ENABLE_CALIPER_PROFILING ON)
    set(BGT_ENABLE_CALIPER ON)
    find_package(caliper QUIET CONFIG)
    if(TARGET caliper)
        set(BGT_CALIPER_TARGET caliper)
    else()
        find_library(BGT_CALIPER_LIBRARY NAMES caliper)
        find_path(BGT_CALIPER_INCLUDE_DIR NAMES caliper/cali.h)
        if(NOT BGT_CALIPER_LIBRARY OR NOT BGT_CALIPER_INCLUDE_DIR)
            message(FATAL_ERROR "BGT_PROFILE_BACKEND=caliper requires Caliper. Install Caliper or set BGT_PROFILE_BACKEND=none.")
        endif()
    endif()
endif()

configure_file(${PROJECT_SOURCE_DIR}/include/bgt/config.hpp.in ${PROJECT_BINARY_DIR}/include/bgt/config.hpp @ONLY)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/BGTConfig.cmake.in
    ${PROJECT_BINARY_DIR}/BGTConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BGT
)
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/BGTConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

set(BGT_COMMON_CXX_COMPILE_OPTIONS -Wall -fvisibility=hidden)
if(BGT_OPTIMIZATION_LEVEL STREQUAL "debug")
    list(APPEND BGT_COMMON_CXX_COMPILE_OPTIONS -O0)
elseif(BGT_OPTIMIZATION_LEVEL STREQUAL "release")
    list(APPEND BGT_COMMON_CXX_COMPILE_OPTIONS -O3)
else()
    list(APPEND BGT_COMMON_CXX_COMPILE_OPTIONS -O3)
endif()
if(BGT_ENABLE_NATIVE_CPU)
    list(APPEND BGT_COMMON_CXX_COMPILE_OPTIONS -march=native)
endif()
if(BGT_ENABLE_FAST_MATH)
    list(APPEND BGT_COMMON_CXX_COMPILE_OPTIONS -ffast-math)
endif()

set(EXAMPLE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/examples")
set(BENCHMARK_SOURCE_DIR "${PROJECT_SOURCE_DIR}/benchmarks")
set(TEST_SOURCE_DIR "${PROJECT_SOURCE_DIR}/tests/cpp")
set(CORE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/core")
set(LATTICE_SOURCE_DIR "${CORE_SOURCE_DIR}/model")
set(RUNTIME_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/runtime")
set(CPU_KERNEL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/kernels/cpu")
set(GPU_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/kernels/cuda")
set(TREE_SOURCE_DIR "${CORE_SOURCE_DIR}/tree")
set(UTIL_SOURCE_DIR "${CORE_SOURCE_DIR}/util")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(BGT_CORE_SOURCES
    ${TREE_SOURCE_DIR}/distributed_tree.cpp
    ${TREE_SOURCE_DIR}/fusion_tree.cpp
    ${TREE_SOURCE_DIR}/global_partial_tree.cpp
    ${TREE_SOURCE_DIR}/global_tree.cpp
    ${TREE_SOURCE_DIR}/global_tree_intra.cpp
    ${TREE_SOURCE_DIR}/tree.cpp
    ${UTIL_SOURCE_DIR}/tree_stat.cpp
    ${UTIL_SOURCE_DIR}/tree_symm.cpp
    ${UTIL_SOURCE_DIR}/tree_trim.cpp
    ${CORE_SOURCE_DIR}/lattice_api.cpp
    ${LATTICE_SOURCE_DIR}/posterior_buffer.cpp
    ${LATTICE_SOURCE_DIR}/shrink_plan.cpp
    ${LATTICE_SOURCE_DIR}/dilution.cpp
    ${LATTICE_SOURCE_DIR}/lattice_base.cpp
    ${LATTICE_SOURCE_DIR}/replicated_lattice.cpp
    ${LATTICE_SOURCE_DIR}/distributed_lattice.cpp
    ${LATTICE_SOURCE_DIR}/lattice_factory.cpp
    ${CPU_KERNEL_SOURCE_DIR}/bbpa.cpp
    ${CPU_KERNEL_SOURCE_DIR}/bbpa_simd.cpp
    ${RUNTIME_SOURCE_DIR}/status.cpp
    ${RUNTIME_SOURCE_DIR}/logging.cpp
    ${RUNTIME_SOURCE_DIR}/performance.cpp
    ${RUNTIME_SOURCE_DIR}/progress.cpp
    ${RUNTIME_SOURCE_DIR}/kernel_registry.cpp
    ${RUNTIME_SOURCE_DIR}/jit_compiler.cpp
    ${RUNTIME_SOURCE_DIR}/simulation_helpers.cpp
    ${RUNTIME_SOURCE_DIR}/reporting.cpp
    ${RUNTIME_SOURCE_DIR}/providers.cpp
    ${RUNTIME_SOURCE_DIR}/simulation_runner.cpp
    ${RUNTIME_SOURCE_DIR}/parallel_simulation.cpp
)
if(BGT_ENABLE_CUDA)
    set(GPU_SOURCES ${GPU_SOURCE_DIR}/cuda_provider.cu)
else()
    set(GPU_SOURCES "")
endif()

function(add_bgt_library TARGET_NAME)
    # Shared library
    add_library(${TARGET_NAME}_shared SHARED
        ${BGT_CORE_SOURCES}
        ${GPU_SOURCES}
    )

    # Static library
    add_library(${TARGET_NAME}_static STATIC
        ${BGT_CORE_SOURCES}
        ${GPU_SOURCES}
    )

    foreach(BGT_CXX_OPTION IN LISTS BGT_COMMON_CXX_COMPILE_OPTIONS)
        target_compile_options(${TARGET_NAME}_shared PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${BGT_CXX_OPTION}>)
        target_compile_options(${TARGET_NAME}_static PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${BGT_CXX_OPTION}>)
    endforeach()
    target_include_directories(${TARGET_NAME}_shared PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_include_directories(${TARGET_NAME}_static PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(${TARGET_NAME}_shared PRIVATE MPI::MPI_CXX)
    target_link_libraries(${TARGET_NAME}_static PRIVATE MPI::MPI_CXX)
    target_include_directories(${TARGET_NAME}_shared PRIVATE ${BGT_MPI_INCLUDE_DIRS})
    target_include_directories(${TARGET_NAME}_static PRIVATE ${BGT_MPI_INCLUDE_DIRS})
    if(BGT_ENABLE_CUDA)
        target_compile_definitions(${TARGET_NAME}_shared PRIVATE BGT_ENABLE_CUDA)
        target_compile_definitions(${TARGET_NAME}_static PRIVATE BGT_ENABLE_CUDA)
        target_include_directories(${TARGET_NAME}_shared PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_include_directories(${TARGET_NAME}_static PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${TARGET_NAME}_shared PRIVATE CUDA::cudart)
        target_link_libraries(${TARGET_NAME}_static PRIVATE CUDA::cudart)
        set_target_properties(${TARGET_NAME}_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
        set_target_properties(${TARGET_NAME}_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
    endif()
    if(BGT_NCCL_ENABLED)
        target_compile_definitions(${TARGET_NAME}_shared PRIVATE BGT_ENABLE_NCCL)
        target_compile_definitions(${TARGET_NAME}_static PRIVATE BGT_ENABLE_NCCL)
        target_include_directories(${TARGET_NAME}_shared PRIVATE ${BGT_NCCL_INCLUDE_DIR})
        target_include_directories(${TARGET_NAME}_static PRIVATE ${BGT_NCCL_INCLUDE_DIR})
        target_link_libraries(${TARGET_NAME}_shared PRIVATE ${BGT_NCCL_LIBRARY})
        target_link_libraries(${TARGET_NAME}_static PRIVATE ${BGT_NCCL_LIBRARY})
    endif()
    if(BGT_NCCL_GIN_ENABLED)
        target_compile_definitions(${TARGET_NAME}_shared PRIVATE BGT_ENABLE_NCCL_GIN)
        target_compile_definitions(${TARGET_NAME}_static PRIVATE BGT_ENABLE_NCCL_GIN)
        target_include_directories(${TARGET_NAME}_shared PRIVATE ${BGT_NCCL_DEVICE_INCLUDE_DIR})
        target_include_directories(${TARGET_NAME}_static PRIVATE ${BGT_NCCL_DEVICE_INCLUDE_DIR})
    endif()

    # Set output name and properties
    set_target_properties(${TARGET_NAME}_shared PROPERTIES OUTPUT_NAME ${TARGET_NAME})
    set_target_properties(${TARGET_NAME}_static PROPERTIES OUTPUT_NAME ${TARGET_NAME})
    set_target_properties(${TARGET_NAME}_shared PROPERTIES VERSION ${PROJECT_VERSION})
    set_target_properties(${TARGET_NAME}_shared PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
    set_target_properties(${TARGET_NAME}_static PROPERTIES VERSION ${PROJECT_VERSION})
    
    # Install targets
    install(TARGETS ${TARGET_NAME}_shared
        EXPORT BGTTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(TARGETS ${TARGET_NAME}_static
        EXPORT BGTTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/bgt DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")
    install(FILES ${PROJECT_BINARY_DIR}/include/bgt/config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bgt)

    # Apply compile definitions if provided
    set(BGT_TARGET_DEFINITIONS)
    if(BGT_ENABLE_SIMD)
        list(APPEND BGT_TARGET_DEFINITIONS ENABLE_SIMD)
    endif()
    if(BGT_ENABLE_OPENMP)
        list(APPEND BGT_TARGET_DEFINITIONS ENABLE_OMP)
    endif()
    if(BGT_MINIMAL_LOGGING)
        list(APPEND BGT_TARGET_DEFINITIONS BGT_MINIMAL_LOGGING)
    endif()
    if(BGT_NCCL_ENABLED)
        list(APPEND BGT_TARGET_DEFINITIONS BGT_ENABLE_NCCL)
    endif()
    if(BGT_NCCL_GIN_ENABLED)
        list(APPEND BGT_TARGET_DEFINITIONS BGT_ENABLE_NCCL_GIN)
    endif()
    if(BGT_PROFILE_BACKEND STREQUAL "caliper")
        list(APPEND BGT_TARGET_DEFINITIONS BGT_ENABLE_CALIPER_PROFILING)
    endif()
    if(ARGN)
        list(APPEND BGT_TARGET_DEFINITIONS ${ARGN})
    endif()
    target_compile_definitions(${TARGET_NAME}_shared PUBLIC ${BGT_TARGET_DEFINITIONS})
    target_compile_definitions(${TARGET_NAME}_static PUBLIC ${BGT_TARGET_DEFINITIONS})
    if(BGT_ENABLE_OPENMP)
        target_link_libraries(${TARGET_NAME}_shared PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(${TARGET_NAME}_static PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(BGT_ENABLE_CALIPER)
        if(BGT_CALIPER_TARGET)
            target_link_libraries(${TARGET_NAME}_shared PUBLIC ${BGT_CALIPER_TARGET})
            target_link_libraries(${TARGET_NAME}_static PUBLIC ${BGT_CALIPER_TARGET})
        else()
            target_include_directories(${TARGET_NAME}_shared PUBLIC ${BGT_CALIPER_INCLUDE_DIR})
            target_include_directories(${TARGET_NAME}_static PUBLIC ${BGT_CALIPER_INCLUDE_DIR})
            target_link_libraries(${TARGET_NAME}_shared PUBLIC ${BGT_CALIPER_LIBRARY})
            target_link_libraries(${TARGET_NAME}_static PUBLIC ${BGT_CALIPER_LIBRARY})
        endif()
    endif()
endfunction()

add_bgt_library(bgt)

if(BGT_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(_core ${PROJECT_SOURCE_DIR}/src/bindings/python/bgt_module.cpp)
    target_link_libraries(_core PRIVATE bgt_static)
    target_include_directories(_core PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
    install(TARGETS _core LIBRARY DESTINATION bayesian_group_testing)
endif()

function(create_executable TARGET_NAME SOURCE_FILE LIBRARY_NAME SHOULD_INSTALL)
    add_executable(${TARGET_NAME} ${SOURCE_FILE} ${ARGN})
    add_dependencies(${TARGET_NAME} ${LIBRARY_NAME})
    target_link_libraries(${TARGET_NAME} ${LIBRARY_NAME} MPI::MPI_CXX)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include)

    # Check if the executable should be installed
    if(SHOULD_INSTALL)
        install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endfunction()


if(BGT_BUILD_EXAMPLES)
    create_executable(bgt_simulate ${EXAMPLE_SOURCE_DIR}/run_simulation.cpp bgt_static TRUE)
    create_executable(model_manipulation_benchmark ${EXAMPLE_SOURCE_DIR}/run_model_manipulation_benchmark.cpp bgt_static TRUE ${BENCHMARK_SOURCE_DIR}/model_manipulation_benchmark.cpp)
    create_executable(BBPA_benchmark_intra ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark_intra.cpp bgt_static TRUE ${BENCHMARK_SOURCE_DIR}/BBPA_benchmark_intra.cpp)
    create_executable(BBPA_benchmark ${EXAMPLE_SOURCE_DIR}/run_BBPA_benchmark.cpp bgt_static TRUE ${BENCHMARK_SOURCE_DIR}/BBPA_benchmark.cpp)
    create_executable(selector_roofline_benchmark ${BENCHMARK_SOURCE_DIR}/selector_roofline_benchmark.cpp bgt_static TRUE)
    create_executable(experiment_selection_benchmark ${BENCHMARK_SOURCE_DIR}/experiment_selection_benchmark.cpp bgt_static TRUE)
endif()

if(BUILD_TESTING)
    function(bgt_add_test TARGET_NAME SOURCE_FILE LABELS)
        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        target_link_libraries(${TARGET_NAME} PRIVATE bgt_static MPI::MPI_CXX)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${PROJECT_BINARY_DIR}/include
            ${TEST_SOURCE_DIR})
        target_compile_definitions(${TARGET_NAME} PRIVATE BGT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
        add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
        set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "${LABELS}")
    endfunction()

    function(bgt_add_unit_test TARGET_NAME SOURCE_FILE)
        bgt_add_test(${TARGET_NAME} ${SOURCE_FILE} "unit")
    endfunction()

    function(bgt_add_integration_test TARGET_NAME SOURCE_FILE)
        bgt_add_test(${TARGET_NAME} ${SOURCE_FILE} "integration")
    endfunction()

    function(bgt_add_regression_test TARGET_NAME SOURCE_FILE)
        bgt_add_test(${TARGET_NAME} ${SOURCE_FILE} "regression")
    endfunction()

    function(bgt_add_parallel_test TARGET_NAME SOURCE_FILE)
        add_executable(${TARGET_NAME} ${SOURCE_FILE})
        target_link_libraries(${TARGET_NAME} PRIVATE bgt_static MPI::MPI_CXX)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${PROJECT_BINARY_DIR}/include
            ${TEST_SOURCE_DIR})
        add_test(NAME ${TARGET_NAME}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:${TARGET_NAME}>)
        set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "integration;parallel")
        add_test(NAME ${TARGET_NAME}_uneven_partition
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 $<TARGET_FILE:${TARGET_NAME}>)
        set_tests_properties(${TARGET_NAME}_uneven_partition PROPERTIES LABELS "integration;parallel")
    endfunction()

    function(bgt_add_cuda_test TARGET_NAME SOURCE_FILE)
        bgt_add_test(${TARGET_NAME} ${SOURCE_FILE} "integration;cuda")
    endfunction()

    bgt_add_unit_test(model_math_test ${TEST_SOURCE_DIR}/unit/model_math_test.cpp)
    bgt_add_unit_test(lattice_api_test ${TEST_SOURCE_DIR}/unit/lattice_api_test.cpp)
    bgt_add_unit_test(dilution_test ${TEST_SOURCE_DIR}/unit/dilution_test.cpp)
    bgt_add_unit_test(compile_options_test ${TEST_SOURCE_DIR}/unit/compile_options_test.cpp)
    bgt_add_unit_test(status_logging_test ${TEST_SOURCE_DIR}/unit/status_logging_test.cpp)
    bgt_add_unit_test(diagnostics_scan_test ${TEST_SOURCE_DIR}/unit/diagnostics_scan_test.cpp)
    bgt_add_unit_test(shrink_plan_test ${TEST_SOURCE_DIR}/unit/shrink_plan_test.cpp)
    bgt_add_integration_test(local_simulation_test ${TEST_SOURCE_DIR}/integration/local_simulation_test.cpp)
    bgt_add_integration_test(simulation_progress_test ${TEST_SOURCE_DIR}/integration/simulation_progress_test.cpp)
    bgt_add_integration_test(runtime_jit_test ${TEST_SOURCE_DIR}/integration/runtime_jit_test.cpp)
    bgt_add_parallel_test(parallel_tree_modes_test ${TEST_SOURCE_DIR}/integration/parallel_tree_modes_test.cpp)
    bgt_add_parallel_test(parallel_shrink_property_test ${TEST_SOURCE_DIR}/integration/parallel_shrink_property_test.cpp)
    bgt_add_regression_test(binary_tree_regression_test ${TEST_SOURCE_DIR}/regression/binary_tree_regression_test.cpp)
    bgt_add_regression_test(shrink_projection_regression_test ${TEST_SOURCE_DIR}/regression/shrink_projection_regression_test.cpp)
    bgt_add_regression_test(shrink_property_regression_test ${TEST_SOURCE_DIR}/regression/shrink_property_regression_test.cpp)
    bgt_add_regression_test(no_shrink_tree_reference_test ${TEST_SOURCE_DIR}/regression/no_shrink_tree_reference_test.cpp)

    if(BGT_BUILD_EXAMPLES)
        add_test(NAME bgt_simulate_report_e2e
            COMMAND $<TARGET_FILE:bgt_simulate>
                --subjects 1
                --prior 0.2
                --provider cpu
                --write-csv
                --output-dir ${CMAKE_BINARY_DIR}/test-reports
                --run-name bgt-simulate-e2e)
        set_tests_properties(bgt_simulate_report_e2e PROPERTIES LABELS "e2e")

        add_test(NAME bgt_simulate_scalar_prior_e2e
            COMMAND /bin/sh -c "$<TARGET_FILE:bgt_simulate> --subjects 3 --variants 1 --prior 0.02 --lattice replicated_non_dilution --mode local_tree --search-depth 5 --provider cpu --selector op_bbpa --threshold-up 0.01 --threshold-lo 0.01 --branch-threshold 0.001 --write-csv --output-dir ${CMAKE_BINARY_DIR}/test-reports --run-name bgt-simulate-scalar-prior && grep -q 'total_leaves,11' ${CMAKE_BINARY_DIR}/test-reports/bgt-simulate-scalar-prior.csv")
        set_tests_properties(bgt_simulate_scalar_prior_e2e PROPERTIES LABELS "e2e;regression")
    endif()

    if(BGT_ENABLE_CUDA)
        bgt_add_cuda_test(cuda_tree_parity_test ${TEST_SOURCE_DIR}/integration/cuda_tree_parity_test.cpp)
    endif()
endif()

include(InstallRequiredSystemLibraries)
install(EXPORT BGTTargets
    NAMESPACE bgt::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BGT)
install(FILES
    ${PROJECT_BINARY_DIR}/BGTConfig.cmake
    ${PROJECT_BINARY_DIR}/BGTConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BGT)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/license.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${BGT_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${BGT_MINOR}")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)
