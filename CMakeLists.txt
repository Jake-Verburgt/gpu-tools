cmake_minimum_required(VERSION 3.18)

project(bpu_tools
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

# ----------------------------
# Options
# ----------------------------
option(BUILD_NCCL_THROTTLE "Build NCCL-based throttle tool" ON)
option(BUILD_GPU_THROTTLE  "Build GPU-throttle tool" ON)

# Choose CUDA arch list at configure time:
#   -DTHROTTLE_CUDA_ARCHS="70;80;90"
set(THROTTLE_CUDA_ARCHS "50;60;70;80;90" CACHE STRING "CUDA architectures (e.g. 70;80;90 or 80-real;80-virtual)")

# ----------------------------
# Global-ish (safe) defaults
# ----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA language standard (only matters for CUDA targets)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Helpful on clusters / mixed toolchains
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Where your headers live
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find CUDA toolkit (gives you imported targets like CUDA::cudart)
find_package(CUDAToolkit REQUIRED)

# --------------------------------------------------------------------------
# Helper function for CUDA executables
function(add_cuda_tool target_name cu_source)
  add_executable(${target_name} ${cu_source})

  # Make sure this target is treated as CUDA (in case you add .cpp later)
  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${THROTTLE_CUDA_ARCHS}"
  )

  target_include_directories(${target_name}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  target_compile_features(${target_name} PRIVATE cxx_std_17)

  # Reasonable warnings; adjust as you like
  target_compile_options(${target_name}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )

  # Common CUDA runtime link
  target_link_libraries(${target_name}
    PRIVATE
      CUDA::cudart
  )
endfunction()

# ----------------------------
# CUDA Targets
# ----------------------------
if(BUILD_NCCL_THROTTLE)
  add_cuda_tool(nccl_throttle ${CMAKE_CURRENT_SOURCE_DIR}/src/nccl_throttle/nccl_throttle.cu)

  # NCCL discovery: prefer environment hint NCCL_ROOT if you set it via module
  #   export NCCL_ROOT=/path/to/nccl
  # Expects:
  #   $NCCL_ROOT/include/nccl.h
  #   $NCCL_ROOT/lib or lib64/libnccl.so
  find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ENV NCCL_ROOT
    PATH_SUFFIXES include
  )

  find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ENV NCCL_ROOT
    PATH_SUFFIXES lib lib64
  )

  if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR
      "Could not find NCCL. Set NCCL_ROOT to the NCCL install prefix.\n"
      "Example:\n"
      "  export NCCL_ROOT=/apps/.../nccl/2.x\n"
    )
  endif()

  target_include_directories(nccl_throttle PRIVATE ${NCCL_INCLUDE_DIR})
  target_link_libraries(nccl_throttle PRIVATE ${NCCL_LIBRARY})
endif()

if(BUILD_GPU_THROTTLE)
  add_cuda_tool(gpu_throttle ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_throttle/gpu_throttle.cu)

  # If you use NVML in the future, you'd do something like:
  # find_library(NVML_LIBRARY NAMES nvidia-ml)
  # target_link_libraries(gpu_throttle PRIVATE ${NVML_LIBRARY})
endif()

# ----------------------------
# Future non-CUDA targets example
# ----------------------------
# add_executable(cpu_tool src/cpu_tool/main.cpp)
# target_include_directories(cpu_tool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# target_compile_features(cpu_tool PRIVATE cxx_std_17)
# target_compile_options(cpu_tool PRIVATE -Wall -Wextra -Wpedantic)
