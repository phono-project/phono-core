cmake_minimum_required(VERSION 3.24)
project(edge_phono_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(PHONO_USE_INSTALLED_EXECUTORCH
       "Use a pre-built+installed ExecuTorch via find_package() instead of add_subdirectory()" OFF)

set(EXECUTORCH_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/executorch" CACHE PATH
    "Path to an ExecuTorch source checkout, used when PHONO_USE_INSTALLED_EXECUTORCH=OFF")
# set(EXECUTORCH_SELECT_OPS_YAML "${CMAKE_SOURCE_DIR}/ops_config/selected_ops.yaml" CACHE PATH "Path to the merged selected operators YAML file for pruning")

# ═══════════════════════════════════════════════════════════════════════
# ExecuTorch
#
# ExecuTorch's own build is orchestrated by Python-driven CMake option 
# presets (EXECUTORCH_BUILD_*) rather than a portfile-friendly, dependency-only
# configure/build/install, so it can't be wrapped as a normal port.
# ═══════════════════════════════════════════════════════════════════════
if(PHONO_USE_INSTALLED_EXECUTORCH)
  find_package(nlohmann_json CONFIG REQUIRED)
  find_package(executorch CONFIG REQUIRED)
else()
  if(NOT EXISTS "${EXECUTORCH_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "ExecuTorch source checkout not found at: ${EXECUTORCH_SOURCE_DIR}\n"
      "Run:\n"
      "  git clone --recursive https://github.com/pytorch/executorch ${EXECUTORCH_SOURCE_DIR}\n"
      "...or build+install ExecuTorch separately and reconfigure with\n"
      "  -DPHONO_USE_INSTALLED_EXECUTORCH=ON -DCMAKE_PREFIX_PATH=<install prefix>\n"
      "See README.md for full instructions.")
  endif()

  set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_KERNELS_LLM OFF CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "" FORCE)

  add_subdirectory(${EXECUTORCH_SOURCE_DIR} ${CMAKE_BINARY_DIR}/executorch_build)
endif()

# ═══════════════════════════════════════════════════════════════════════
# phono_core: config, tokenizer, trie + Viterbi decoding, KV cache, context,
# inference engine. Everything except src/core and src/algo is ExecuTorch-
# dependent (see src/context/kv_cache.cpp, src/engine/inference_engine.cpp).
# ═══════════════════════════════════════════════════════════════════════
add_library(phono_core STATIC
    src/core/config.cpp
    src/core/tokenizer.cpp
    src/algo/trie.cpp
    src/algo/viterbi.cpp
    src/context/kv_cache.cpp
    src/context/context.cpp
    src/engine/inference_engine.cpp
    src/custom_ops/update_kv_cache.cpp
)
target_include_directories(phono_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  include)
target_compile_features(phono_core PUBLIC cxx_std_20)
target_link_libraries(phono_core PUBLIC nlohmann_json::nlohmann_json)

# Core runtime + the two extensions we need.
target_link_libraries(phono_core PUBLIC executorch)
if(TARGET extension_module_static)
  target_link_libraries(phono_core PUBLIC extension_module_static)
elseif(TARGET extension_module)
  target_link_libraries(phono_core PUBLIC extension_module)
endif()
if(TARGET extension_tensor)
  target_link_libraries(phono_core PUBLIC extension_tensor)
endif()
if(TARGET extension_data_loader)
  target_link_libraries(phono_core PUBLIC extension_data_loader)
endif()

if(TARGET xnnpack_backend)
  target_link_libraries(phono_core PUBLIC xnnpack_backend)
  if(COMMAND executorch_target_link_options_shared_lib)
    executorch_target_link_options_shared_lib(xnnpack_backend)
  endif()
endif()
if(TARGET portable_ops_lib)
  target_link_libraries(phono_core PUBLIC portable_ops_lib)
  if(COMMAND executorch_target_link_options_shared_lib)
    executorch_target_link_options_shared_lib(portable_ops_lib)
  endif()
endif()

if(COMMAND executorch_target_link_options_shared_lib)
  executorch_target_link_options_shared_lib(phono_core)
endif()

# ═══════════════════════════════════════════════════════════════════════
# Output layout
#
# The model package (+ config) lives in results/, so the runnable demo
# binaries are placed right next to it (results/). This is done per-target
# (RUNTIME_OUTPUT_DIRECTORY) instead of via the global
# CMAKE_RUNTIME_OUTPUT_DIRECTORY so that ExecuTorch's own subdirectory
# executables (e.g. executor_runner) keep landing in build/.
# ═══════════════════════════════════════════════════════════════════════
set(PHONO_RUNTIME_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/results")

# ═══════════════════════════════════════════════════════════════════════
# StreamingBenchmarkDemo
# ═══════════════════════════════════════════════════════════════════════
add_executable(streaming_benchmark_demo apps/streaming_benchmark_demo.cpp)
target_link_libraries(streaming_benchmark_demo PRIVATE phono_core)
set_target_properties(streaming_benchmark_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PHONO_RUNTIME_OUTPUT_DIR}")