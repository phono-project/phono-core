cmake_minimum_required(VERSION 3.24)

# Keep an explicitly supplied triplet intact for cross-compilation.
if(CMAKE_HOST_WIN32 AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING
      "vcpkg triplet used to build phono-core dependencies")
endif()

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE FILEPATH "vcpkg toolchain file")
endif()

project(edge_phono_engine CXX)

include(CTest)

# The pixi environment's Python drives the configure-time codegen: the
# zh2Hans table below and the selective-build manifest merge. Resolve it up
# front so it is available before the ExecuTorch sub-build is added.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Set MSVC runtime library to MT to avoid DLL dependencies on Windows.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

option(PHONO_USE_INSTALLED_EXECUTORCH
       "Use a pre-built+installed ExecuTorch via find_package() instead of add_subdirectory()" OFF)
option(PHONO_XNNPACK_WEIGHT_CACHE
       "Share XNNPACK packed weights across delegates" ON)

set(EXECUTORCH_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/executorch" CACHE PATH
    "Path to an ExecuTorch source checkout, used when PHONO_USE_INSTALLED_EXECUTORCH=OFF")

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
      "  git clone --depth 1 https://github.com/pytorch/executorch ${EXECUTORCH_SOURCE_DIR}\n"
      "...or build+install ExecuTorch separately and reconfigure with\n"
      "  -DPHONO_USE_INSTALLED_EXECUTORCH=ON -DCMAKE_PREFIX_PATH=<install prefix>\n"
      "See README.md for full instructions.")
  endif()

  # ═════════════════════════════════════════════════════════════════════
  # Selective operator build from ops_config manifests
  #
  # PhonoP2C/export.py writes per-model + merged operator manifests (in
  # ExecuTorch's selected_operators.yaml format) recording exactly which
  # operators/dtypes the deployed models use. Copy the manifests you want to
  # include into ${PHONO_OPS_CONFIG_DIR} (default ops_config/). When at least
  # one manifest is present, the ExecuTorch kernel registration is pruned to
  # the union of those manifests via EXECUTORCH_SELECT_OPS_LIST; with
  # PHONO_DTYPE_SELECTIVE_BUILD=ON the portable kernels are additionally pruned
  # per-dtype.
  #
  # ExecuTorch's upstream dtype-selective-build only accepts a single .pte
  # model (EXECUTORCH_SELECT_OPS_MODEL), which cannot represent phono-core's
  # two-model (pre/post) deployment. We therefore generate the dtype
  # allow-list header from the merged multi-model manifest and apply it to the
  # portable_kernels library directly (EXECUTORCH_SELECTIVE_BUILD_DTYPE). This
  # is the same mechanism upstream uses, driven by a manifest instead of a
  # single .pte file.
  # ═════════════════════════════════════════════════════════════════════
  set(PHONO_OPS_CONFIG_DIR "${CMAKE_SOURCE_DIR}/ops_config" CACHE PATH
      "Directory holding ExecuTorch selective-build manifests from PhonoP2C/export.py")
  set(PHONO_OPS_MANIFESTS "" CACHE STRING
      "Semicolon-separated list of manifest YAMLs to merge (default: <tag>_ops.yaml in PHONO_OPS_CONFIG_DIR)")
  set(PHONO_DTYPE_SELECTIVE_BUILD ON CACHE BOOL
      "Prune portable kernels to the dtypes present in the merged manifest")

  set(_phono_ops_select_dir "${CMAKE_BINARY_DIR}/phono_ops_config")
  set(_phono_manifest_merged "${_phono_ops_select_dir}/selected_operators.yaml")
  set(_phono_root_ops_file "${_phono_ops_select_dir}/selected_root_ops.txt")
  set(_phono_manifest_inputs "")

  if(PHONO_OPS_MANIFESTS)
    string(REPLACE ";" " " _phono_manifest_inputs "${PHONO_OPS_MANIFESTS}")
    separate_arguments(_phono_manifest_inputs)
  else()
    # Prefer the merged manifest(s) produced by export.py (e.g.
    # phonop2c_<version>_ops.yaml); per-model manifests (…_pre_ops.yaml /
    # …_post_ops.yaml) are only picked up when no merged one exists.
    file(GLOB _phono_merged_globs
         "${PHONO_OPS_CONFIG_DIR}/*_ops.yaml"
         "${PHONO_OPS_CONFIG_DIR}/*_ops.yml")
    list(FILTER _phono_merged_globs EXCLUDE REGEX ".*_(pre|post)_ops\\.ya?ml$")
    if(_phono_merged_globs)
      list(SORT _phono_merged_globs)
      set(_phono_manifest_inputs ${_phono_merged_globs})
    else()
      file(GLOB _phono_per_model_globs
           "${PHONO_OPS_CONFIG_DIR}/*_ops.yaml"
           "${PHONO_OPS_CONFIG_DIR}/*_ops.yml")
      if(_phono_per_model_globs)
        list(SORT _phono_per_model_globs)
        set(_phono_manifest_inputs ${_phono_per_model_globs})
      endif()
    endif()
  endif()

  if(_phono_manifest_inputs)
    file(MAKE_DIRECTORY "${_phono_ops_select_dir}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/codegen/ops_manifest_codegen.py"
              --output "${_phono_manifest_merged}"
              --root-ops-output "${_phono_root_ops_file}"
              ${_phono_manifest_inputs}
      RESULT_VARIABLE PHONO_OPS_MANIFEST_RES)
    if(NOT PHONO_OPS_MANIFEST_RES EQUAL 0)
      message(FATAL_ERROR "phono-core: ops manifest merge failed (see output above)")
    endif()
    file(READ "${_phono_root_ops_file}" _phono_root_ops)
    string(STRIP "${_phono_root_ops}" _phono_root_ops)
    if(_phono_root_ops)
      set(EXECUTORCH_SELECT_OPS_LIST "${_phono_root_ops}" CACHE STRING "" FORCE)
      # The three specifiers are mutually exclusive upstream; clear any stale
      # YAML/MODEL values so a re-configure cannot trigger the conflict check.
      set(EXECUTORCH_SELECT_OPS_YAML "" CACHE STRING "" FORCE)
      set(EXECUTORCH_SELECT_OPS_MODEL "" CACHE STRING "" FORCE)
      message(STATUS "phono-core: selective build enabled for ${_phono_manifest_inputs}")
      message(STATUS "phono-core: selected root ops: ${_phono_root_ops}")
    else()
      message(STATUS "phono-core: manifests opt into all operators; selective build disabled")
    endif()
  else()
    set(EXECUTORCH_SELECT_OPS_LIST "" CACHE STRING "" FORCE)
    set(EXECUTORCH_SELECT_OPS_YAML "" CACHE STRING "" FORCE)
    set(EXECUTORCH_SELECT_OPS_MODEL "" CACHE STRING "" FORCE)
    message(STATUS "phono-core: no manifests found in ${PHONO_OPS_CONFIG_DIR}; building full kernel library")
  endif()

  # Prevent windows.h macro pollution (min/max) on Windows MSVC
  if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()

  # ═══════════════════════════════════════════════════════════════════════
  # Pre-build host 'flatc' tool during CMake configure time if not found.
  # This avoids target name collisions and version mismatches (especially on Windows).
  # ═══════════════════════════════════════════════════════════════════════
  find_program(FLATC_EXECUTABLE flatc)

  if(NOT FLATC_EXECUTABLE)
    set(FLATBUFFERS_SRC_DIR "${EXECUTORCH_SOURCE_DIR}/third-party/flatbuffers")
    set(FLATBUFFERS_BUILD_DIR "${CMAKE_BINARY_DIR}/host_flatc_build")

    if(WIN32)
      set(FLATC_BIN_PATH "${FLATBUFFERS_BUILD_DIR}/Release/flatc.exe")
      if(NOT EXISTS "${FLATC_BIN_PATH}")
        set(FLATC_BIN_PATH "${FLATBUFFERS_BUILD_DIR}/flatc.exe")
      endif()
    else()
      set(FLATC_BIN_PATH "${FLATBUFFERS_BUILD_DIR}/flatc")
    endif()

    if(NOT EXISTS "${FLATC_BIN_PATH}")
      message(STATUS "Pre-building host flatc compiler from ExecuTorch third-party/flatbuffers...")
      
      execute_process(
        COMMAND ${CMAKE_COMMAND} -S "${FLATBUFFERS_SRC_DIR}" -B "${FLATBUFFERS_BUILD_DIR}"
                -DFLATBUFFERS_BUILD_FLATC=ON
                -DFLATBUFFERS_BUILD_TESTS=OFF
                -DCMAKE_BUILD_TYPE=Release
        RESULT_VARIABLE CMAKE_FLATC_CONFIG_RES
      )
      
      execute_process(
        COMMAND ${CMAKE_COMMAND} --build "${FLATBUFFERS_BUILD_DIR}" --config Release --target flatc
        RESULT_VARIABLE CMAKE_FLATC_BUILD_RES
      )

      if(NOT CMAKE_FLATC_BUILD_RES EQUAL 0)
        message(FATAL_ERROR "Failed to pre-build host flatc executable!")
      endif()
    endif()

    set(FLATC_EXECUTABLE "${FLATC_BIN_PATH}" CACHE FILEPATH "Path to flatc compiler" FORCE)
  endif()

  message(STATUS "Using FLATC_EXECUTABLE: ${FLATC_EXECUTABLE}")

  set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE ${PHONO_XNNPACK_WEIGHT_CACHE}
      CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_KERNELS_LLM OFF CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "" FORCE)
  # phono-core ships its own demo executables, so skip ExecuTorch's
  # executor_runner (and the runner_util/evalue_util extensions it pulls in).
  set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL OFF CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL OFF CACHE BOOL "" FORCE)

  # ExecuTorch only pulls the googletest submodule when BUILD_TESTING is ON;
  # phono-core uses its own hand-rolled CTest suite and does not need it.
  # Disable BUILD_TESTING for the ExecuTorch sub-build and restore it for the
  # phono-core test targets declared below.
  set(_phono_build_testing_saved "${BUILD_TESTING}")
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

  add_subdirectory(${EXECUTORCH_SOURCE_DIR} ${CMAKE_BINARY_DIR}/executorch_build)

  set(BUILD_TESTING "${_phono_build_testing_saved}" CACHE BOOL "" FORCE)

  # Apply per-dtype pruning to the portable kernel library. ExecuTorch's
  # dtype-selective-build only accepts a single .pte model, so the allow-list
  # header is generated from the merged multi-model manifest instead (see the
  # comment above). This must run after add_subdirectory() so the target
  # exists; the compile definition/header only affect how its sources are
  # compiled, which happens at build time.
  if(_phono_manifest_inputs AND PHONO_DTYPE_SELECTIVE_BUILD AND TARGET portable_kernels)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${EXECUTORCH_SOURCE_DIR}/codegen/tools/gen_selected_op_variants.py"
              --yaml-file-path "${_phono_manifest_merged}"
              --output-dir "${_phono_ops_select_dir}"
      RESULT_VARIABLE PHONO_OPS_VARIANTS_RES)
    if(NOT PHONO_OPS_VARIANTS_RES EQUAL 0)
      message(FATAL_ERROR "phono-core: dtype-selective-build header generation failed (see output above)")
    endif()
    target_compile_definitions(portable_kernels PRIVATE EXECUTORCH_SELECTIVE_BUILD_DTYPE=1)
    target_include_directories(portable_kernels PRIVATE "${_phono_ops_select_dir}")
    message(STATUS "phono-core: dtype-selective-build enabled for portable_kernels")
  endif()
endif()

find_package(uni-algo CONFIG REQUIRED)
set(PHONO_UNI_ALGO_LIBRARIES uni-algo::uni-algo)

# ═══════════════════════════════════════════════════════════════════════
# zh2Hans dictionary codegen
#
# The Traditional -> Simplified rules are kept as a small JSON resource
# (res/zh2hans.json) and compiled into a native C++ table at configure time
# by the pixi Python interpreter (codegen/zh2hans_codegen.py). The output
# header is src/gen/zh2hansdict.h.
# ═══════════════════════════════════════════════════════════════════════

set(PHONO_ZH2HANS_JSON "${CMAKE_SOURCE_DIR}/res/zh2hans.json")
set(PHONO_ZH2HANS_HEADER "${CMAKE_SOURCE_DIR}/src/gen/zh2hansdict.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PHONO_ZH2HANS_JSON}")

execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/codegen/zh2hans_codegen.py"
          --input "${PHONO_ZH2HANS_JSON}" --output "${PHONO_ZH2HANS_HEADER}"
  RESULT_VARIABLE PHONO_ZH2HANS_CODEGEN_RES)
if(NOT PHONO_ZH2HANS_CODEGEN_RES EQUAL 0)
  message(FATAL_ERROR "phono-core: zh2hans codegen failed (see output above)")
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
# phono_core: config, tokenizer, KV cache, context,
# inference engine. Everything except src/core and src/algo is ExecuTorch-
# dependent (see src/context/kv_cache.cpp, src/engine/inference_engine.cpp).
# The same sources also back the C-ABI shared library phono_core_shared
# (with interface/phono_api.cpp), which is what the demo/CLI and third-party
# platforms link against.
# ═══════════════════════════════════════════════════════════════════════
set(PHONO_CORE_SOURCES
    src/core/config.cpp
    src/core/tokenizer.cpp
    src/core/text_normalizer.cpp
    src/core/pinyin_normalizer.cpp
    src/algo/trie.cpp
    src/algo/pinyin_segment.cpp
    src/algo/zh2hans.cpp
    src/context/kv_cache.cpp
    src/context/context.cpp
    src/engine/inference_engine.cpp
    src/custom_ops/update_kv_cache.cpp
)

# Links the ExecuTorch runtime + the extensions/backends used by phono-core
# into a target, and flags it for shared-library linkage when ExecuTorch's
# helper command is available. VISIBILITY is PUBLIC for the static core (its
# consumers need the runtime too) and PRIVATE for the self-contained C-ABI
# shared library so that ExecuTorch's static kernel registrations are not
# duplicated into every consumer executable.
function(phono_link_executorch TARGET VISIBILITY)
  # Core runtime and the two extensions.
  target_link_libraries(${TARGET} ${VISIBILITY} executorch)
  if(TARGET extension_module_static)
    target_link_libraries(${TARGET} ${VISIBILITY} extension_module_static)
  elseif(TARGET extension_module)
    target_link_libraries(${TARGET} ${VISIBILITY} extension_module)
  endif()
  if(TARGET extension_tensor)
    target_link_libraries(${TARGET} ${VISIBILITY} extension_tensor)
  endif()
  if(TARGET extension_data_loader)
    target_link_libraries(${TARGET} ${VISIBILITY} extension_data_loader)
  endif()

  if(TARGET xnnpack_backend)
    target_link_libraries(${TARGET} ${VISIBILITY} xnnpack_backend)
    if(COMMAND executorch_target_link_options_shared_lib)
      executorch_target_link_options_shared_lib(xnnpack_backend)
    endif()
  endif()
  # Kernel library: under selective build (ops_config manifests) ExecuTorch
  # produces executorch_selected_kernels (only the selected operators/dtypes);
  # otherwise fall back to the full portable_ops_lib.
  if(TARGET executorch_selected_kernels)
    target_link_libraries(${TARGET} ${VISIBILITY} executorch_selected_kernels)
    if(COMMAND executorch_target_link_options_shared_lib)
      executorch_target_link_options_shared_lib(executorch_selected_kernels)
    endif()
  elseif(TARGET portable_ops_lib)
    target_link_libraries(${TARGET} ${VISIBILITY} portable_ops_lib)
    if(COMMAND executorch_target_link_options_shared_lib)
      executorch_target_link_options_shared_lib(portable_ops_lib)
    endif()
  endif()

  if(MSVC)
    # Use /WHOLEARCHIVE instead of executorch_target_link_options_shared_lib on MSVC.
    foreach(lib IN LISTS EXT_LIBS KERNEL_LIB)
      target_link_libraries(${TARGET} ${VISIBILITY} "$<LINK_LIBRARY:WHOLEARCHIVE,${lib}>")
    endforeach()
    target_link_libraries(${TARGET} ${VISIBILITY} ${EXT_LIBS} ${KERNEL_LIB})
  else()
    target_link_libraries(${TARGET} ${VISIBILITY} ${EXT_LIBS} ${KERNEL_LIB})
    
    if(COMMAND executorch_target_link_options_shared_lib)
      foreach(lib IN LISTS EXT_LIBS KERNEL_LIB)
         executorch_target_link_options_shared_lib(${lib})
      endforeach()

      if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Windows"))
          executorch_target_link_options_shared_lib(${TARGET})
      endif()
    endif()
  endif()
endfunction()

function(phono_configure_target TARGET LINK_VISIBILITY)
  target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/interface
    include)
  target_compile_features(${TARGET} PUBLIC cxx_std_20)
  target_link_libraries(${TARGET} ${LINK_VISIBILITY} nlohmann_json::nlohmann_json)
  target_link_libraries(${TARGET} ${LINK_VISIBILITY} ${PHONO_UNI_ALGO_LIBRARIES})
  phono_link_executorch(${TARGET} ${LINK_VISIBILITY})
endfunction()

add_library(phono_core STATIC ${PHONO_CORE_SOURCES})
phono_configure_target(phono_core PUBLIC)

# ═══════════════════════════════════════════════════════════════════════
# phono_core_shared: the C-ABI shared library (libphono_core.so /
# phono_core.dll / libphono_core.dylib). Exports the stable C API declared in
# interface/phono_api.h. core_config is passed in as a JSON string and parsed
# internally; invalid parameters surface as phono_status error enums.
# ═══════════════════════════════════════════════════════════════════════
add_library(phono_core_shared SHARED ${PHONO_CORE_SOURCES} interface/phono_api.cpp)
phono_configure_target(phono_core_shared PRIVATE)
target_compile_definitions(phono_core_shared PRIVATE PHONO_BUILDING_SHARED)
set_target_properties(phono_core_shared PROPERTIES
    OUTPUT_NAME "phono_core"
    RUNTIME_OUTPUT_DIRECTORY "${PHONO_RUNTIME_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PHONO_RUNTIME_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${PHONO_RUNTIME_OUTPUT_DIR}")

# ═══════════════════════════════════════════════════════════════════════
# Demo: links libphono_core.so and talks to it exclusively through the stable
# C interface.
# ═══════════════════════════════════════════════════════════════════════
add_executable(streaming_benchmark_demo_capi apps/streaming_benchmark_demo_capi.cpp)
target_link_libraries(streaming_benchmark_demo_capi PRIVATE phono_core_shared)
set_target_properties(streaming_benchmark_demo_capi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PHONO_RUNTIME_OUTPUT_DIR}")

add_executable(performance_benchmark apps/performance_benchmark.cpp)
target_link_libraries(performance_benchmark PRIVATE phono_core)
set_target_properties(performance_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PHONO_RUNTIME_OUTPUT_DIR}")

add_executable(ime_demo_capi apps/ime_demo_capi.cpp)
target_link_libraries(ime_demo_capi PRIVATE phono_core_shared)
set_target_properties(ime_demo_capi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PHONO_RUNTIME_OUTPUT_DIR}")

if(BUILD_TESTING)
  add_executable(phono_core_tests tests/core_tests.cpp)
  target_link_libraries(phono_core_tests PRIVATE phono_core)
  add_test(NAME phono_core_tests COMMAND phono_core_tests)

  # C-ABI smoke tests: verify error-code surfaces (bad package path, invalid
  # core_config JSON, model-limit violations) without needing a model.
  add_executable(phono_capi_tests tests/capi_tests.cpp)
  target_link_libraries(phono_capi_tests PRIVATE phono_core_shared)
  add_test(NAME phono_capi_tests COMMAND phono_capi_tests)

  # Model-based session tests skip unless PHONO_TEST_MODEL_DIR is set to
  # a v2 model package directory.
  add_executable(phono_engine_tests tests/engine_tests.cpp)
  target_link_libraries(phono_engine_tests PRIVATE phono_core)
  add_test(NAME phono_engine_tests COMMAND phono_engine_tests)

  add_executable(phono_ime_editor_tests tests/ime_editor_tests.cpp)
  target_include_directories(phono_ime_editor_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME phono_ime_editor_tests COMMAND phono_ime_editor_tests)
  add_test(NAME ime_demo_capi_help COMMAND ime_demo_capi --help)
endif()
