[windows]
set shell := ["cmd.exe", "/c"]

EXECUTORCH_REF := '00f06544be6b674f2ce8c1a746b76f23cf585d21'
EXECUTORCH_SUBS := trim(replace('''
  backends/xnnpack/third-party/FP16
  backends/xnnpack/third-party/FXdiv
  backends/xnnpack/third-party/XNNPACK
  backends/xnnpack/third-party/cpuinfo
  backends/xnnpack/third-party/pthreadpool
  kernels/optimized/third-party/eigen
  third-party/flatbuffers
  third-party/flatcc
  third-party/gflags
  third-party/json
  third-party/pocketfft
''', "\n", " "))

# Fetch the ExecuTorch source and initialize *only* the submodules phono-core
# actually needs for its CMake configuration:
#   - third-party/{flatbuffers,flatcc,json,gflags}: added unconditionally by
#     executorch's third-party/CMakeLists.txt
#   - backends/xnnpack/third-party/{FP16,FXdiv,XNNPACK,cpuinfo,pthreadpool}:
#     required by the XNNPACK backend (EXECUTORCH_BUILD_XNNPACK=ON) and the
#     default cpuinfo/pthreadpool build
#   - kernels/optimized/third-party/eigen and third-party/pocketfft:
#     upstream optimized CPU kernels and their portable fallbacks
# googletest is intentionally excluded: phono-core builds BUILD_TESTING=OFF for
# the ExecuTorch sub-build (it has its own hand-rolled CTest suite).

[working-directory: 'third_party/executorch']
prepare-executorch:
  git init -q
  git config remote.origin.url https://github.com/pytorch/executorch.git
  git config remote.origin.promisor true
  git config remote.origin.partialclonefilter "blob:none"
  git fetch --depth 1 origin {{EXECUTORCH_REF}}
  git checkout --detach {{EXECUTORCH_REF}}
  git submodule update --init --recursive --depth 1 {{EXECUTORCH_SUBS}}

prepare-models:
  hf download afirelily/phonop2c_v2_2_base_w4a8_model --local-dir cache/models/w4a8
  hf download afirelily/phonop2c_v2_2_base_w8a8_model --local-dir cache/models/w8a8
