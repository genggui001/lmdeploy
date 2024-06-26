#!/bin/sh

builder="-G Ninja"

if [ "$1" == "make" ]; then
    builder=""
fi

cmake ${builder} .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DBUILD_PY_FFI=ON \
    -DBUILD_MULTI_GPU=ON \
    -DBUILD_CUTLASS_MOE=OFF \
    -DBUILD_CUTLASS_MIXED_GEMM=OFF \
    -DCMAKE_CUDA_FLAGS="-lineinfo" \
    -DUSE_NVTX=ON
