#!/usr/bin/env bash

set -eou pipefail

TOPDIR=$(git rev-parse --show-toplevel)/builder

CUDA_VER=${CUDA_VER:-11.8}

PLAT_NAME=manylinux2014_x86_64
for cuver in ${CUDA_VER}; do
    DOCKER_TAG=cuda${cuver}
    OUTPUT_FOLDER=cuda${cuver}_dist
    for pyver in py38 py39 py310 py311; do
        bash ${TOPDIR}/manywheel/build_wheel.sh ${pyver} ${PLAT_NAME} ${DOCKER_TAG} ${OUTPUT_FOLDER} \
            |& tee ${PLAT_NAME}.${pyver}.cuda${cuver}.log.txt
    done
done
