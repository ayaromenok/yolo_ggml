cmake_minimum_required(VERSION 3.10)
project(yolo_ggml_app)

set(CMAKE_CXX_STANDARD 17)

# GGML Paths
set(GGML_INCLUDE_DIR /home/az/sdk/src/ggml/include)
set(GGML_LIB_DIR /home/az/sdk/build/ggml/cuda/src)
set(GGML_CUDA_LIB_DIR /home/az/sdk/build/ggml/cuda/src/ggml-cuda)

include_directories(${GGML_INCLUDE_DIR})

add_executable(yolo_ggml 
    src/main.cpp
    src/common.cpp
)

# Libraries to link
find_library(GGML_LIB ggml PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_BASE_LIB ggml-base PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB ggml-cpu PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_CUDA_LIB ggml-cuda PATHS ${GGML_CUDA_LIB_DIR} NO_DEFAULT_PATH)

target_link_libraries(yolo_ggml 
    ${GGML_LIB}
    ${GGML_BASE_LIB}
    ${GGML_CPU_LIB}
    ${GGML_CUDA_LIB}
    pthread
    dl
)

# Set rpath so it can find the libraries at runtime
set_target_properties(yolo_ggml PROPERTIES INSTALL_RPATH "${GGML_LIB_DIR}:${GGML_CUDA_LIB_DIR}")
set_target_properties(yolo_ggml PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
