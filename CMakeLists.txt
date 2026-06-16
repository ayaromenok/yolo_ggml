cmake_minimum_required(VERSION 3.10)
project(yolo_ggml_app)

set(CMAKE_CXX_STANDARD 17)

# Select GGML Backend: CPU, CUDA, VULKAN, OPENCL
set(GGML_BACKEND "CUDA" CACHE STRING "GGML Backend (CPU, CUDA, VULKAN, OPENCL)")

# GGML Paths
set(GGML_INCLUDE_DIR ~/sdk/src/ggml/include)
include_directories(${GGML_INCLUDE_DIR})

add_executable(yolo_ggml 
    src/main.cpp
    src/common.cpp
    src/stats.cpp
)

# Unset cached library paths to force re-discovery when GGML_BACKEND changes
unset(GGML_LIB CACHE)
unset(GGML_BASE_LIB CACHE)
unset(GGML_CPU_LIB CACHE)
unset(GGML_CUDA_LIB CACHE)
unset(GGML_VULKAN_LIB CACHE)
unset(GGML_OPENCL_LIB CACHE)

if (GGML_BACKEND STREQUAL "CPU")
    set(GGML_LIB_DIR ~/sdk/build/ggml/cpu/src)
    find_library(GGML_LIB ggml PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_BASE_LIB ggml-base PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_CPU_LIB ggml-cpu PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)

    target_compile_definitions(yolo_ggml PRIVATE GGML_BACKEND_CPU)
    target_link_libraries(yolo_ggml 
        ${GGML_LIB}
        ${GGML_BASE_LIB}
        ${GGML_CPU_LIB}
        pthread
        dl
    )
    set_target_properties(yolo_ggml PROPERTIES INSTALL_RPATH "${GGML_LIB_DIR}")

elseif (GGML_BACKEND STREQUAL "CUDA")
    set(GGML_LIB_DIR ~/sdk/build/ggml/cuda/src)
    set(GGML_BACKEND_LIB_DIR ~/sdk/build/ggml/cuda/src/ggml-cuda)
    find_library(GGML_LIB ggml PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_BASE_LIB ggml-base PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_CPU_LIB ggml-cpu PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_CUDA_LIB ggml-cuda PATHS ${GGML_BACKEND_LIB_DIR} NO_DEFAULT_PATH)

    target_compile_definitions(yolo_ggml PRIVATE GGML_BACKEND_CUDA)
    target_link_libraries(yolo_ggml 
        ${GGML_LIB}
        ${GGML_BASE_LIB}
        ${GGML_CPU_LIB}
        ${GGML_CUDA_LIB}
        pthread
        dl
    )
    set_target_properties(yolo_ggml PROPERTIES INSTALL_RPATH "${GGML_LIB_DIR}:${GGML_BACKEND_LIB_DIR}")

elseif (GGML_BACKEND STREQUAL "VULKAN")
    set(GGML_LIB_DIR ~/sdk/build/ggml/vulkan/src)
    set(GGML_BACKEND_LIB_DIR ~/sdk/build/ggml/vulkan/src/ggml-vulkan)
    find_library(GGML_LIB ggml PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_BASE_LIB ggml-base PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_CPU_LIB ggml-cpu PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_VULKAN_LIB ggml-vulkan PATHS ${GGML_BACKEND_LIB_DIR} NO_DEFAULT_PATH)

    target_compile_definitions(yolo_ggml PRIVATE GGML_BACKEND_VULKAN)
    target_link_libraries(yolo_ggml 
        ${GGML_LIB}
        ${GGML_BASE_LIB}
        ${GGML_CPU_LIB}
        ${GGML_VULKAN_LIB}
        pthread
        dl
    )
    set_target_properties(yolo_ggml PROPERTIES INSTALL_RPATH "${GGML_LIB_DIR}:${GGML_BACKEND_LIB_DIR}")

elseif (GGML_BACKEND STREQUAL "OPENCL")
    set(GGML_LIB_DIR ~/sdk/build/ggml/opencl/src)
    set(GGML_BACKEND_LIB_DIR ~/sdk/build/ggml/opencl/src/ggml-opencl)
    find_library(GGML_LIB ggml PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_BASE_LIB ggml-base PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_CPU_LIB ggml-cpu PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
    find_library(GGML_OPENCL_LIB ggml-opencl PATHS ${GGML_BACKEND_LIB_DIR} NO_DEFAULT_PATH)

    target_compile_definitions(yolo_ggml PRIVATE GGML_BACKEND_OPENCL)
    target_link_libraries(yolo_ggml 
        ${GGML_LIB}
        ${GGML_BASE_LIB}
        ${GGML_CPU_LIB}
        ${GGML_OPENCL_LIB}
        pthread
        dl
    )
    set_target_properties(yolo_ggml PROPERTIES INSTALL_RPATH "${GGML_LIB_DIR}:${GGML_BACKEND_LIB_DIR}")

else()
    message(FATAL_ERROR "Unsupported GGML_BACKEND: ${GGML_BACKEND}")
endif()

set_target_properties(yolo_ggml PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
