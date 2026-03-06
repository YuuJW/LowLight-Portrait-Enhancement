cmake_minimum_required(VERSION 3.10)
project(RetinexFormerEngine)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
set(OpenCV_DIR "D:/Softwares/OpenCV/opencv/build")
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_DIR "D:/yujw_Projects/img_pro/references/onnxruntime" CACHE PATH "ONNX Runtime directory")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Project headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/onnx_wrapper.cpp
    src/session_pool.cpp
    src/tiling_manager.cpp
    src/retinexformer_engine.cpp
)

# Build library
add_library(retinexformer_engine STATIC ${SOURCES})
target_link_libraries(retinexformer_engine ${OpenCV_LIBS} onnxruntime)

# Test program
add_executable(test_engine src/main.cpp)
target_link_libraries(test_engine retinexformer_engine)
