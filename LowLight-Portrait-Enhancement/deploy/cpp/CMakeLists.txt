cmake_minimum_required(VERSION 3.10)
project(RetinexFormerEngine)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Backend selection: ONNXRUNTIME (default) or NCNN
option(USE_ONNXRUNTIME "Use ONNX Runtime backend" ON)

# OpenCV
set(OpenCV_DIR "D:/Softwares/OpenCV/opencv/build")
find_package(OpenCV REQUIRED)

# Project headers
include_directories(${CMAKE_SOURCE_DIR}/include)

if(USE_ONNXRUNTIME)
    message(STATUS "Building with ONNX Runtime backend")
    add_definitions(-DUSE_ONNXRUNTIME)

    # ONNX Runtime
    set(ONNXRUNTIME_DIR "D:/yujw_Projects/img_pro/references/onnxruntime" CACHE PATH "ONNX Runtime directory")
    include_directories(${ONNXRUNTIME_DIR}/include)
    link_directories(${ONNXRUNTIME_DIR}/lib)

    set(SOURCES
        src/onnx_wrapper.cpp
        src/tiling_manager.cpp
        src/retinexformer_engine.cpp
    )

    add_library(retinexformer_engine STATIC ${SOURCES})
    target_link_libraries(retinexformer_engine ${OpenCV_LIBS} onnxruntime)

else()
    message(STATUS "Building with NCNN backend")

    # NCNN
    set(ncnn_DIR "D:/yujw_Projects/img_pro/references/ncnn-tools/ncnn-20240820-windows-vs2022/x64/lib/cmake/ncnn")
    find_package(ncnn REQUIRED)

    # OpenMP (NCNN dependency)
    find_package(OpenMP)

    set(SOURCES
        src/ncnn_wrapper.cpp
        src/tiling_manager.cpp
        src/retinexformer_engine.cpp
    )

    add_library(retinexformer_engine STATIC ${SOURCES})
    target_link_libraries(retinexformer_engine ${OpenCV_LIBS} ncnn)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(retinexformer_engine OpenMP::OpenMP_CXX)
    endif()
endif()

# Test program
add_executable(test_engine src/main.cpp)
target_link_libraries(test_engine retinexformer_engine)
