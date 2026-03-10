cmake_minimum_required(VERSION 3.16)
project(RetinexFormerEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(WORKSPACE_ROOT "${PROJECT_ROOT}/.." ABSOLUTE)

if(NOT OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR "$ENV{OpenCV_DIR}" CACHE PATH "Path to OpenCVConfig.cmake")
endif()
find_package(OpenCV REQUIRED)

if(DEFINED ENV{ONNXRUNTIME_DIR})
    set(_default_onnxruntime_dir "$ENV{ONNXRUNTIME_DIR}")
else()
    set(_default_onnxruntime_dir "${WORKSPACE_ROOT}/references/onnxruntime")
endif()
set(ONNXRUNTIME_DIR "${_default_onnxruntime_dir}" CACHE PATH "Root directory of an ONNX Runtime release")

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNXRUNTIME_DIR must point to an ONNX Runtime release with include/ and lib/.\n"
        "Current value: ${ONNXRUNTIME_DIR}\n"
        "Example: cmake -S deploy/cpp -B deploy/cpp/build -DONNXRUNTIME_DIR=D:/sdk/onnxruntime"
    )
endif()

set(SOURCES
    src/onnx_wrapper.cpp
    src/session_pool.cpp
    src/tiling_manager.cpp
    src/retinexformer_engine.cpp
)

add_library(retinexformer_engine STATIC ${SOURCES})
target_include_directories(retinexformer_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${ONNXRUNTIME_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(retinexformer_engine PRIVATE ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(retinexformer_engine PUBLIC ${OpenCV_LIBS} PRIVATE onnxruntime)

add_executable(test_engine src/main.cpp)
target_link_libraries(test_engine PRIVATE retinexformer_engine)

add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE retinexformer_engine)
