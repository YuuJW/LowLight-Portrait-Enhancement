cmake_minimum_required(VERSION 3.10)
project(RetinexFormerEngine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# NCNN 路径 (需要用户配置)
set(NCNN_DIR "${CMAKE_SOURCE_DIR}/../../references/ncnn" CACHE PATH "NCNN directory")
include_directories(${NCNN_DIR}/src)
link_directories(${NCNN_DIR}/build/install/lib)

# 项目头文件
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(SOURCES
    src/ncnn_wrapper.cpp
    src/tiling_manager.cpp
    src/retinexformer_engine.cpp
)

# 创建库
add_library(retinexformer_engine STATIC ${SOURCES})
target_link_libraries(retinexformer_engine ${OpenCV_LIBS} ncnn)

# 测试程序
add_executable(test_engine src/main.cpp)
target_link_libraries(test_engine retinexformer_engine)
