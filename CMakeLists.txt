cmake_minimum_required(VERSION 4.2)

project(geomalg_project)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++26 -Xclang -fheavy")
include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(include/geomalg)

