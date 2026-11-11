cmake_minimum_required(VERSION 3.20)

project(mlir-learn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LLVM/MLIR 安装目录
set(LLVM_INSTALL_DIR
    "/data8/liyinggang/Projects/CXX/llvm-project/lyg-install"
)

# 告诉 CMake 去哪里找 LLVMConfig.cmake
list(APPEND CMAKE_PREFIX_PATH
    "${LLVM_INSTALL_DIR}/lib/cmake/llvm"
)

# 告诉 CMake 去哪里找 MLIRConfig.cmake
list(APPEND CMAKE_PREFIX_PATH
    "${LLVM_INSTALL_DIR}/lib/cmake/mlir"
)

# 找 LLVM
find_package(LLVM REQUIRED CONFIG)

# 找 MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM_DIR = ${LLVM_DIR}")
message(STATUS "MLIR_DIR = ${MLIR_DIR}")

# LLVM / MLIR 头文件
include_directories(
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

# LLVM / MLIR 编译定义
add_definitions(
    ${LLVM_DEFINITIONS}
)

add_executable(mlir-learn
    day2.cpp
)

# 我们当前需要的 MLIR libraries
target_link_libraries(mlir-learn
    PRIVATE
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRArithDialect
        MLIRFuncDialect
)