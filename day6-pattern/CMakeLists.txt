cmake_minimum_required(VERSION 3.20)

project(mlir-learn-day5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LLVM/MLIR 安装目录
set(LLVM_INSTALL_DIR
    "/data8/liyinggang/Projects/CXX/llvm-project/lyg-install"
)

list(APPEND CMAKE_PREFIX_PATH
    "${LLVM_INSTALL_DIR}/lib/cmake/llvm"
    "${LLVM_INSTALL_DIR}/lib/cmake/mlir"
)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM_DIR = ${LLVM_DIR}")
message(STATUS "MLIR_DIR = ${MLIR_DIR}")

# 让 mlir_tablegen / add_mlir_dialect 可用
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# 头文件：LLVM/MLIR 安装目录 + 当前构建目录（生成的 .inc 在这里）
# 注意：必须放在 add_mlir_dialect 之前 —— tablegen 宏在此时读取
# INCLUDE_DIRECTORIES 目录属性作为 mlir-tblgen 的 -I 搜索路径，
# 否则找不到 mlir/IR/OpBase.td
include_directories(
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_definitions(${LLVM_DEFINITIONS})

# Declare a dialect in the include directory
# function(add_mlir_dialect dialect dialect_namespace)
#   set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
#   mlir_tablegen(${dialect}.h.inc -gen-op-decls)
#   mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
#   mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
#   mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
#   mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
#   mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
#   add_mlir_dialect_tablegen_target(MLIR${dialect}IncGen)
# endfunction()
add_mlir_dialect(FusionOps fusion)

add_library(FusionPlugin SHARED
    FusionDialect.cpp
    FusionOps.cpp
    FusionPasses.cpp
    FusionPassesPlugin.cpp
)

# 依赖 tablegen 生成的头文件
add_dependencies(FusionPlugin MLIRFusionOpsIncGen)

target_link_libraries(FusionPlugin
    PRIVATE
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRPass
        MLIRLinalgDialect
        MLIRArithDialect
        MLIRFuncDialect
        MLIRTensorDialect
        MLIRTransforms
        MLIRDialectUtils
        MLIRAffineDialect
)
