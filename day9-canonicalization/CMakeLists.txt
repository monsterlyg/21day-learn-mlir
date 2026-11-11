cmake_minimum_required(VERSION 3.16)

# 兼容性：如果 cmake 版本不支持 CMP0116，跳过
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

project(mlir-learn-day9 LANGUAGES CXX)

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
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_definitions(${LLVM_DEFINITIONS})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/IR/MyNPU.td)
mlir_tablegen(MyNPUOps.h.inc           -gen-op-decls)
mlir_tablegen(MyNPUOps.cpp.inc         -gen-op-defs)
mlir_tablegen(MyNPUDialect.h.inc    -gen-dialect-decls  -dialect=mynpu)
mlir_tablegen(MyNPUDialect.cpp.inc  -gen-dialect-defs   -dialect=mynpu)
add_mlir_dialect_tablegen_target(MLIRMyNPUIncGen)
set(TABLEGEN_OUTPUT)

# Generate the OpInterface from its own TableGen input, like the Toy tutorial.
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/IR/MyNPUCostOpInterface.td)
mlir_tablegen(MyNPUCostOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(MyNPUCostOpInterface.cpp.inc -gen-op-interface-defs)
add_mlir_generic_tablegen_target(MLIRMyNPUCostOpInterfaceIncGen)
set(TABLEGEN_OUTPUT)
# macro(add_mlir_generic_tablegen_target target)
#     add_public_tablegen_target(${target})
#     add_dependencies(mlir-generic-headers ${target})
# endmacro()

# macro(add_mlir_dialect_tablegen_target target)
#     add_public_tablegen_target(${target})
#     add_dependencies(mlir-headers ${target})
# endmacro()

add_library(MyNPU  SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/IR/MyNPU.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IR/MyNPUOps.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/CostAnalysisPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/LoweringPasses.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/PassesPlugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/passes/ShapeInferencePass.cpp
)

# 依赖 tablegen 生成的头文件
add_dependencies(MyNPU MLIRMyNPUIncGen MLIRMyNPUCostOpInterfaceIncGen)

target_link_libraries(MyNPU
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
        MLIRAsmParser
)
