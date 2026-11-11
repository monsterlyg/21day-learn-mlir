# Day 18：Pass Pipeline——把变换串起来并可回看

## 今天要解决什么问题？

Day 9 你运行过单个 `canonicalize`，Day 15 你把 bufferize 和 lowering 串在一行里。现在要系统回答：

> 多个 Pass 按什么顺序跑？每一步 IR 变成了什么样？如果结果不对，怎么定位是哪一步改坏的？

Pass pipeline 不只是「把命令拼长」。它涉及合法性、顺序、重复运行和可观察性。今天的目标：

1. 掌握 `-p` 的 pipeline 语法和 anchor。
2. 用 `canonicalize`、`cse`、`symbol-dce` 做一轮清理。
3. 用 IR 打印选项观察每一步的变化。
4. 建立一个「分阶段保存 IR」的工作习惯。

## 环境

```bash
MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
```

## 第一站：pipeline 的语法

基本形式：

```text
<anchor-op>(pass1, pass2, ...)
```

例如：

```bash
$MLIR_OPT -p 'builtin.module(canonicalize,cse)' input.mlir
```

几个规则：

- **anchor 是 op 名**，不是 Pass 名。`builtin.module` 表示在 `ModuleOp` 上跑。
- 嵌套用括号：`builtin.module(func.func(tosa-to-linalg-named))`。
- Pass 的顺序就是执行顺序。
- 带选项的 Pass 用 `{}`：`one-shot-bufferize{bufferize-function-boundaries=1}`。

Day 16 已经踩过一次坑：`tosa-to-linalg-named` 是 `FunctionOpInterface` pass，挂在 `builtin.module` 上会报：

```text
error: unable to schedule pass ... on a PassManager intended to run on 'builtin.module'
```

遇到这个错误，先查 Pass 定义里的 op 类型，而不是怀疑 IR。

## 第二站：一轮基础清理

准备一个同时包含冗余计算、重复计算和死函数的 Module：

```mlir
module {
  func.func private @dead()

  func.func @f(%x: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %a = arith.addi %x, %c0 : i32
    %b = arith.addi %x, %c0 : i32
    %r = arith.addi %a, %b : i32
    return %r : i32
  }
}
```

运行（把上面的 module 保存为 `input.mlir`）：

```bash
$MLIR_OPT --canonicalize --cse --symbol-dce input.mlir
```

输出：

```mlir
module {
  func.func @f(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    return %0 : i32
  }
}
```

三个 Pass 各做了什么：

- `canonicalize`：`x + 0 → x`，把 `%a`、`%b` 化简成 `%x`。
- `cse`：两个相同的 `arith.addi %x, %x` 只保留一个。
- `symbol-dce`：删除没有任何引用的私有函数 `@dead`。

注意顺序：如果先 CSE 再 canonicalize，最终结果通常一样，但中间 IR 不同。对于更复杂的规则，顺序会影响能否匹配。

## 第三站：看清每一步

### 打印每个 Pass 之后的 IR

```bash
$MLIR_OPT -mlir-print-ir-after-all -p 'builtin.module(canonicalize,cse)' input.mlir
```

stderr 会出现：

```text
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module { ... }
```

### 只在 IR 真的变化时打印

```bash
$MLIR_OPT -mlir-print-ir-after-all -mlir-print-ir-after-change \
  -p 'builtin.module(canonicalize,cse)' input.mlir
```

这个组合在长 pipeline 里非常有用：跳过「跑了但没改」的 Pass。

### 把每一步写成文件

```bash
$MLIR_OPT -mlir-print-ir-after-all \
  --mlir-print-ir-tree-dir=./ir-tree \
  -p 'builtin.module(canonicalize,cse)' input.mlir
```

会生成类似：

```text
ir-tree/builtin_module_no-symbol-name/0_canonicalize.mlir
ir-tree/builtin_module_no-symbol-name/1_cse.mlir
```

`--mlir-print-ir-tree-dir` 必须和 `-mlir-print-ir-after-*` 一起用，单独写不会生成文件（本机已验证）。

## 第四站：分阶段保存

对一条较长 pipeline，更实用的做法是分段保存：

```bash
# 阶段 1：清理
$MLIR_OPT --canonicalize --cse input.mlir -o stage1.mlir

# 阶段 2：结构优化
$MLIR_OPT --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
  stage1.mlir -o stage2.mlir

# 阶段 3：bufferize
$MLIR_OPT -p 'builtin.module(one-shot-bufferize{bufferize-function-boundaries=1})' \
  stage2.mlir -o stage3.mlir

# 阶段 4：降到循环
$MLIR_OPT --convert-linalg-to-loops stage3.mlir -o stage4.mlir
```

好处：

- 每一步都有文件，可以 diff。
- 某一步失败时，输入就是上一步的产物，容易复现。
- 以后可以只重跑受影响的后半段。

## 第五站：顺序为什么重要

一个典型的 tensor → buffer 流程：

```text
canonicalize → CSE → linalg 优化 → bufferize → linalg-to-loops
```

如果反过来先 bufferize：

```text
bufferize → linalg 优化
```

很多 tensor 级别的 pattern 就不再生效，因为 op 已经变成 memref 版本，语义是原地写。优化不是不能做，而是可用的规则集合变了。

另一个例子：

```text
tosa-to-linalg → canonicalize
```

通常比

```text
canonicalize → tosa-to-linalg
```

更有效，因为 lowering 会新造出 `tensor.empty`、broadcast generic 等结构，正好交给 canonicalize 清理。

可以记成两条原则：

1. **先做能产生更多优化机会的转换，再做清理。**
2. **不要依赖 canonicalize 保证正确性**；它只是 best-effort 的整理。

## 今天的练习路线

### 练习 A：观察每一步

对第二站的输入运行：

```bash
$MLIR_OPT -mlir-print-ir-after-all -mlir-print-ir-after-change \
  -p 'builtin.module(canonicalize,cse,symbol-dce)' input.mlir
```

记录哪几个 Pass 真正改变了 IR。

### 练习 B：调换顺序

分别运行：

```text
canonicalize,cse
cse,canonicalize
```

比较最终 IR 是否相同。再构造一个例子，让顺序真的影响结果（提示：先融合再清理 vs 先清理再融合）。

### 练习 C：保存一棵 IR 树

用 `--mlir-print-ir-tree-dir` 跑一遍包含 3 个 Pass 的 pipeline，打开生成的文件，按顺序读一遍。这是排查「哪一步改坏了」最快的方法。

### 练习 D：负例

把 `one-shot-bufferize` 放在 `linalg-fuse-elementwise-ops` 之前，跑一遍并观察 fusion 是否还能发生。解释原因。

## 常见坑

- **anchor 写错**：FunctionOpInterface pass 需要 `func.func(...)`。
- **裸 flag 找不到插件 Pass**：插件里的 Pass 只能写在 `-p` 字符串里（Day 9 解释过原因）。
- **`-mlir-print-ir-tree-dir` 不生效**：必须配合 `-mlir-print-ir-after-all` 或 `-mlir-print-ir-before-all`。
- **以为 Pass 顺序无所谓**：不同阶段能匹配的 op 形态不同。
- **一次性写超长 pipeline**：出错时很难定位，分段保存更可控。
- **用 `-o` 时忘了看 stderr**：诊断和 IR 打印走 stderr，不一定会进输出文件。

## 今天结束时，试着讲给别人听

1. `builtin.module(...)` 里的 `builtin.module` 是什么？
2. 为什么更推荐「先 lowering 再 canonicalize」？
3. `-mlir-print-ir-after-change` 解决什么问题？
4. 分阶段保存 IR 对调试有什么帮助？
5. 为什么不能让 pipeline 依赖 canonicalize 来保证正确性？

## 参考资料

- [MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/)
- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/#pass-pipelines)
- [MLIR Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
- 本机命令：`$MLIR_OPT --help | grep mlir-print-ir`
