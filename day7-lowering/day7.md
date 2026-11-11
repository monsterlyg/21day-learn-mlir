可以。你现在前 6 天的路线已经把 “看懂 MLIR → 遍历 IR → 修改 IR → 定义自己的 IR” 这条链路打通了。

从 Day 7 开始，学习重点要发生一次明显变化：

前 6 天：主要是在学习 MLIR 的 IR 编程模型。
Day 7 以后：开始学习 Compiler Pipeline，也就是一个 Dialect 如何逐步变成另一个 Dialect，最终变成可执行代码。

这其实正好进入你最终目标最核心的部分：

ONNX
  ↓
Graph / ONNX Dialect
  ↓
High-Level IR
  ↓
Linalg / TOSA
  ↓
NPU Dialect
  ↓
Low-Level IR
  ↓
Runtime / LLVM / CodeGen
一、Day 7 总目标：真正理解 MLIR Lowering

我建议 Day 7 不要一开始就学习完整的 ONNX → LLVM。

那样跨度太大。

Day 7 应该只集中解决一个问题：

如何把一种 Operation / Type / Dialect，合法地转换成另一种？

例如：

%0 = mymath.add %a, %b : f32

Lowering 成：

%0 = arith.addf %a, %b : f32

再进一步：

mymath.add
    ↓
arith.addf
    ↓
LLVM dialect
    ↓
LLVM IR

Day 7 建议安排为：

Day 7：Dialect Conversion / Lowering
│
├── Part 1：为什么不能只用 RewritePattern
├── Part 2：ConversionTarget
├── Part 3：Legalization
├── Part 4：Conversion Pattern
├── Part 5：applyPartialConversion
├── Part 6：applyFullConversion
├── Part 7：TypeConverter
└── Part 8：实战
       mymath → arith
       tensor type → memref / 自定义转换
二、Day 7 Part 1：为什么已经学了 RewritePattern，还需要 Dialect Conversion？

这是 Day 7 最重要的第一个问题。

你 Day 5 学的是：

class ConvAddFusionPattern
    : public OpRewritePattern<SomeOp> {
public:
  LogicalResult matchAndRewrite(
      SomeOp op,
      PatternRewriter &rewriter) const override {

    // match
    // rewrite
    // replaceOp
  }
};

它可以完成：

A
↓
B

例如：

arith.addf
    +
arith.mulf
    ↓
mydialect.fused_op

但是 RewritePattern 本身并不关心：

最后哪些 Operation 可以留下，哪些 Operation 必须全部消失？

例如你想做：

mydialect
    ↓
arith

最终要求：

module {
  arith.addf
  arith.mulf
}

不能剩下：

mydialect.add

这时你需要定义一个 目标 IR 的合法性规则。

这就是：

Dialect Conversion

核心区别可以先这样理解：

	RewritePattern	Dialect Conversion
核心目的	局部优化	IR 层级转换
是否定义合法性	否	是
是否要求旧 Op 消失	不一定	可以要求
Type 是否自动转换	否	支持
典型场景	Fusion / Simplify	Lowering
核心 API	PatternRewriter	ConversionTarget + TypeConverter

一句话：

RewritePattern：
“我想把这个东西改掉。”

Dialect Conversion：
“我的目标 IR 只能存在这些合法的东西，
所有不合法的东西都必须被转换掉。”
三、Day 7 Part 2：ConversionTarget —— 定义“什么是合法 IR”

假设我们有：

mymath.add

希望 Lowering 成：

arith.addf

首先定义 ConversionTarget：

ConversionTarget target(getContext());

target.addLegalDialect<arith::ArithDialect>();

target.addIllegalDialect<mymath::MyMathDialect>();

含义非常直接：

arith Dialect
    ↓
合法，可以保留

mymath Dialect
    ↓
非法，最终不能留下

例如原始 IR：

module {
  %0 = mymath.add %a, %b : f32
}

转换后：

module {
  %0 = arith.addf %a, %b : f32
}

如果 Pattern 没有把 mymath.add 转换掉：

Conversion Failed

因为：

target.addIllegalDialect<mymath::MyMathDialect>();

告诉框架：

这个 Dialect 里的 Operation 最终不能存在。

1. Legal Dialect
target.addLegalDialect<arith::ArithDialect>();

表示：

arith.*

全部合法。

例如：

arith.addf
arith.mulf
arith.constant

都允许存在。

2. Illegal Dialect
target.addIllegalDialect<mymath::MyMathDialect>();

表示：

mymath.*

全部非法。

最终必须：

mymath.add
        ↓
arith.addf

或者：

mymath.add
        ↓
其他合法 Operation
3. Unknown Operation

通常你还需要：

target.markUnknownOpDynamicallyLegal(
    [](Operation *op) {
      return true;
    });

意思是：

除了我明确声明 Illegal 的 Operation，其他 Operation 默认都允许保留。

这是非常常见的写法。

完整逻辑：

ConversionTarget target(getContext());

target.addIllegalDialect<mymath::MyMathDialect>();

target.addLegalDialect<arith::ArithDialect>();

target.markUnknownOpDynamicallyLegal(
    [](Operation *op) {
      return true;
    });

IR 可以理解成：

                 ConversionTarget

mymath.add        arith.addf       func.func
    ❌                ✓                ✓
必须消失           可以保留          可以保留
四、Day 7 Part 3：Legalization 到底是什么？

Legalization 是整个 Dialect Conversion 的核心概念。

假设：

输入 IR
    ↓

mymath.add
mymath.mul
mymath.relu
    ↓

Dialect Conversion
    ↓

arith.addf
arith.mulf
arith.maximumf
    ↓

所有 Operation 都合法

这个过程就叫：

Legalization

也就是说：

通过一系列 Conversion Pattern，把 Illegal Operation 转换成 Legal Operation。

例如：

mymath.add
    ↓
arith.addf

Pattern 做具体转换。

ConversionTarget

负责检查：

转换后是否合法？

所以职责划分是：

ConversionTarget
       │
       │ 定义规则
       ▼
哪些 Op 合法？
哪些 Op 非法？
       │
       ▼
Conversion Pattern
       │
       │ 负责具体转换
       ▼
mymath.add
       ↓
arith.addf
       │
       ▼
applyFullConversion
       │
       ▼
检查最终是否合法
五、Day 7 Part 4：Conversion Pattern

最简单的 Pattern 可以写成：

class AddLoweringPattern
    : public OpConversionPattern<mymath::AddOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mymath::AddOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<arith::AddFOp>(
        op,
        adaptor.getLhs(),
        adaptor.getRhs());

    return success();
  }
};

注意，这里相比 Day 5 有几个重要变化。

Day 5：

OpRewritePattern

Day 7：

OpConversionPattern

Day 5 的函数：

matchAndRewrite(
    OpT op,
    PatternRewriter &rewriter)

Day 7：

matchAndRewrite(
    OpT op,
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter)

多了：

OpAdaptor adaptor

这非常重要。

六、Day 7 Part 5：OpAdaptor 是干什么的？

这是很多初学 MLIR Dialect Conversion 时最容易困惑的地方。

先看普通 Rewrite：

%0 = mymath.add %a, %b : f32

在普通 Pattern 中：

op.getLhs()
op.getRhs()

直接拿原始 operand 就可以。

但是 Dialect Conversion 可能发生这种情况：

原始 operand
    ↓
Type Conversion
    ↓
新的 operand

例如：

tensor<1x3x224x224xf32>
          ↓
memref<1x3x224x224xf32>

这时：

op.getOperand()

拿到的是：

Original Value

而：

adaptor.getOperands()

拿到的是：

Converted Value

所以推荐形成一个固定习惯：

matchAndRewrite(
    OpT op,
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter)

使用：

adaptor.getOperands()

而不是：

op.getOperands()

可以理解为：

Original IR

%m = mymath.add %a, %b
                 │
                 │
                 ▼
        Type Conversion
                 │
                 ▼

OpAdaptor

adaptor.getLhs()
adaptor.getRhs()
七、Day 7 Part 6：完整实战——mymath.add → arith.addf

这是我建议 Day 7 的第一个完整项目。

假设 Day 6 你已经定义了：

mymath

Dialect。

其中有：

mymath.add
1. 输入 MLIR
module {
  func.func @test(%a: f32, %b: f32) -> f32 {
    %0 = mymath.add %a, %b : f32
    return %0 : f32
  }
}

目标：

module {
  func.func @test(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    return %0 : f32
  }
}
2. 定义 Conversion Pattern
class MyMathAddLowering
    : public OpConversionPattern<mymath::AddOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mymath::AddOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<arith::AddFOp>(
        op,
        adaptor.getOperands()[0],
        adaptor.getOperands()[1]);

    return success();
  }
};

逻辑：

mymath.add
    │
    │ Pattern Match
    ▼
MyMathAddLowering
    │
    │ replaceOpWithNewOp
    ▼
arith.addf
3. Pass
struct LowerMyMathToArithPass
    : public PassWrapper<
          LowerMyMathToArithPass,
          OperationPass<ModuleOp>> {

  void runOnOperation() override {
    MLIRContext &context = getContext();

    ConversionTarget target(context);

    // mymath 最终不允许存在
    target.addIllegalDialect<mymath::MyMathDialect>();

    // arith 可以存在
    target.addLegalDialect<arith::ArithDialect>();

    // func/module 允许存在
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(&context);

    patterns.add<MyMathAddLowering>(&context);

    if (failed(applyFullConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

这是 Day 7 必须彻底理解的一段代码。

4. 执行流程

整个运行过程：

runOnOperation()
      │
      ▼
ConversionTarget
      │
      ├── mymath = Illegal
      │
      ├── arith = Legal
      │
      ├── func = Legal
      │
      └── module = Legal
      │
      ▼
RewritePatternSet
      │
      ▼
MyMathAddLowering
      │
      ▼
applyFullConversion()
      │
      ▼
遍历 IR
      │
      ▼
发现 mymath.add
      │
      ▼
这是 Illegal
      │
      ▼
查找 Pattern
      │
      ▼
MyMathAddLowering
      │
      ▼
替换为 arith.addf
      │
      ▼
重新检查
      │
      ▼
全部 Legal
      │
      ▼
Success
八、Day 7 Part 7：applyPartialConversion 和 applyFullConversion

这是必须掌握的。

1. applyPartialConversion
applyPartialConversion(
    op,
    target,
    patterns);

含义：

尽可能转换，但是允许最后仍然存在 Illegal Operation。

例如：

输入：

mymath.add
mymath.special

你只有：

mymath.add
    ↓
arith.addf

Pattern。

结果可能：

arith.addf
mymath.special

即使：

mymath.special

没有转换，也可能成功。

适合：

部分 Lowering
逐步转换
Pipeline 中间阶段
2. applyFullConversion
applyFullConversion(
    op,
    target,
    patterns);

含义：

最终所有 Operation 必须合法。

如果：

mymath.add

没有 Pattern：

mymath.add
    ↓
无法转换

那么：

Conversion Failed

适合真正的：

Dialect A
    ↓
Dialect B

完整 Lowering。

3. 还有一个 applyAnalysisConversion

后面可以了解：

applyAnalysisConversion

它主要用于：

不真正修改 IR，只分析哪些 Operation 可以成功转换。

Day 7 暂时只要求知道：

Partial = 尽力转换

Full = 必须全部合法
九、Day 7 Part 8：TypeConverter

这部分是 Day 7 最重要、也是最容易理解不深的内容。

前面的例子：

mymath.add : f32
    ↓
arith.addf : f32

类型没有变化。

所以：

TypeConverter

看起来没什么用。

但是现实中的 Lowering，最重要的事情之一恰恰是：

Type Lowering

例如：

tensor<1x3x224x224xf32>
           ↓
memref<1x3x224x224xf32>

或者：

!mynpu.tensor<...>
           ↓
!llvm.ptr

这时就需要：

TypeConverter
1. 最简单的 TypeConverter
TypeConverter typeConverter;

如果你什么都不设置，它不会知道如何转换。

例如可以加：

typeConverter.addConversion(
    [](Type type) {
      return type;
    });

表示：

任何 Type
    ↓
保持不变

即：

f32
 ↓
f32

i32
 ↓
i32
2. Tensor → MemRef

假设：

tensor<1x3x224x224xf32>

转换成：

memref<1x3x224x224xf32>

可以写：

typeConverter.addConversion(
    [](RankedTensorType type)
        -> std::optional<Type> {

      return MemRefType::get(
          type.getShape(),
          type.getElementType());
    });

逻辑：

RankedTensorType
      │
      ▼
tensor<1x3x224x224xf32>
      │
      ▼
TypeConverter
      │
      ▼
memref<1x3x224x224xf32>
3. 加上 Identity Conversion

通常还需要：

typeConverter.addConversion(
    [](Type type) {
      return type;
    });

完整：

TypeConverter typeConverter;

typeConverter.addConversion(
    [](RankedTensorType type)
        -> std::optional<Type> {

      return MemRefType::get(
          type.getShape(),
          type.getElementType());
    });

typeConverter.addConversion(
    [](Type type) {
      return type;
    });

注意：

Conversion 的注册顺序

也需要理解。

MLIR 会根据注册的 Conversion 规则选择合适的转换。

十、Day 7 最重要的实战：Type + Operation 一起 Lowering

我建议 Day 7 最终不要停留在：

mymath.add → arith.addf

因为这个例子没有真正体现 TypeConverter 的价值。

应该最终做到：

mytensor.add

!mytensor.tensor<1x3x224x224xf32>

        ↓

memref + arith

一个概念上的完整流程：

Input

%0 = mytensor.add %a, %b
     : tensor<4xf32>

          │
          ▼

TypeConverter

tensor<4xf32>
       ↓
memref<4xf32>

          │
          ▼

OpConversionPattern

mytensor.add
       ↓
memref/arith operation

          │
          ▼

Legalization

只剩下 Legal IR