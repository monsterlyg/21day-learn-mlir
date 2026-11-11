# Day 17：模型导入——先读图，再读 IR

## 今天要解决什么问题？

前端把 ONNX/PyTorch 模型交给编译器时，并不是把一段文本「翻译」成 MLIR。它做的是一次**信息映射**：

```text
模型里的 graph / node / initializer / attribute
        ↓
MLIR 里的 module / func / op / constant / attribute
```

如果不先看清楚模型里到底有什么，直接读导入后的 IR 很容易迷路：这个 `1x16x16x8` 从哪来？`pads=[1,1,1,1]` 为什么变成这样？为什么多了一个 transpose？

今天的目标：

1. 用本机的 PyTorch + ONNX 导出一个小模型，直接读图。
2. 建立 ONNX 概念到 MLIR 概念的映射。
3. 认识本机工具链的边界：没有现成 ONNX importer 时，怎样继续练习。
4. 亲手把 ONNX Conv 的参数翻译成 TOSA。

## 环境

```bash
MLIR_OPT=/data8/liyinggang/Projects/CXX/llvm-project/lyg-install/bin/mlir-opt
python3 --version
python3 -c "import torch, onnx; print(torch.__version__, onnx.__version__)"
```

本机已验证：PyTorch 2.4、ONNX 1.14。若版本不同，导出的 opset 和节点属性可能略有差异。

## 第一站：导出一个最小模型并读图

准备一个 `Conv2d + ReLU`：

```python
import io
import torch
import torch.nn as nn
import onnx

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

m = M().eval()
buf = io.BytesIO()
torch.onnx.export(
    m,
    torch.randn(1, 3, 16, 16),
    buf,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
)
model = onnx.load_model_from_string(buf.getvalue())

print("ir_version:", model.ir_version)
print("opset:", [(o.domain, o.version) for o in model.opset_import])

for inp in model.graph.input:
    print("input:", inp.name, inp.type.tensor_type.shape)

for node in model.graph.node:
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    print("node:", node.op_type, attrs)
```

实际输出（本机）：

```text
ir_version: 8
opset: [('', 17)]
input: input dim { dim_value: 1 } dim { dim_value: 3 }
       dim { dim_value: 16 } dim { dim_value: 16 }
node: Conv {'dilations': [1, 1], 'group': 1, 'kernel_shape': [3, 3],
            'pads': [1, 1, 1, 1], 'strides': [1, 1]}
node: Relu {}
```

先不要急着转成 MLIR。问自己四个问题：

1. **输入 layout 是什么？** PyTorch/ONNX 是 NCHW：`1x3x16x16`。
2. **卷积分组是多少？** `group=1`，普通卷积。
3. **padding 是多少？** `pads=[1,1,1,1]`，在 ONNX 里是 `[top, bottom, left, right]`。
4. **ReLU 是独立节点还是属性？** 独立节点，`op_type=Relu`。

这四条信息决定了后面 MLIR IR 里必须出现什么。

## 第二站：ONNX 和 MLIR 的概念映射

| ONNX | MLIR | 说明 |
|---|---|---|
| `graph` | `module` / `func.func` | 一个计算图通常落成一个函数 |
| `node` | 某个 dialect 的 op | `Conv` 可能变成 `tosa.conv2d`、`top.Conv` 或 `linalg` |
| `initializer` | `arith.constant` / `tosa.const` | 权重和 bias 变成常量 |
| `attribute` | op 的 attribute | `pads`、`strides`、`dilations` |
| tensor shape | `tensor<...>` 类型 | 静态维度写数字，动态维度写 `?` |
| `opset` | 前端/importer 的版本信息 | 影响支持哪些算子 |

最重要的差异是 **layout**：

```text
ONNX Conv:  input NCHW, weight OIHW
TOSA Conv:  input NHWC, weight OHWI
```

同一份权重，从 OIHW 到 OHWI 需要转置，或者由 importer 在生成常量时调整。读导入后的 IR 时，先检查 layout，不要只看 shape 数字。

## 第三站：本机工具链的边界

本机的 LLVM/MLIR 安装**没有 ONNX importer**。`mlir-opt` 只认识 MLIR，`mlir-translate` 也没有 `--import-onnx` 之类的选项。所以「把 `model.onnx` 直接变成 MLIR」这一步，需要额外的前端。

常见的三条路径：

### 路径 1：torch-mlir

```bash
# 需要先安装 torch-mlir（本机未安装）
python -m torch_mlir.tools.import_onnx model.onnx -o model.mlir
# 或从 PyTorch 直接导出
python -m torch_mlir.tools.torchscript ...
```

产物通常是 `torch` 方言或 `linalg`，之后可以继续降到 `tosa`/`stablehlo`。

### 路径 2：onnx-mlir

```bash
# 需要安装 onnx-mlir
onnx-mlir --EmitMLIR model.onnx
```

产物通常是 `onnx` 方言或 `krnl` 方言。

### 路径 3：TPU-MLIR

```bash
# 需要 TPU-MLIR 环境
python tools/model_transform.py \
  --model_name conv \
  --model_def conv.onnx \
  --input_shapes [[1,3,16,16]] \
  --mlir conv.mlir
```

产物是 TPU-MLIR 的 `top` 方言 IR，这是 Day 20/21 的主角。

三条路径的共同点是：**导入器负责方言选择和 layout 转换**，不是 `mlir-opt` 的职责。

## 第四站：没有 importer 也能练

我们可以把「导入产物」手写成等价的 TOSA IR，然后继续做后面所有天的练习。根据第一站的 ONNX 信息：

- input：`1x3x16x16` NCHW → TOSA `1x16x16x3` NHWC
- weight：`8x3x3x3` OIHW → TOSA `8x3x3x3` OHWI（这里 H=W=3，转置后形状数字碰巧一样）
- bias：`8`
- pads：`[1,1,1,1]`
- strides：`[1,1]`
- output：`1x16x16x8` NHWC

写成 TOSA：

```mlir
module {
  func.func @conv_relu(%input: tensor<1x16x16x3xf32>,
                       %weight: tensor<8x3x3x3xf32>,
                       %bias: tensor<8xf32>) -> tensor<1x16x16x8xf32> {
    %zp = arith.constant dense<0.0> : tensor<1xf32>

    %conv = tosa.conv2d %input, %weight, %bias, %zp, %zp {
      acc_type = f32,
      dilation = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>,
      stride = array<i64: 1, 1>
    } : (tensor<1x16x16x3xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>,
         tensor<1xf32>, tensor<1xf32>) -> tensor<1x16x16x8xf32>

    %relu = tosa.clamp %conv {min_val = 0.0 : f32, max_val = 3.4e38 : f32}
      : (tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>

    return %relu : tensor<1x16x16x8xf32>
  }
}
```

验证（把上面的 module 保存为 `conv_relu.mlir`）：

```bash
$MLIR_OPT --tosa-validate conv_relu.mlir
```

手算 output：

```text
OH = (16 + 1 + 1 - 3) / 1 + 1 = 16
OW = (16 + 1 + 1 - 3) / 1 + 1 = 16
OC = 8
```

这和 ONNX 的输出 shape 一致。注意这里 ReLU 用了 `tosa.clamp`，因为本机 TOSA 版本没有 `tosa.relu`（Day 16 已确认）。

## 第五站：导入后必须检查的五件事

拿到任何 importer 的产物，先按这个清单过一遍：

1. **shape**：每个 op 的结果 shape 是否和模型一致？动态维是否合理？
2. **layout**：NCHW/NHWC、OIHW/OHWI/FHWC 是否统一？
3. **dtype**：f32/f16/i8、量化类型是否正确？
4. **attribute**：pads/strides/dilations/group 是否原样保留？
5. **未支持 op**：有没有 fallback 成通用 op、或者直接报错？

把这张清单写进你的实验记录，比「看起来跑通了」可靠得多。

## 今天的练习路线

### 练习 A：改一个参数

把 ONNX 脚本里的 `padding=1` 改成 `padding=0`，重新导出并读图。然后修改 TOSA 例子的 `pad`，验证 output shape 是否跟着变。

### 练习 B：读 weight 的 layout

打印 PyTorch `m.conv.weight.shape`，再对照 ONNX initializer 的形状，解释 `OIHW` 和 `OHWI` 的差别。尝试写一个小 Python 脚本做转置，确认形状变化。

### 练习 C：加一个算子

把模型改成 `Conv + ReLU + MaxPool2d`，读 ONNX 图，然后查 TOSA 里对应的 `tosa.max_pool2d` 参数，写进 TOSA 例子。

### 练习 D：负例

把 TOSA 例子的 `pad` 从 `[1,1,1,1]` 改成 `[1,0,1,0]`，先手算新的 output shape，再用 `--tosa-validate` 验证。观察错误或 shape 变化，理解 pad 顺序的重要性。

## 常见坑

- **以为 `mlir-opt` 能导入 ONNX**：它只处理 MLIR 文本/字节码。
- **layout 直接照抄**：ONNX 的 NCHW 和 TOSA 的 NHWC 是两套约定。
- **pad 顺序**：ONNX/TOSA 都是 `[top, bottom, left, right]`，不是 `[H, W]`。
- **opset 差异**：不同 opset 的节点属性可能不同，读图时要打印出来确认。
- **只看 shape 不看 dtype**：`i8` 和 `f32` 的 Conv 语义不同，后面量化会用到。

## 今天结束时，试着讲给别人听

1. 模型导入时，哪些信息必须保留，哪些可以重新推导？
2. ONNX 的 `Conv` 和 TOSA 的 `conv2d` 在 layout 上差在哪？
3. 为什么 `mlir-opt` 本身不能完成 ONNX 导入？
4. 导入后检查清单里的五项分别防什么错误？
5. 如果 importer 不支持某个 op，你会怎样定位和替代？

## 参考资料

- [ONNX Operators](https://onnx.ai/onnx/operators/)
- [torch-mlir](https://github.com/llvm/torch-mlir)
- [onnx-mlir](https://github.com/onnx/onnx-mlir)
- [TPU-MLIR](https://github.com/sophgo/tpu-mlir)
- 本机源码：`mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td`
