# 融合 int8 SDPA → Burn f32 attention

> 2026-09-03。要证明的判断：VNNI + 融 DQL 之后，512 tok 仍剩 ~1.2 s
> **既不是 GEMM 也不是 DQL**；ORT 赢在「一整块 Transformer 当执行单元 +
> 不物化 `[heads,S,S]`」。对拍数字见 `notes/poc-results.md`「融合 attention」。

## 做了什么

e5 的 ONNX **没有** `Attention` 节点，也没有 f32 `MatMul`。每层是：

```
Q/K/V proj (MMI+dequant) → reshape [B,S,12,32]
Q: permute [0,2,1,3] → [B,H,S,D]
K: permute [0,2,3,1] → [B,H,D,S]   // 已是 K^T
V: permute [0,2,1,3] → [B,H,S,D]
DQL(Q), DQL(K), DQL(V)
QK: MMI → Cast → Mul(q_s*k_s) → Div(√32) → Add(mask) → Softmax
DQL(softmax)   // 最大的 DQL，[12,S,S] f32
PV: MMI → Cast → Mul → permute → reshape [B,S,384]
```

`onnx-ir` 已有的 `coalesce_attention` 只认 f32 `MatMul → Softmax → MatMul`，
对不上。这次在同一 pass 里加了 int8 模式：

1. **`vendor/burn-onnx-coalesce-int8-attn` `f78e156`**
   - 从 Softmax 往回剥 Add(mask) / Div(√d) / 反量化 Mul / Cast，落到
     `MatMulInteger`；Q/K/V 取 DQL **之前** 的 f32。
   - 用已有 `Attention` 节点替换 PV 反量化 Mul（输出名不变）。
   - K 的 `[0,2,3,1]` 相对 Q 的 `[0,2,1,3]` 是末两维对调，复用 corrective
     `Transpose [0,1,3,2]`。
   - 运行时 `q_s*k_s` **不**折进 `Attention.scale`。mask 是图级共享的
     `[B,1,1,S]` 加法 bias，不要求 single-use。
   - 单测 4 条（e5 布局、共享 mask、动态 Div、缺 softmax DQL 不匹配）+
     原有 25 条 f32 SDPA 仍过。

2. **`vendor/burn-flash-512` `245ab35`**
   - flex `attention()` 的 naive 预算从 `<= 256K` 改成 `< 256K`。
   - 512×512 = 256K 刚好压在旧边界上，会物化 `[H,S,S]`；改完后走 flash
     （online softmax，不物化完整 scores）。16 tok 仍走 naive。

没有新 kernel，没有 `unsafe`。数值会相对 ORT 的 int8 QK/PV 路径漂移，
预期仍在 0.99x；若崩再改 streaming int8。

## 验收（用来证明或证伪判断）

- 生成代码出现 `burn::tensor::module::attention`，12 个 Softmax 消失。
- 正确性：mean cos 仍约 0.99x；top-1 检索不崩。
- 性能：512 从 1.27 s 掉到 **约 300–500 ms** 才算分水岭（相对 54 ms ORT
  进个位数倍）。若几乎没动，说明瓶颈不在 `[S,S]`，判断被证伪。
- 内存：4×512 HWM 相对 315 MB 应明显下降。
- 短句 16 tok 可能几乎不动（调度税 + naive 小路），与先前判断一致。
