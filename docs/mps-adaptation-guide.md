# nano-vllm MPS 适配与调试指南

> **读者**：已完成或已理解 CPU 适配（见 [cpu-adaptation-guide.md](cpu-adaptation-guide.md)）的工程师。
> **目标**：在已有 CPU 后端的仓库上增量接入 Apple Silicon MPS 后端，掌握
> "能力探测 → 设备上下文陷阱 → 精度策略 → 性能测量" 这套 GPU 类设备的通用适配节奏，
> 并学会在 bf16 低精度下**证明实现正确**的方法。
> **参考实现**：commit `f2b7f46`（nanovllm: add MPS backend）。
> **说明**：本文方法论同样适用于后续接入新加速设备（如 torch_npu）；架构与通用调试手段
> 不再重复，重点讲 MPS 特有的问题。

---

## 1. MPS 适配与 CPU 适配的关系：什么能复用，什么不能

| 维度 | CPU 适配 | MPS 适配 | 原因 |
|---|---|---|---|
| KV cache 写入 / gather | 全部复用 | **原样复用** | 纯 torch 索引，设备无关 |
| SDPA prefill/decode | 全部复用 | 原样复用 | SDPA 原生支持 MPS |
| 分布式后端 | gloo | gloo（但 **tp 只能为 1**） | gloo 集合通信不支持 MPS 张量 |
| torch.compile | 禁用 | 禁用（实测无增益，见 §7） | Inductor-MPS 对小批量分发受限场景无效 |
| dtype | fp32 | **bf16** | 实测 bf16 矩阵乘快 2.7 倍，KV cache 减半 |
| 张量落位 | 隐式（默认设备即 CPU） | **必须显式** | 本文 §4.2 的核心陷阱 |
| 数值验证 | fp32 逐 token 一致即可 | fp32 锚点 + bf16 tie 分析 | bf16 有固有平局翻转（§5） |

> **方法论**：接入新设备先做这张"复用/新写"判表——CPU 适配留下的纯 torch 路径
> （`*_torch` 系列函数）本来就是设备无关的，能复用的不该重写。

**第一步永远是重命名**：`*_cpu` → `*_torch`。名字里的 `_cpu` 是谎言（这些函数
跑在 MPS 上），误导后来者。改名零风险但语义收益大：

```
store_kvcache_cpu   → store_kvcache_torch
gather_kvcache_cpu  → gather_kvcache_torch
sdpa_cpu            → sdpa_torch
forward_cpu         → forward_torch
```

分发逻辑保持二分：`cuda 走 flash-attn，其余走 torch 路径`。未来加新设备时，
问的问题只有一个：它属于"其余"吗（SDPA 可用吗）？

---

## 2. Phase 2：设备能力探测（不能跳过）

MPS 后端的能力子集与 CUDA 不同，**任何算子都不要假设可用**。开工前的探测脚本：

```python
import torch, torch.nn.functional as F
dev = "mps"

for dt in (torch.float32, torch.bfloat16):
    q = torch.randn(1, 16, 8, 128, dtype=dt, device=dev)   # 16 Q 头
    k = torch.randn(1, 8, 8, 128, dtype=dt, device=dev)    # 8 KV 头
    o = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, scale=0.09)
    print(dt, "SDPA + enable_gqa OK")

m = torch.ones(8, 8, dtype=torch.bool, device=dev).tril()   # bool 掩码
F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)
print("bool mask OK")

# dtype 性能对比（决定 dtype 策略）
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev); b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
torch.mps.synchronize(); t = time.perf_counter()
for _ in range(10): a @ b
torch.mps.synchronize(); print(f"bf16: {time.perf_counter()-t:.2f}s")
```

本机实测结论（记录进设计文档，作为决策依据）：

| 探测项 | 结果 | 决策 |
|---|---|---|
| SDPA `enable_gqa`（fp32/bf16） | 均可用 | 无需手动 expand 回退 |
| bool 掩码 | 可用 | chunked prefill 掩码方案可复用 |
| bf16 vs fp32 matmul | **快 2.7 倍** | MPS 走 bf16（与 CUDA 一致），KV cache 减半 |

> **探测结论要落文档**。三个月后没人记得"为什么 MPS 用 bf16"，
> 但 probe 脚本 + 数据表格可以。

---

## 3. 引擎层改动

### 3.1 dtype 策略与 Config.dtype

bf16 的收益（2.7 倍矩阵乘 + KV cache 减半）值得作为 MPS 默认，但精度应是**可配置的**：

```python
# config.py
dtype: str = "auto"          # "auto" | "float32" | "bfloat16"
assert self.dtype in ("auto", "float32", "bfloat16")

# model_runner.py
if config.dtype == "auto":
    run_dtype = torch.float32 if self.device == "cpu" else hf_config.dtype
else:
    run_dtype = getattr(torch, config.dtype)
torch.set_default_dtype(run_dtype)
```

语义：`auto` = CPU 用 fp32（CPU 指南的决策），其余设备跟随模型权重的 checkpoint dtype。
显式覆盖用于调试与验证（§5 的 fp32 锚点法就依赖这个开关）。

### 3.2 tensor_parallel_size=1 硬限制

```python
assert self.tensor_parallel_size == 1, "mps only supports tensor_parallel_size=1"
```

gloo 的集合通信（all_reduce/gather）不支持 MPS 张量。**在配置层快速失败**
比在运行时收到看不懂的报错好一百倍。

---

## 4. 两个真实的设备上下文陷阱（本文核心）

MPS 适配的代码量只有 ~30 行，但有两个 bug 值得每个适配新手背下来。
它们同根同源：

> **引擎初始化结束后会把 `torch.set_default_device("cpu")` 恢复回去**（上游为了
> 让调度器等 CPU 侧 Python 逻辑新建张量时落在主机上）。此后任何**不带显式 device**
> 的张量创建，都会落在 CPU——而模型和张量都在 MPS 上。CPU 适配时这不叫问题
> （默认设备恰好就是 CPU），MPS 上它就是 bug。

### 陷阱 1：注意力掩码落错设备

chunked prefill 掩码最初写成：

```python
torch.ones(sq, sk, dtype=torch.bool)          # 落在"当前默认设备" = CPU！
```

q/k/v 在 MPS，掩码在 CPU → SDPA 设备不匹配报错。修复：

```python
torch.ones(sq, sk, dtype=torch.bool, device=q.device)
```

**规则：函数内部构造的辅助张量，device 跟随输入张量，永远不依赖默认设备。**

### 陷阱 2：to_device 的隐式落位

CPU 版的 `to_device` 依赖"默认设备"兜底：

```python
t = torch.tensor(data, dtype=dtype, pin_memory=self.device == "cuda")
return t.cuda(non_blocking=True) if self.device == "cuda" else t    # 非 cuda → 留在默认设备
```

CPU 上兜底结果恰好正确；MPS 上 `slot_mapping`/`block_tables`/`context_lens`
全部留在 CPU，attention 里 `block_table_row`（CPU）去索引 MPS 的 cache 直接失败。
修复——显式表达目标：

```python
def to_device(self, data, dtype: torch.dtype):
    if self.device == "cuda":
        t = torch.tensor(data, dtype=dtype, pin_memory=True)
        return t.cuda(non_blocking=True)
    return torch.tensor(data, dtype=dtype, device=self.device)   # 显式！
```

**规则：数据搬运路径上不允许出现"隐式落位"。** 排查手法：全局搜不带 device 参数的
`torch.tensor(` / `torch.ones(` / `torch.zeros(`，逐个确认其设备语义。

---

## 5. bf16 下的数值验证方法论（如何证明"不是 bug"）

### 5.1 问题

bf16 精度下，nano-vllm 与 transformers 的 greedy 生成会在个别 token 分叉。
分叉 ≠ 错误——两个实现的**计算图不同**（本引擎逐序列 SDPA + 手动 gather，
transformers 整批 SDPA），核函数的浮点求和顺序不同，误差在 28 层间累积。
你需要一套方法**区分"bf16 固有数值噪声"和"实现逻辑错误"**。

### 5.2 fp32 锚点法（决定性证据）

fp32 精度足够高，若实现正确，两个实现应**逐 token 一致**（CPU 指南 §6.2 的脚本，
两边都切到 fp32 跑）。本适配实测：

```
--- case 0 (chat, float32): tokens IDENTICAL
--- case 1 (chat, float32): tokens IDENTICAL
--- case 2 (raw, float32): tokens IDENTICAL
RESULT [float32]: PASS
```

fp32 全等 ⇒ 代码路径（paged gather、批量解码、GQA、RoPE、RMSNorm）在数学上正确。
剩下的 bf16 分叉只可能是舍入噪声。**这个测试就是 MPS 适配的 L2 验收标准。**

### 5.3 tie 分析（定量刻画分叉）

对 bf16 的每个分叉点，检查分叉处参考实现的 top-1/top-2 logit 差：

```python
top2 = lg_hf.topk(2).values
gap = (top2[0] - top2[1]).item()        # 分叉步的 top1-top2 差
max_diff = (lg_nv - lg_hf).abs().max()  # 两侧 logits 的最大偏差
```

本适配实测：

```
--- case 0 (chat, bfloat16): tokens DIVERGE
    max|logit diff| = 0.3438 on logit scale 19.5
    hf logit: hf-token 19.500 vs nv-token 19.500
    hf top1-top2 gap = 0.0000 (TIE)
--- case 2 (raw, bfloat16): tokens DIVERGE
    hf logit: hf-token 16.625 vs nv-token 16.625
    hf top1-top2 gap = 0.0000 (TIE)
```

**解读**：所有分叉点的 top1-top2 gap 均为 0.0000（两个候选 token 的 logit 在 bf16
下**完全相等**，是严格平局），且两侧各自 top-1 的 logit 数值相同——
argmax 在平局上的选择由第 5 位小数的舍入决定，任何核序差异都会翻转它。
max|Δlogits| ≈ 0.3（logit 量纲 ~20 的 1.2%）与 28 层 bf16 累积噪声量级吻合。

**结论模板**（写进 commit/PR）：
> bf16 下生成链仅在 top1-top2 gap=0.0000 的严格平局处分叉，属 bf16 固有舍入特性，
> 非实现缺陷；fp32 下逐 token 一致作为正确性依据。

反例判据：如果分叉点的 gap **很大**（如 >0.5）且 fp32 也不一致 → 实现有逻辑错误，
回 CPU 指南 §6.1 走二分流程。

---

## 6. 先怀疑测试脚手架：一个真实的对照脚本 bug

bf16 初次对拍时出现大面积 MISMATCH，且 transformers 参考侧输出乱码
（`'ThePrime\n\n \n\n\n\n\nHere\n\n Thank...'`）。根因**不在引擎**，
而在对照脚本：手动 greedy 循环里追加了新 token 却**没有同步扩展 attention_mask**，

```python
cur["input_ids"] = torch.cat([cur["input_ids"], nxt.view(1, 1)], dim=1)
# attention_mask 停留在 prompt 长度 → 输入内部自相矛盾 → 参考侧输出垃圾
```

单序列无 padding 场景根本不需要传 mask，去掉后参考侧恢复正常，fp32 全等。

**教训**：对照验证出现 MISMATCH 时的检查顺序：
1. 参考侧单独跑是否正常？（generate API vs 手动循环，先信官方 API）
2. 两边 prompt/tokenizer 是否严格同源？
3. 比较逻辑本身（曾把 token id 列表和**解码后的字符串**做相等比较，恒为 False）；
4. 最后才怀疑被测实现。

---

## 7. 性能分析与测量（dispatch-bound 的完整案例）

### 7.1 现象与测量

MPS 跑通后 decode 仅 ~15-18 tok/s（bs=2），与 CPU（12-16）几乎持平——
远低于"苹果 GPU 应有的水平"。按"先测量再优化"原则分解：

- decode 每步 ~66ms ÷ 28 层 ≈ 2.4ms/层；
- 每层 ~40 个小算子 → **每步 ~1100 次算子派发**；
- 单算子计算量极小（bs=2 的 decode），派发开销 >> 计算时间 ⇒ **dispatch-bound**。

### 7.2 优化尝试一：消除隐式主机同步

原始解码是逐序列 Python 循环，每层每序列有 `.item()`（GPU→CPU 强制同步）。
改为**批量 gather + 张量掩码**，解码路径零同步：

```python
idx = bt.clamp(min=0)                       # padding 槽位 gather 块 0，随后被掩码
k_all = k_cache[idx].reshape(bs, -1, H, D)  # 一次索引取出全部序列的 KV
mask = torch.arange(kv_len, device=q.device) < context.context_lens.unsqueeze(1)
o = F.scaled_dot_product_attention(
    q.unsqueeze(2), k_all.transpose(1, 2), v_all.transpose(1, 2),
    attn_mask=mask[:, None, None], scale=self.scale, enable_gqa=True,
).squeeze(2)
```

要点：
- `context_lens` 全程留在设备上做掩码比较，**不回传主机**；
- padding 槽位（block_table 的 `-1`）clamp 到块 0 后取出垃圾数据，但被
  `arange < context_lens` 掩码精确屏蔽；
- 大批量下 gather 会展开全部 KV，设 **256 MiB 上限**，超限回退逐序列循环
  （保护 CPU 默认配置 512 序列 × 4096 长度的场景）。

实测：decode 无明显变化（15→15-18 tok/s 波动内）。**同步不是主要瓶颈**——
但这步仍然值得做：它同时让 CPU 路径受益（消除 Python 循环），且是后续任何
图捕获优化的前置条件。负结果也要记录。

### 7.3 优化尝试二：torch.compile

理论：编译融合小算子可减少派发次数。实验（dynamo 开关是调用时动态检查的，
引擎初始化后再打开即可）：

```python
llm = LLM(..., device="mps")
torch._dynamo.config.disable = False     # 让 @torch.compile 生效
outputs = llm.generate(prompts, params)
```

实测 decode 14 tok/s，**无增益**（Inductor-MPS 对本模型的小批量 eager 场景
没有产生有效融合）。结论：保留 eager，写进文档，留给未来（MPSGraph 级
图捕获或自定义 Metal kernel 是下一个量级的工程）。

### 7.4 最终性能口径

| 指标 | CPU (fp32) | MPS (bf16) |
|---|---|---|
| Prefill | ~176 tok/s | ~230 tok/s |
| Decode (bs=2) | ~12-16 tok/s | ~15-18 tok/s |
| KV cache | fp32（×2 内存） | bf16（减半） |

> **诚实汇报性能是工程素养**：MPS 解码提升有限的原因（dispatch-bound、
> 编译无效）连同测量过程一起写进 commit，比一句"支持 MPS 加速"有价值得多。
> prefill 提升 30% 且内存减半，对长上下文场景才是 MPS 的实际收益点。

---

## 8. 算子适配验收矩阵（MPS 列）

| 算子/功能 | MPS 实现方式 | 验证手段 | 状态 |
|---|---|---|---|
| KV cache 写入 | 复用 `store_kvcache_torch` | fp32 锚点全等 | ✅ |
| paged 读取（prefill） | 复用 `gather_kvcache_torch` | fp32 锚点全等 | ✅ |
| paged 读取（decode） | 批量 gather + len 掩码 | fp32 锚点全等 + bf16 tie 分析 | ✅ |
| prefill attention | SDPA + 右对齐掩码 | fp32 锚点全等 | ✅ |
| GQA | SDPA `enable_gqa` | probe + 端到端 | ✅ |
| RMSNorm/RoPE/MLP | 原实现（eager） | fp32 锚点全等 | ✅ |
| 采样 | 原实现（bf16 logits → fp32） | 链路对比 | ✅ |
| torch.compile | 禁用 | — | 已知限制 |
| tp > 1 | 配置层拒绝 | assert | 已知限制 |
| bf16 生成与参考逐 token 一致 | — | — | **不可能也不要求**（见 §5） |

---

## 9. 调试工具箱

```bash
# 提取吞吐（引擎 tqdm 输出）
python3 example_mps.py 2>&1 | grep -oE "Prefill=[0-9]+tok/s, Decode=[0-9]+tok/s" | tail -1

# 强制同步后再计时（MPS 是异步提交！）
torch.mps.synchronize()

# dtype 覆盖（验证 fp32 锚点时用）
llm = LLM(path, device="mps", dtype="float32", ...)
```

常见报错速查：

| 报错/现象 | 根因 | 修复 |
|---|---|---|
| `RuntimeError: ... expected all tensors to be on the same device` | 辅助张量/掩码落在了恢复后的默认设备（CPU） | `device=q.device` / `to_device` 显式落位（§4） |
| bf16 生成与参考不一致 | 先按 §6 排查脚手架，再按 §5.3 做 tie 分析 | — |
| decode 吞吐异常低 | 逐序列循环里的 `.item()` 隐式同步 / dispatch-bound | 批量 gather（§7.2），接受 eager 上限 |
| `Default process group has not been initialized` | 层构造期调用 `dist.get_rank()`，未 init dist | 引擎内已处理；裸调模型时先 `init_process_group("gloo", ...)` |

---

## 10. 操作清单（在上游仓库上从零执行）

```
[ ] 1. 完成（或理解）CPU 适配，确立 *_torch 共享路径
[ ] 2. *_cpu → *_torch 重命名，分发改为 cuda / 非-cuda 二分
[ ] 3. §2 能力探测脚本全绿，结论表格落档
[ ] 4. Config: device 支持 mps + dtype 选项 + tp=1 硬限制
[ ] 5. 排查隐式落位：全局搜不带 device 的张量创建（§4 两个陷阱）
[ ] 6. 解码批量化 + 256MiB 回退阈值
[ ] 7. fp32 锚点验证：与 transformers 同设备同精度逐 token 一致（L2）
[ ] 8. bf16 端到端冒烟 + tie 分析记录（§5.3 结论模板）
[ ] 9. CPU/CUDA 回归：共享路径的改动必须重跑原设备验证
[ ] 10. 性能测量 + 负结果（compile 无增益）一并记录
[ ] 11. 内核风格 commit：动机 → 模块改动 → 验证数据（见 f2b7f46）
```

> 最后一条是最容易省略也最不该省略的：**共享路径的每一行改动都是对所有后端的
> 改动**。本次适配中批量化解码触碰了 CPU 路径，因此 CPU 的 greedy 对比全部重跑。
