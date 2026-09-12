# nano-vllm 张量并行（TP）适配与调试指南 — CPU tp=2 实例

> **读者**：已完成 CPU 适配（[cpu-adaptation-guide.md](cpu-adaptation-guide.md)）、想理解张量并行的工程师。
> **目标**：以"两个 CPU 进程、tp=2 跑 Qwen3-0.6B"为载体，学透 TP 的三个核心问题——
> **权重怎么切**、**通信发生在哪**、**多进程怎么协同**；并建立"TP 什么时候有收益"的量化判断力。
> **参考实现**：`example_tp2.py` + commit `653c4be` 之后的代码（本次适配**零代码改动**跑通，
> 原因见 §6——这本身是重要的架构结论）。
> **前置知识**：gloo 是 PyTorch 的 CPU 张量分布式后端；nccl 只支持 CUDA 张量。

---

## 1. TP 的数学原理（10 分钟版）

把一个线性层 `Y = X·Wᵀ` 拆到 2 个进程上有两种切法：

```
按输出列切（ColumnParallel）：W = [W₀; W₁] 沿输出维切
    rank_i 计算 Y_i = X·W_iᵀ  （Y_i 是 Y 的一段列）
    ✔ 输出天然分片，无需通信
    ✘ 下游若需要完整的 Y，必须再接一个按输入切分的层

按输入行切（RowParallel）：W = [W₀, W₁] 沿输入维切，X 也按维切成 [X₀, X₁]
    rank_i 计算部分和 Y_i = X_i·W_iᵀ，完整结果 Y = Y₀ + Y₁
    ✔ 输出天然完整，但需要一次 all_reduce（求和 + 广播）
```

**Transformer 的标准组合**（nano-vllm 同款）：

```
X ──▶ ColumnParallel(QKV)      [输出分片，免通信]
   ──▶ Attention（每 rank 只算自己的头）
   ──▶ RowParallel(o_proj)      [输入分片，输出 all_reduce] ──▶ 完整 hidden
X ──▶ ColumnParallel(gate_up)   [输出分片]
   ──▶ SiluAndMul（逐元素，天然分片安全）
   ──▶ RowParallel(down_proj)   [all_reduce] ──▶ 完整 hidden
```

每层 2 次 all_reduce；28 层 + embedding 1 次 + lm_head 1 次 gather。
**TP 的全部性能故事，就是这些通信和切分后计算量的博弈**（§9）。

---

## 2. 进程模型与控制面

```
主进程 (rank 0)                         spawn 子进程 (rank 1)
─────────────────────                  ─────────────────────
LLMEngine.__init__()
  ModelRunner(config, 0, [event])
    init_process_group("gloo")          ModelRunner(config, 1, event)
    建模 / 加载权重分片                    init_process_group("gloo")
    warmup / 分配 KV cache                建模 / 加载权重分片（同分片规则）
    SharedMemory(create, 1MB)             barrier()
    barrier()                             SharedMemory(attach)
    ↑ 主进程继续：tokenizer/scheduler       loop(): 死等 event
                                            读 shm → 反序列化 → 执行同名方法
```

- **控制面**：`SharedMemory(1MB)` 传 `pickle([method_name, *args])`（如 `("run", seqs, is_prefill)`），
  `Event` 做唤醒。注意 `Sequence.__getstate__` 在 decode 阶段只序列化 `last_token`
  而非全量 token_ids——控制面的带宽优化。
- **数据面**：真正的张量同步走 `dist.*` 集合通信（gloo，TCP loopback）。
- **会合（rendezvous）纪律**：集合通信要求所有 rank 以**相同顺序、相同次数**调用。
  两 rank 各自独立执行 `call("run")` → 各自进入同序的 58 次集合通信 → 天然会合。
  任何一侧多调/少调/乱序一次，另一侧就**永久阻塞**——这是 TP 调试最常见的死锁形态（§10）。

---

## 3. 权重切分全景表（Qwen3-0.6B @ tp=2）

模型配置：hidden=1024，16Q/8KV 头，head_dim=128，intermediate=3072，
vocab=151936，28 层，attention_bias=False（启用 q/k-norm），tie_word_embeddings=True。

> 代码入口：`linear.py` / `embed_head.py` 的 `weight_loader`；`packed_modules_mapping`
> 把 HF 命名（q_proj/k_proj/v_proj/gate_proj/up_proj）映射到合并参数。

### 3.1 逐张量切分表

| 张量 | 全量形状 | 每 rank 形状 | 切分方式 | 通信 |
|---|---|---|---|---|
| `embed_tokens.weight` | [151936, 1024] | [75968, 1024] | 按 vocab 行二分 | forward: all_reduce |
| `qkv_proj.weight` | [4096, 1024] | [2048, 1024] | 按头：q 8头 + k 4头 + v 4头 | 无 |
| `o_proj.weight` | [1024, 2048] | [1024, 1024] | 按输入维（= 头维）二分 | forward: all_reduce |
| `gate_up_proj.weight` | [6144, 1024] | [3072, 1024] | gate 1536 行 + up 1536 行 | 无 |
| `down_proj.weight` | [1024, 3072] | [1024, 1536] | 按输入维二分 | forward: all_reduce |
| `lm_head.weight` | [151936, 1024] | [75968, 1024] | 按 vocab 行二分（与 embed 同切法，因 tied） | forward: gather |

### 3.2 qkv_proj 的行偏移细读（最容易错的一张表）

每 rank 输出 2048 行 = `q(8头×128) + k(4头×128) + v(4头×128)`，
`QKVParallelLinear.weight_loader` 按 shard_id 定位：

| shard_id | shard_size | shard_offset（在 rank 权重行内的起点） | 全量来源 | 全量 → rank 的取法 |
|---|---|---|---|---|
| `"q"` | 8×128 = 1024 | 0 | `q_proj.weight` [2048, 1024] | `chunk(2, dim=0)[rank]` |
| `"k"` | 4×128 = 512 | 1024 | `k_proj.weight` [1024, 1024] | `chunk(2, dim=0)[rank]` |
| `"v"` | 4×128 = 512 | 1536 | `v_proj.weight` [1024, 1024] | `chunk(2, dim=0)[rank]` |

> 读法：rank0 拿 q_proj 的**前 8 个头**、k_proj 的**前 4 个头**；rank1 拿后 8/后 4。
> Attention 是按头独立的，所以"rank0 的头 0-7 配 rank0 的 KV 头 0-3"严格对齐。

### 3.3 gate_up_proj 的 shard_offset 逻辑

`MergedColumnParallelLinear`（两个输出 3072+3072 合并成一个参数）：

```
shard_size   = output_sizes[id] // tp        = 3072 // 2 = 1536
shard_offset = sum(output_sizes[:id]) // tp   # id=0 → 0；id=1 → 3072//2 = 1536
```

rank 权重的行布局 `[0,1536)=gate 分片、[1536,3072)=up 分片`，
与 `SiluAndMul` 的 `chunk(2, dim=-1)` 切分点 1536 **严格对齐**——
若 offset 公式忘了 `// tp`，silu 会作用到 up 分量上，输出静默错乱。

### 3.4 为什么这套切分是"TP 安全"的

逐算子检查**切分后每个算子是否只依赖本 rank 的数据**：

| 算子 | 是否 TP 安全 | 原因 |
|---|---|---|
| RMSNorm（hidden 1024，整行归一化） | ✅ 但只在完整 hidden 上做 | 布局在 all_reduce 之后，输入已是完整值 |
| q_norm / k_norm（每头内部归一化） | ✅ | 归一化维 = head_dim，头与头独立 |
| RoPE（按位置、按头独立旋转） | ✅ | 每个头独立计算，无需跨头信息 |
| Attention（score = q·k over 本 rank 头） | ✅ | 头维度切分正是按"头组"切的 |
| SiluAndMul（gate/up 逐元素相乘） | ✅ | gate_i 与 up_i 在同一 rank 内成对 |
| VocabParallelEmbedding（整词表查找） | ⚠ 需要通信 | 单个 token 只落在一个 rank 的分片 → 掩码置零 + all_reduce 求和 |
| lm_head（输出整个词表分布） | ⚠ 需要通信 | 每 rank 只有半张词表的 logits → gather 拼接 |

**审查方法**：对每个算子问"它的归一化维 / 归约维 / 查找维是什么？这个维被我切了吗？"
归一化维、逐元素维可以切；归约维（softmax over 词表、整行 norm）切了就必须加通信。

---

## 4. 激活张量流向（一次 decode，bs=2）

| 张量 | 形状（rank0 = rank1） | 复制 or 分片 |
|---|---|---|
| `input_ids` / `positions` / `slot_mapping` / `block_tables` | [2] / [2] / [2] / [2, L] | **复制**（两 rank 完全相同） |
| hidden_states | [2, 1024] | 复制（all_reduce 后保证一致） |
| qkv 输出 | [2, 2048] | **分片**（每 rank 自己的 8Q+4KV 头） |
| attention 输出 o | [2, 8, 128] | 分片 → o_proj 后 all_reduce 还原 |
| KV cache | [2, 28, blocks, 256, **4**, 128] | 分片（每 rank 4 个 KV 头） |
| gate_up 输出 | [2, 3072] | 分片 |
| logits（gather 前） | [2, 75968] | 分片（半张词表） |

**KV cache 一致性**：两 rank 的 `BlockManager` 对相同 token 序列算出相同哈希 →
分配出**相同的 block_table** → `slot_mapping` 一致 → 两 rank 的 cache 槽位语义对齐
（各自存不同 KV 头）。这是"复制控制面、分片数据面"能成立的关键。
验证手段见 §10（打印两 rank 的 block_table 对比）。

---

## 5. 通信清单与开销模型

每步 decode（bs=2，fp32）：

| 通信 | 次数/步 | 张量大小 | 实测单次延迟* |
|---|---|---|---|
| o_proj / down_proj all_reduce | 28×2 = 56 | 2×1024×4B = **8KB** | ~208 µs（gloo loopback） |
| embedding all_reduce | 1 | 8KB | ~208 µs |
| lm_head gather（dst=rank0） | 1 | 2×75968×4B ≈ 590KB | ~500 µs（1MB 量级） |

\* 本机（Apple Silicon，2 进程 TCP loopback）实测：8KB→208µs，1MB→543µs。

合计 ≈ 57×208µs + 0.5ms ≈ **12.4ms 通信/步**。对照 §9 的实测：decode 15→11 tok/s
（步时 ~130ms → ~180ms，差值 ≈ 50ms > 12.4ms——多出的部分来自小算子派发开销与
BLAS 线程被瓜分，见 §9.2）。

---

## 6. 为什么本次"零代码改动"就跑通了

这是整个适配最有价值的架构结论。检查 `git diff` 后的每一层：

| 层 | tp 相关逻辑 | 是否设备耦合 |
|---|---|---|
| 权重切分 / weight_loader | 纯 torch 张量操作 | ❌ 设备无关 |
| 集合通信（all_reduce/gather） | gloo 对 **CPU 张量**原生支持 | ❌（CPU 后端恰好命中） |
| 控制面（shm + Event） | 纯 Python/multiprocessing | ❌ |
| 每 rank 设备初始化 | `torch.set_default_device(config.device)` | ❌（CPU 适配的产物） |
| KV cache 按 `world_size` 分头 | `num_kv_heads // world_size` | ❌ |
| CUDA Graph / nccl / flash-attn | tp 无关，且 CPU 已强制绕开 | ✅ 仅 CUDA 路径 |

结论：**CPU 适配把"设备耦合"收敛到了有限的守卫点，剩下的代码天然支持 tp>1**。
反过来说，如果你在 GPU 适配版上直接改 `tensor_parallel_size=2` + gloo，会立刻撞上
nccl/CUDA Graph 的耦合——这印证了"耦合点清单先行"方法论的价值。

---

## 7. 复现步骤（从当前仓库）

```bash
# 1. tp=2 冒烟（两个 CPU 进程）
python3 example_tp2.py          # tensor_parallel_size=2, max_num_seqs=2, max_model_len=1024

# 2. 观察：启动时 spawn 1 个子进程；进度条吞吐来自 rank0
# 3. 结束后确认无残留进程 / 无 SharedMemory 泄漏警告
ps aux | grep example_tp2 ; ls /dev/shm 2>/dev/null || ipcs | grep nanovllm
```

> **spawn 的头号坑（本次真实踩到）**：子进程会 **重新 import `__main__`**。
> 验证脚本里若把"创建 LLM"写在模块顶层（没有 `if __name__ == "__main__":` 保护），
> 子进程 import 时会**递归创建引擎、递归 spawn**，表现为"启动后永久挂起、CPU 占用翻倍"。
> 修复：入口必须包在 `if __name__ == "__main__":` 里。上游 `example.py` 有此保护，
> 自己写脚本时极易遗漏。

---

## 8. 数值验证：三方对拍

方法（完整脚本思路）：monkeypatch `Sampler.forward = argmax`（确定性 greedy），
依次创建 tp=2 / tp=1 引擎（**先 `llm.exit()` 再 `atexit.unregister(llm.exit)`**，
否则第二次 `init_process_group` 报"already initialized"），再与 transformers fp32 greedy 对比。

本次实测（fp32，3 用例 × 32 token）：

```
--- case 0/1/2 (chat×2 + raw)
    tp2==tp1: True | tp2==transformers: True | tp1==transformers: True
RESULT: PASS   ← 逐 token 全等
```

**为什么 tp=2 能与 tp=1 全等**：fp32 下 all_reduce 的求和顺序差异（`Σᵢ Yᵢ` 的浮点
结合序 vs 单进程一次 matmul 的内部累加序）产生的偏差在 1e-6 量级，不足以翻转
greedy 的 argmax。bf16 下这个余量会缩小——若做 bf16 TP，预期在严格平局处出现
分叉，套用 MPS 指南 §5.3 的 tie-gap 分析即可。

**TP 特有的数值风险点**（fp32 也应关注，换模型/精度时复查）：
- all_reduce 语义必须是**求和**（gloo 默认 sum ✓）——若某实现误用 max/mean，输出静默错误；
- embedding 的 all_reduce 依赖"非本 rank 分片严格置零"——若掩码条件写错（如边界 token
  `x == vocab_end_idx`），同一 token 会被两个 rank 各加一次。

---

## 9. 性能实测与"TP 何时有收益"

### 9.1 实测（Qwen3-0.6B，fp32，bs=2，本机）

| 配置 | Prefill | Decode |
|---|---|---|
| tp=1 | 177 tok/s | 15 tok/s |
| tp=2 | **190 tok/s（+7%）** | **11 tok/s（-27%）** |

### 9.2 解读（这套推理过程比数字更重要）

- **Prefill 略升**：prefill 是计算密集（几千 token 一次前向），每 rank 计算量减半的收益
  > 每步 2 次 8KB all_reduce + 1 次 gather 的通信成本 → 小幅净赚。
- **Decode 下降**：三个因素叠加——
  1. 每步 57 次集合通信 × ~208µs ≈ +12ms（§5）；
  2. decode 本身 dispatch-bound（每步 ~1100 次小算子派发，见 MPS 指南 §7），
     **每 rank 算子数量不变**（只是张量变小），派发次数不减反增（多了集合通信的派发）；
  3. 单进程 BLAS 本来就能吃满所有核，切成 2 进程各持一半线程，矩阵乘效率反而降。
- **结论公式**（判断 TP 是否有收益）：

```
TP 有收益 ⇔ 计算收益（FLOPs 减半 × 利用率不塌） > 通信成本（层数 × 集合通信延迟）
          ⇔ 模型大（计算重）、batch 大（摊薄通信）、设备间带宽高（NVLink 级）
本例（0.6B + bs=2 + loopback TCP）：三条全不满足 → decode 必然更慢，实测确认。
```

### 9.3 什么时候该用 TP

- 模型单卡放不下（容量动机）——注意 Apple 统一内存下 MPS 本就能访问全部 RAM，此动机弱；
- 服务吞吐瓶颈在计算且 batch 大（吞吐动机）；
- 多机分布式（网络动机）——那是 gloo over TCP/IB 的另一个量级。

---

## 10. TP 调试手段

| 症状 | 首要怀疑 | 手段 |
|---|---|---|
| 启动即挂死、CPU 占用高 | spawn 重入 `__main__`（§7 坑） | 检查 `if __name__ == "__main__"` 保护 |
| 生成第一步挂死（进度条停在 0） | 集合通信不对称（某 rank 多/少/乱序调用） | `py-spy dump --pid <两个进程>` 看各自栈停在哪个 `dist.*`；rank 间加带 rank 前缀的日志对齐调用序 |
| 两 rank 输出不一致/乱码 | 权重切分错误（offset 公式、chunk 维度） | 单元对拍：`model.get_parameter(name)` 与全量 safetensors 的 `chunk(tp,dim)[rank]` 逐张量比对 |
| 部分 token 正常部分乱码 | embedding 掩码边界 / lm_head gather 拼接顺序 | 打印两 rank 对同一 token 的局部 embedding，验证恰有一个非零 |
| 结束后卡住不退出 | exit 的 barrier 不对称 / shm 未 unlink | `ps` 找残留进程；`ipcs`/`/dev/shm` 找泄漏 |
| tp=2 数值与 tp=1 差异大 | all_reduce 语义 / 求和顺序（bf16） | fp32 复测 + tie 分析（MPS 指南 §5.3） |

**切分正确性的快速自检脚本思路**：

```python
from safetensors import safe_open
with safe_open(shard_file, "pt", "cpu") as f:
    full_q = f.get_tensor("model.layers.0.self_attn.q_proj.weight")   # [2048, 1024]
# rank0 进程内：
param = model.get_parameter("model.layers.0.self_attn.qkv_proj.weight")  # [2048, 1024]
assert torch.equal(param.data[:1024], full_q[:1024])   # rank0 = chunk(2)[0]
```

---

## 11. TP 验收矩阵

| 验收项 | 标准 | 状态 |
|---|---|---|
| tp=2 端到端生成 | 冒烟输出连贯（L1） | ✅ |
| 三方逐 token 对拍 | tp=2 == tp=1 == transformers（L2，fp32） | ✅ |
| 权重切分 | 每类参数与全量 chunk 逐张量相等 | ✅（加载无 assert 失败 + 对拍通过） |
| KV cache 一致性 | 两 rank block_table 相同（日志验证） | ✅（数值对拍间接证明） |
| 进程生命周期 | 正常退出、无残留进程/shm 泄漏 | ✅ |
| 性能口径 | 如实记录：prefill +7% / decode -27%，并解释原因 | ✅（§9） |
| 约束 | tp ≤ 8 且必须整除头数/intermediate/vocab（`divide()` assert 兜底） | ✅ |

---

## 12. 进阶预告：异构 TP（MPS rank0 + CPU rank1）

在本文基础上加两件事即可（留作练习）：

1. **per-rank 设备**：`Config` 增加秩→设备映射（rank0="mps"、rank1="cpu"），
   spawn 时分别传入；注意 MPS 分支已强制 tp=1 的 assert 要放开并替换为该映射逻辑。
2. **MPS 张量的通信包装**：`c10d::allreduce_` 不支持 MPS 设备（实测报错），
   需要在集合通信前后做 MPS→CPU→MPS staging；每次 staging 是一个强制同步点。
3. **预期**：bs=2 下比单 MPS 更慢（MPS 指南 §7 的 dispatch-bound 结论 + 本指南 §9 的
   通信账叠加）——做完就能用数据回答"MPS rank0 值不值"，这正是这份练习的意义。
