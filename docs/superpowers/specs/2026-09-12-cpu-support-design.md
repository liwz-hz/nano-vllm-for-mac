# nano-vllm CPU 支持（第一阶段）设计

日期：2026-09-12
状态：已批准（方案 A：原地改造 + device 分支）

## 背景

nano-vllm 当前硬耦合 CUDA（flash-attn、triton、nccl、CUDA Graph、显存探测）。
目标：在 Mac CPU 上跑通 Qwen3-0.6B（ModelScope 缓存
`~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`），小规模参数
（max_num_seqs=2、max_model_len=1024、max_tokens=64）。MPS 支持为第二阶段。

模型确认：ModelScope 下载，权重/tokenizer/config 齐全（model.safetensors 1.5GB）。

## 方案

原地最小改造，新增 `device` 配置；GPU 路径保持原样，CPU 走新分支。

### 1. config.py
- 新增 `device: str = "cpu"`，允许 `cpu` / `cuda` / `mps`，不可用值报错。
- 其余默认值不变；小参数由 example_cpu.py 传参。

### 2. layers/attention.py
- `flash_attn` / `triton` 改为条件导入（ImportError 时置标志），CPU 环境无需安装。
- CPU KV cache 写入：纯 PyTorch 索引（`slot != -1` 掩码 + 平坦视图赋值），
  替代 triton kernel。
- CPU 注意力前向：`F.scaled_dot_product_attention`，`enable_gqa=True`（16Q/8KV）：
  - Prefill：按 `cu_seqlens_q` 逐序列切 q；`block_tables` 非空时按块从 paged cache
    展开 k/v（`cache[bt[:nblocks]].reshape(-1, H, D)[:seqlen_k]`），否则切当前 k/v。
    `sq == sk` 用 `is_causal=True`；chunked prefill（sq < sk）用
    `tril(diagonal=sk-sq)` 布尔掩码。
  - Decode：逐序列 gather k/v，单 token query 做 attention（`is_causal=False`）。
- GPU 路径 flash_attn 原样保留，按 `k_cache.device.type` 分支。

### 3. engine/model_runner.py
- dist 后端：cuda 用 nccl，否则 gloo（tp=1 也需 init，layer 构造依赖
  `dist.get_rank/world_size`）。
- `torch.cuda.set_device` / `torch.cuda.synchronize` / `empty_cache` 仅 cuda 执行。
- 默认 dtype：cpu 用 float32（bf16 矩阵乘在 Mac CPU 上慢且不稳），cuda 保持
  `hf_config.dtype`。
- CPU 禁用 torch.compile（`torch._dynamo.config.disable = True`，layernorm/
  rotary/sampler/activation 的 @torch.compile 自动回落 eager）。
- CPU 强制 eager（跳过 CUDA Graph）。
- `pin_memory=True + .cuda(non_blocking)` 收敛为 `to_device(data, dtype)` 辅助方法：
  cuda 行为不变，cpu 返回普通张量。
- KV cache 容量：cuda 沿用显存探测；cpu 按
  `max_num_seqs × ceil(max_model_len/block_size) + 1` 精确计算，并以可用物理内存
  25% 为上限（`os.sysconf`）。block_bytes 的 itemsize 用 `torch.get_default_dtype()`
  （分配时与原逻辑在 cuda 上等价）。
- warmup：cuda 保持全量 warmup（显存峰值估算依赖它）；cpu 用 1×16 token 迷你
  warmup（仅为打通代码路径，避免 CPU 上数千 token 的预热耗时）。

### 4. pyproject.toml
- `flash-attn`、`triton` 移入 `[project.optional-dependencies] cuda`。

### 5. example_cpu.py（新增）
- 模型路径指向 ModelScope 缓存；`max_num_seqs=2`、`max_model_len=1024`、
  `max_tokens=64`、`enforce_eager=True`；chat template 用 `enable_thinking=False`
  （64 token 内 think 块会耗尽预算）。原 example.py 不动。
- 直接用 conda base（torch 2.13 / transformers 5.8）运行，不做 pip 安装。

## 错误处理
- device 非法/后端不可用：`__post_init__` assert。
- CPU 环境 import flash_attn/triton 失败：条件导入，仅 GPU 路径使用时才要求安装。

## 验证
1. `python example_cpu.py` 两条 prompt 输出连贯合理。
2. 数值对照：monkeypatch Sampler 为 argmax（greedy），与 transformers
   `AutoModelForCausalLM`（fp32 CPU）greedy 生成对比前若干 token 一致。
