import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context

try:
    import triton
    import triton.language as tl

    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


if HAS_FLASH_ATTN:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


    def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_torch(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    mask = slot_mapping != -1
    slots = slot_mapping[mask]
    N, num_heads, head_dim = key.shape
    k_flat = k_cache.view(-1, num_heads, head_dim)
    v_flat = v_cache.view(-1, num_heads, head_dim)
    k_flat[slots] = key[mask]
    v_flat[slots] = value[mask]


def gather_kvcache_torch(cache: torch.Tensor, block_table_row: torch.Tensor, seqlen: int) -> torch.Tensor:
    num_heads, head_dim = cache.shape[-2:]
    block_size = cache.shape[1]
    nblocks = (seqlen + block_size - 1) // block_size
    return cache[block_table_row[:nblocks].long()].reshape(-1, num_heads, head_dim)[:seqlen]


def sdpa_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    sq: int,
    sk: int,
) -> torch.Tensor:
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    if sq == sk:
        attn_mask, is_causal = None, True
    elif sq == 1:
        attn_mask, is_causal = None, False
    else:    # chunked prefill: bottom-right causal alignment
        attn_mask, is_causal = torch.ones(sq, sk, dtype=torch.bool, device=q.device).tril(diagonal=sk - sq), False
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale, enable_gqa=True)
    return o.squeeze(0).transpose(0, 1)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if k_cache.device.type == "cuda":
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            else:
                store_kvcache_torch(k, v, k_cache, v_cache, context.slot_mapping)
        if k_cache.device.type != "cuda":
            return self.forward_torch(q, k, v, context, k_cache, v_cache)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    def forward_torch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context, k_cache: torch.Tensor, v_cache: torch.Tensor):
        o = torch.empty_like(q)
        if context.is_prefill:
            cu_q = context.cu_seqlens_q.tolist()
            cu_k = context.cu_seqlens_k.tolist()
            for i in range(len(cu_q) - 1):
                start, end = cu_q[i], cu_q[i + 1]
                seqlen_k = cu_k[i + 1] - cu_k[i]
                if context.block_tables is not None:    # prefix cache / tokens already in paged cache
                    k_i = gather_kvcache_torch(k_cache, context.block_tables[i], seqlen_k)
                    v_i = gather_kvcache_torch(v_cache, context.block_tables[i], seqlen_k)
                else:
                    k_i = k[cu_k[i]:cu_k[i + 1]]
                    v_i = v[cu_k[i]:cu_k[i + 1]]
                o[start:end] = sdpa_torch(q[start:end], k_i, v_i, self.scale, end - start, seqlen_k)
        else:    # decode
            bt = context.block_tables
            bs, max_blocks = bt.shape
            block_size, num_heads, head_dim = k_cache.shape[-3:]
            kv_bytes = bs * max_blocks * block_size * num_heads * head_dim * q.element_size()
            if kv_bytes <= 256 * 1024 * 1024:    # batched: no host sync, matters for mps
                idx = bt.clamp(min=0)    # padded slots gather block 0, masked out below
                k_all = k_cache[idx].reshape(bs, -1, num_heads, head_dim)
                v_all = v_cache[idx].reshape(bs, -1, num_heads, head_dim)
                kv_len = k_all.size(1)
                mask = torch.arange(kv_len, device=q.device) < context.context_lens.unsqueeze(1)
                o = F.scaled_dot_product_attention(
                    q.unsqueeze(2), k_all.transpose(1, 2), v_all.transpose(1, 2),
                    attn_mask=mask[:, None, None], scale=self.scale, enable_gqa=True,
                ).squeeze(2)
            else:    # large batch: per-sequence loop keeps the gather bounded
                for i in range(q.size(0)):
                    seqlen = context.context_lens[i].item()
                    k_i = gather_kvcache_torch(k_cache, bt[i], seqlen)
                    v_i = gather_kvcache_torch(v_cache, bt[i], seqlen)
                    o[i:i+1] = sdpa_torch(q[i:i+1], k_i, v_i, self.scale, 1, seqlen)
        return o
