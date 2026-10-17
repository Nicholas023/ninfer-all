# Qwen3.6-35B-A3B in a community IQ2_M GGUF

Notes from taking apart a community `qwen35moe` artifact — a fine-tune, not an official
release — with a zero-load reader, and reading its tensor table against the engine. It is a
worked example of two things that matter before a conversion or a VRAM fit decision:

- where the bytes of a "35B-A3B" file actually go once a community mixed recipe is applied;
- which components a fine-tune recipe silently drops while its `config.json` still declares
  them.

The artifact dissected is
`Qwen3.6-35B-A3B-uncensored-heretic-IQ2_M.gguf` (11.66 GB on disk, 10.85 GiB of tensor
payload). It is a community abliterated/uncensored fine-tune of the Qwen3.6-35B-A3B weights,
quantized with an Unsloth importance matrix. **It is not an official release** and its numbers
should not be read as the shape of the official model — only as a concrete case of the
`qwen35moe` container. The full numbers reproduce with
[`tools/inspect_gguf_metadata.py`](../tools/inspect_gguf_metadata.py); the commands are at the
end of this note.

## What the container says

```
general.architecture          = qwen35moe
general.name                  = Qwen3.6 35B A3B Uncensored Heretic
general.finetune              = uncensored-heretic
general.size_label            = 35B-A3B
general.file_type             = 29            # MOSTLY_IQ2_M — and "mostly" is doing work here
general.quantization_version  = 2
GGUF.version                  = 3
GGUF.tensor_count             = 733
GGUF.kv_count                 = 45
```

The calibration is recorded in the file (`quantize.imatrix.dataset = unsloth_calibration_Qwen3.6-35B-A3B.txt`),
so the mixed recipe below is reproducible from the metadata alone.

## Architecture: a hybrid, not a dense Qwen

The architecture keys and the base `config.json` agree, and they describe the same hybrid
stack NInfer serves under the `qwen3_6_35b_a3b` recipe:

| Property | Value |
|---|---|
| Layers | 40 |
| Attention | **10 full-attention layers + 30 linear-attention (GDN) layers**, `full_attention_interval = 4` |
| Full-attention heads | 16 Q / 2 KV, `head_dim` 256, gated output (`attn_output_gate = true`) |
| Linear-attention (GDN) | `ssm.group_count` 16, `inner_size` 4096, `state_size` 128, `conv_kernel` 4, `time_step_rank` 32 |
| Experts | **256 routed + 1 shared**, top-8 per token, per-expert FFN 512 |
| Embedding | 2048, vocab 248320 |
| Position | MRoPE, `rope.dimension_count` 64 = 256 × partial 0.25, `freq_base` 1e7, interleaved sections `[11, 11, 10, 0]` |
| Context | 262144 |

The layer map confirms the interleaving exactly — every fourth block is a full-attention block:

```
blk.0  linear     blk.3  full       blk.7  full       blk.11 full
...
blk.35 full       blk.39 full
```

A full-attention block carries separate `attn_q` / `attn_k` / `attn_v` / `attn_output` plus
QK-norm; a linear block carries a fused `attn_qkv`, a `attn_gate` branch (`attn_output_gate`),
and the GDN state tensors `ssm_conv1d` / `ssm_a` / `ssm_dt.bias` / `ssm_alpha` / `ssm_beta` /
`ssm_norm` / `ssm_out`. The three-to-one ratio is the whole point of the architecture: one
quadratic-attention layer per four for recall, three constant-state layers per four that keep
no KV cache. Those two paths are exactly the engine's
[`src/ops/linear_attention/gated_delta_net`](../src/ops/linear_attention/gated_delta_net) and
[`src/ops/softmax_attention`](../src/ops/softmax_attention).

The routed experts are tiny (3 × 2048 × 512 ≈ 3.15 M parameters each), so activating eight of
256 plus the shared one is ≈ 1.1 B parameters of expert FFN per token across the stack — the
"A3B" in the label — while the parameters that are *stored* are dominated by the full expert
bank.

## Where the bytes actually go

`general.file_type = 29` says `MOSTLY_IQ2_M`, and the table shows how mixed the recipe is:

| Type | Tensors | Bytes | Share |
|---|---:|---:|---:|
| IQ2_S | 375 | 9.611 GiB | 88.6% |
| IQ3_S | 16 | 0.560 GiB | 5.2% |
| Q5_K | 1 | 0.326 GiB | 3.0% |
| Q4_K | 40 | 0.269 GiB | 2.5% |
| F32 | 301 | 0.083 GiB | 0.8% |

By tensor family (top of the census):

| Family | × | Bytes | Type | Shape |
|---|---:|---:|---|---|
| `ffn_down_exps` | 40 | 3.285 GiB | IQ2_S (3 × IQ3_S) | `[512, 2048, 256]` |
| `ffn_gate_exps` | 40 | 3.203 GiB | IQ2_S | `[2048, 512, 256]` |
| `ffn_up_exps` | 40 | 3.203 GiB | IQ2_S | `[2048, 512, 256]` |
| `output` | 1 | 0.326 GiB | **Q5_K** | `[2048, 248320]` |
| `attn_qkv` | 30 | 0.264 GiB | **Q4_K** | `[2048, 8192]` |
| `token_embd` | 1 | 0.204 GiB | **IQ3_S** | `[2048, 248320]` |
| `ffn_gate_inp` | 40 | 0.078 GiB | F32 | `[2048, 256]` |
| `attn_gate` | 30 | 0.075 GiB | IQ2_S | `[2048, 4096]` |
| `ssm_out` | 30 | 0.075 GiB | IQ2_S | `[4096, 2048]` |
| `attn_output` | 10 | 0.034 GiB | **IQ3_S** | `[4096, 2048]` |
| `attn_v` | 10 | 0.005 GiB | **Q4_K** | `[2048, 512]` |
| `ssm_conv1d` | 30 | 0.004 GiB | F32 | `[4, 8192]` |

The pattern is deliberate: push the routed experts to IQ2_S (they are ~91% of the file and the
part an importance matrix can protect least badly), keep the attention projections at
Q4_K/Q3_S, keep the LM head at Q5_K, and leave every norm, bias and small GDN state at F32
(301 tensors, but only 0.8% of the bytes). The result is 35B of parameters in 10.85 GiB of
payload — about 2.5 bits per weight — which is what makes it runnable on a small card.

This is also why a fit calculation that trusts `file_type` alone will be wrong: the expert
banks and the dense projections behave very differently under the same nominal label.

## Two components the recipe drops

`config.json` declares `mtp_num_hidden_layers: 1` and a 27-layer Vision tower, and the chat
template still carries the `<|vision_start|><|image_pad|>` placeholders. **Neither is in the
file:**

```
MTP / nextn draft head : absent
Vision tower           : absent
```

So this artifact supports neither speculative decoding (no MTP/nextn head to draft with) nor
images, whatever the base config says. A conversion that needs MTP must supply and share the
block separately, as [`docs/qwen3-8-flash-next.md`](qwen3-8-flash-next.md) describes for the
Flash-Next releases. This is the single most common surprise in community GGUF fine-tunes and
the reason the census prints both components explicitly.

## Tokenizer and chat template

- BPE (`tokenizer.ggml.model = gpt2`, `pre = qwen35`), 248320 tokens; control ids
  `<|endoftext|>` 248044, `<|im_start|>` 248045, `<|im_end|>` 248046, `<tool_call>` 248058.
- The embedded Jinja template supports `tools`, a ` thinking` reasoning block with
  `preserve_thinking`, and the multimodal placeholders — i.e. it is a thinking model with tool
  calling.
- One inconsistency worth knowing: the GGUF records `eos_token_id = 248046` (`<|im_end|>`),
  while the base `config.json` carries 248044 and `generation_config.json` lists both. The
  container wins when serving from the GGUF.

## Reproduce

The census needs no `gguf` wheel, no `numpy` and no GPU — it reads the header only:

```bash
python tools/inspect_gguf_metadata.py MODEL.gguf
python tools/inspect_gguf_metadata.py MODEL.gguf --families 30
python tools/inspect_gguf_metadata.py MODEL.gguf --json --no-tensors > census.json
```

To turn such a GGUF into a NInfer artifact, see [`docs/gguf.md`](gguf.md) for the block
formats kept in place and the `qwen3_6_35b_a3b` recipe in
[`tools/convert/official_recipes.py`](../tools/convert/official_recipes.py) for the mapping.
