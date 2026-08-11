# L2cache: H800 FlashMLA / SGLang Sparse MLA Benchmark

## 测试设置

| 项目 | 设置 |
|---|---|
| GPU | 单张 NVIDIA H800 SXM 80GB，SM90，700W power limit |
| 软件 | Driver 580.126.20，CUDA 13.0.48，PyTorch 2.9.1+cu130 |
| FlashMLA | commit `15f13e5030374295491c5ce31b02d7e63a7772c6`，仅编译 SM90 |
| SGLang Q8×KV8 | commit `5d85f25f75b6b6c937ac85bdc57ba0d19ebbbd7c` 的原生 SM90 sparse prefill kernel |
| 模型形状 | `h_q=64`，absorbed `d_qk=576`，`d_v=512`，`topk=2048` |
| Softmax scale | `1/sqrt(256)`，对应 GLM-5.1 原始 `qk_head_dim=256` |
| 端到端计时 | 每 case 10 次 warmup、30 次 CUDA event 测量，steady-state |
| 图中 TFLOPS 计时 | 每个成功 case 进行 10 次 Kineto/CUDA profiler kernel 采样 |
| Dense decode | FlashMLA BF16 `flash_mla_with_kvcache` |
| Sparse decode | FlashMLA FP8-KV/BF16 compute，656 bytes/token |
| Dense chunk | FlashMLA SM90 dense decode kernel 的 causal multi-query 模式 |
| Sparse chunk | FlashMLA BF16 `flash_mla_sparse_fwd` |
| Sparse Q8 chunk | SGLang `sparse_prefill_q8kv8`，FP8 Q + FP8 KV，BF16 output |

FlashMLA 固定提交的支持矩阵没有 SM90 dense MLA prefill；它只有 SM90
dense decode 和 sparse prefill。因此本报告没有使用 FA3 充当 dense 曲线。
对 `Q>1`，dense 曲线明确使用 FlashMLA 的 SM90 dense decode kernel，设置
`cache_seqlens=H+Q`、`causal=True`，使新增 token `i` 严格可见 `H+i+1`
个 KV。这是 native FlashMLA causal multi-query 测量，不冒充官方 dense
prefill kernel。支持边界见固定提交的
[FlashMLA support matrix](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/README.md#requirements)。

Correctness 中 dense multi-query 路径对 prefix+chunk FP32 reference 的元素
通过率为 100%，`max_abs=1.706e-3`、cosine diff `1.598e-6`。新增的
Q8×KV8 kernel 对 selected-token FP32 reference 同样为 100% 元素通过，
`max_abs=7.553e-5`、cosine diff `1.009e-4`。完整测试为 12/12 通过。

## Figure 2 TFLOPS 口径

本 README 的纵坐标按 ECHO Figure 2 所称的 **hardware utilization in
TFLOPS** 重新计算，不再使用端到端有效 TFLOPS。依据 FlashMLA 固定提交的
[dense benchmark](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/tests/test_flash_mla_dense_decoding.py)，
dense 只计 `flash_fwd_splitkv_mla_kernel` 主 kernel 时间，工作量为：

```text
dense Figure 2 FLOPs = 2 * B * Q * cache_seqlen * h_q * (d_qk + d_v)
cache_seqlen          = H                  (Q=1 decode)
                      = H+Q                (Q>1 multi-query extension)
```

Sparse 沿用固定提交的
[sparse benchmark counting](https://github.com/deepseek-ai/FlashMLA/blob/15f13e5030374295491c5ce31b02d7e63a7772c6/tests/lib.py)：
只计实际选择的最多 2048 个 KV。Sparse decode 计 split-to-combine 时间，
sparse prefill 计 `sparse_attn_fwd` kernel 时间。论文图见
[ECHO Figure 2](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf)。

这是 benchmark 的名义工作量口径。尤其在 `Q>1` 且 `Q` 相对 `H` 很大时，
dense 分子按完整矩形计数，包含 causal mask 跳过的上三角，因此可能超过
`989.5 TFLOPS` 的 H100 参考分母；这不代表 H800 的真实物理吞吐超过该峰值。
原先按 causal attended pairs 和完整调用时间计算的值仍保存在
`effective_e2e_tflops`，但不用于本 README 的图。

## 参数网格

- Batch：`1, 2, 4, 8, 16, 32, 64, 128, 256`
- 历史长度：`2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K`
- 每 batch 新增长度：`1`（decode）、`2, 4, 8, 16, 32, 64, 128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K`
- 每个 `(B,Q,H)` 都分别记录 dense 和 sparse，共 2,592 个目标点

本次补测已删除 `B*Q<=32768` 和固定 60 GiB 两个人工门槛。regular case
仅在估算的实际 live tensors 超过运行时可用显存（约 78.6 GiB）时预检为
`skipped_memory_limit`；若分配或量化真实触发 CUDA OOM，则标记该点，并停止
同一 `(backend,B,Q)` 曲线的更大历史点。regular sparse index tensor 按
`B*Q*2048*4` bytes 计入实际显存需求，不再另设工作量上限。

Full-topk 只是独立诊断代理，仍保留 4 GiB index tensor 和 8K 最大可见长度
边界，不进入本 README 的 2,592 点 dense/sparse 主网格。图中的灰色 `x`
表示物理显存不足或真实 CUDA OOM，不代表 0 TFLOPS。

## 覆盖结果

本次报告专用数据集有 2,592/2,592 个目标点，0 missing：

| Q | 模式 | 成功 | 物理 OOM | Dense 最大 B（任一 H / 全 H） | Sparse 最大 B（任一 H / 全 H） |
|---:|---|---:|---:|---:|---:|
| 1 | decode | 160 | 2 | 256 / 128 | 256 / 128 |
| 2 | chunk | 160 | 2 | 256 / 128 | 256 / 128 |
| 4 | chunk | 160 | 2 | 256 / 128 | 256 / 128 |
| 8 | chunk | 160 | 2 | 256 / 128 | 256 / 128 |
| 16 | chunk | 160 | 2 | 256 / 128 | 256 / 128 |
| 32 | chunk | 160 | 2 | 256 / 128 | 256 / 128 |
| 64 | chunk | 160 | 2 | 256 / 128 | 256 / 128 |
| 128 | chunk | 159 | 3 | 256 / 128 | 256 / 128 |
| 256 | chunk | 157 | 5 | 256 / 64 | 256 / 128 |
| 512 | chunk | 156 | 6 | 256 / 64 | 256 / 64 |
| 1K | chunk | 154 | 8 | 256 / 64 | 256 / 64 |
| 2K | chunk | 144 | 18 | 128 / 64 | 256 / 64 |
| 4K | chunk | 129 | 33 | 64 / 32 | 128 / 64 |
| 8K | chunk | 113 | 49 | 32 / 16 | 64 / 32 |
| 16K | chunk | 97 | 65 | 16 / 16 | 32 / 16 |
| 32K | chunk | 80 | 82 | 8 / 8 | 16 / 8 |

总计 2,309 个成功测量、283 个明确的物理显存边界、0 missing、0 时间预算
跳过、0 最终 kernel failure。283 个边界均由 live-tensor 预检判定；经过
chunked FP8 cache 量化修正后，最终主数据中没有分配阶段的 CUDA OOM。表中
“全 H”表示该 batch 在 2K 到 512K 九个历史点全部
成功。图中 TFLOPS 使用上述 Figure 2 hardware-utilization 口径；sparse 只计
最多 2048 个 KV，因此还应结合 CSV 中的 latency、pairs 和
`effective_e2e_tflops` 查看。

最终 profiler 最大值：decode dense `383.2 TFLOPS`，decode sparse
`339.2 TFLOPS`；prefill dense 矩形名义值 `1151.5 TFLOPS`，prefill sparse
`649.5 TFLOPS`。按固定 `(backend,B,Q)` 的历史曲线检查相邻点，未发现低于
相邻几何均值 75% 的孤立性能突降。一次持续满载导致的 dense 热降频点已在
42°C 冷机状态下从 `358.3` 重测为 `665.1 TFLOPS`，最终图使用后者。

## SGLang Q8×KV8 Sparse Prefill 重测

使用 SGLang commit `5d85f25f75b6b6c937ac85bdc57ba0d19ebbbd7c` 的
[原生 SM90 Q8×KV8 sparse prefill kernel](https://github.com/sgl-project/sglang/blob/5d85f25f75b6b6c937ac85bdc57ba0d19ebbbd7c/python/sglang/kernels/ops/attention/sparse_mla_q8kv8_prefill_sm90.py)
重测上述完整 prefill 网格。vendored 8 个 CUDA/header 文件在启动时逐个校验
Git blob SHA1；JIT 只编译 `sm_90a`。

该 kernel 直接接收 contiguous `float8_e4m3fn` Q 和 KV，以及
`int32 [B*Q,1,2048]` causal indices。Q/KV 的 FP8 转换、indices 和首次 JIT
均不在 CUDA event 计时区间内，并分别记录在 setup latency。这里测的是
Q8×KV8 attention kernel，不是 SGLang 端到端 scheduler，也不是 decode 使用的
656-byte paged FP8 KV layout。

完整 Q8 网格共 1,215 点：1,116 成功、99 个明确显存边界、0 kernel failure、
0 时间预算跳过。99 个显存边界中 6 个由真实 CUDA allocation OOM 触发；其余
在分配前由实际 live-tensor 估算判定。Q8 最大吞吐为 `728.7 TFLOPS`。

与旧 FlashMLA BF16 sparse prefill 有 1,094 个同 shape 可比点。按实际选择的
最多 2048 KV 计算 FLOPs，并除以单个 attention kernel 时间，Q8×KV8 的中位
吞吐为 BF16 的 `1.263×`。按 history 分组，中位比值从 2K 的 `1.185×` 增至
512K 的 `1.478×`。在同时覆盖 2K 和 512K 的 107 条 `(B,Q)` 曲线上，Q8
吞吐的 512K/2K 中位比值为 `0.731`，BF16 为 `0.587`：FP8 减轻了长 history
下的稀疏访存损失，但没有消除该趋势。

![Q8xKV8 vs BF16 throughput ratio](assets/q8kv8/figures/q8kv8_vs_bf16_speedup.png)

下列每张图固定新增长度 Q，3×3 面板固定 batch，横轴为历史长度，纵轴为
selected-pair TFLOPS。蓝线是旧 BF16 sparse，红线是本次 Q8×KV8；灰色 `x`
是 Q8 显存不足。PNG 和 PDF 均已生成。

| Q=2 | Q=4 |
|---|---|
| ![Q8 Q=2](assets/q8kv8/figures/q8kv8_history_scaling_q2.png) | ![Q8 Q=4](assets/q8kv8/figures/q8kv8_history_scaling_q4.png) |
| Q=8 | Q=16 |
| ![Q8 Q=8](assets/q8kv8/figures/q8kv8_history_scaling_q8.png) | ![Q8 Q=16](assets/q8kv8/figures/q8kv8_history_scaling_q16.png) |
| Q=32 | Q=64 |
| ![Q8 Q=32](assets/q8kv8/figures/q8kv8_history_scaling_q32.png) | ![Q8 Q=64](assets/q8kv8/figures/q8kv8_history_scaling_q64.png) |
| Q=128 | Q=256 |
| ![Q8 Q=128](assets/q8kv8/figures/q8kv8_history_scaling_q128.png) | ![Q8 Q=256](assets/q8kv8/figures/q8kv8_history_scaling_q256.png) |
| Q=512 | Q=1K |
| ![Q8 Q=512](assets/q8kv8/figures/q8kv8_history_scaling_q512.png) | ![Q8 Q=1K](assets/q8kv8/figures/q8kv8_history_scaling_q1024.png) |
| Q=2K | Q=4K |
| ![Q8 Q=2K](assets/q8kv8/figures/q8kv8_history_scaling_q2048.png) | ![Q8 Q=4K](assets/q8kv8/figures/q8kv8_history_scaling_q4096.png) |
| Q=8K | Q=16K |
| ![Q8 Q=8K](assets/q8kv8/figures/q8kv8_history_scaling_q8192.png) | ![Q8 Q=16K](assets/q8kv8/figures/q8kv8_history_scaling_q16384.png) |
| Q=32K |  |
| ![Q8 Q=32K](assets/q8kv8/figures/q8kv8_history_scaling_q32768.png) |  |

## Decode / Prefill 长 history 机制实验

为解释“decode 基本不随 history 变慢，而 sparse prefill 会下降”，额外运行了
一组不依赖 NCU 的受控实验。固定 `topk=2048`，history 取 `2K/32K/512K`，
分别控制同一 chunk 内各 query row 的 selected-KV 集合：

- `shared`：所有 query row 读取同一组 KV，保持最大跨 query 重用；
- `independent`：每个 query row 读取不同 KV，消除大部分跨 query 重用；
- `contiguous/dispersed`：在重用条件不变时，单独改变选中地址的空间局部性；
- `isolated`：64 个 query row 位于 64 个独立 sequence，作为无同序列重用的
  prefill-kernel 对照；另测 native FlashMLA FP8 decode `B=64`。

Q8×KV8 使用 `N=64/256` query rows，FlashMLA BF16 使用 `N=64`。每个点均做
升序和降序 history 两遍，最终取两遍 median；同时测 steady 与每次调用前读取
256 MiB flush buffer 的 L2-cold。共 204/204 个原始 case 成功，得到 102 个
聚合点。升降序最大差异 `4.61%`，L2-cold/steady 的中位延迟比为 `1.015×`。

| 受控 case | 512K / 2K 延迟 |
|---|---:|
| Q8，N=256，independent dispersed | `1.475×` |
| Q8，N=256，shared dispersed | `1.003×` |
| Q8，N=64，independent dispersed | `1.234×` |
| Q8，N=64，shared dispersed | `1.003×` |
| BF16，N=64，independent dispersed | `1.220×` |
| BF16，N=64，shared dispersed | `1.002×` |
| Q8 prefill kernel，64 个独立 sequence | `1.058×` |
| native FlashMLA FP8 decode，B=64 | `1.046×` |

在 Q8、`N=256,H=512K` 下，保持 selected set 完全共享、只把地址改为离散，
延迟仅为 contiguous 的 `1.003×`；保持每行连续、只移除跨 query 重用，延迟
升至 `1.336×`；已经移除重用后再改为离散，额外增加到 `1.097×`。对应的
unique selected-KV working set 从 shared 的 2,048 token（50 MiB L2 的
`0.023×`）增长到 independent contiguous 的 524,288 token（`5.760×`）。

![Controlled Q8 N=256 history scaling](assets/cache_locality_clean/figures/controlled_sglang_q8kv8_n256_history.png)

![Decode and prefill controls](assets/cache_locality_clean/figures/decode_vs_prefill_control.png)

![Selected working set versus latency](assets/cache_locality_clean/figures/working_set_vs_latency.png)

**客观结论：**history 的可寻址范围本身不是主因；主要的软件可见因素是一个
prefill 调用内相邻 query 的 selected-KV 重叠随 history 增长而消失，导致
unique KV working set 和实际需要服务的数据增长。地址离散/空间局部性是次要
因素。Decode 每个 sequence 每次只有一个 query，不存在这种“原本可复用、随后
丢失”的同序列跨 query 重用，因此保持近似平坦。L2-cold 与 steady 很接近，
说明跨 kernel 调用的 L2 驻留也不是主因，关键重用发生在单次 kernel 内。

这组实验支持“memory-hierarchy pressure 增加”，但没有 NCU counter，不能进一步
区分 HBM 流量、L2 miss、TLB、sector efficiency 或 long-scoreboard stall 各自的
占比，也不能把结果直接表述成已测得的“带宽利用率下降”。实验使用确定性合成
indices；真实 indexer trace 若保留更高的相邻 query overlap，下降幅度会不同。
完整数值和分析见
[`assets/cache_locality_clean/analysis.md`](assets/cache_locality_clean/analysis.md)。

### 相邻 Query Overlap 匹配与反向干预

上述 shared/independent 极端对照证明跨-query selected-KV 重用会显著改变性能，
但不能单独证明 `adjacent_overlap` 这一个统计量足以解释原始 history 曲线。新增
实验复用 Q8KV8 full grid 中所有成功且 `H>2048,Q>=2` 的 shape；同一 shape 只
分配一次 Q/KV，三组只替换 indices：

| Pattern | Index 构造 | 目的 |
|---|---|---|
| `original_strided` | 原始 causal 互质步长 | 复现待解释曲线并计算真实相邻 overlap |
| `matched_contiguous` | 物理连续窗口，相邻 overlap 匹配原始值 | 去除行内离散性后检验 overlap 是否足以复现性能 |
| `inverted_contiguous` | 物理连续窗口，相邻 overlap 为 `1-original` | 构造随 history 反向变化的剂量响应 |

连续窗口使用反射步进，保证每一行都是单段连续 2,048 KV，且所有位置位于历史
prefix 内。除 `adjacent_overlap` 外同时记录全调用 `unique_kv` 和 `reuse_factor`；
原因是相邻交集相同并不保证非相邻 query 的重用拓扑或全 chunk union 相同。

本组只有在 smoke 通过并生成完整三组配对数据后才写性能结论；执行顺序为先跑
smoke，再按 55 分钟预算跑完整 shape 集并生成图表。

### Native Decode Selected-KV 重用对照

进一步对 native FlashMLA FP8 sparse decode 做单次 kernel 内的重用干预。无 NCU
无法验证“100% L2 hit/miss”，而用不同前置 kernel 预热/驱逐还会混入 GPU DVFS；
因此最终实验让两种 case 在每次计时前都执行相同的 256 MiB flush，只改变 timed
kernel 的物理 indices：

- `shared`：所有 batch row 读取同一组 2,048 KV；
- `independent`：每个 batch row 各自读取 2,048 KV。

两者的 kernel、Q/KV allocation、batch、topk、FLOPs 和立即前置 workload 完全
相同。`B<=64` 测完整 `H=4K..512K` 八个 history 点；另在
`H=4K/32K/256K` 补测 `B=128/256`。所有点均做升序/降序；三个漂移超过 5% 的
shape 以 50 repeat 重测覆盖，最终最大 pass spread 为 `3.38%`。

| Batch | Independent/shared 中位延迟比 | 范围 |
|---:|---:|---:|
| 1 | `0.999×` | `0.997–1.001×` |
| 8 | `1.032×` | `1.026–1.041×` |
| 16 | `1.042×` | `1.035–1.051×` |
| 32 | `1.059×` | `1.055–1.062×` |
| 64 | `1.081×` | `1.074–1.091×` |
| 128 | `1.096×` | `1.089–1.099×` |
| 256 | `1.159×` | `1.150–1.165×` |

![Native decode selected-KV reuse ratio](assets/decode_kv_reuse/figures/decode_shared_independent_ratio.png)

![Native decode shared/independent latency](assets/decode_kv_reuse/figures/decode_shared_independent_latency.png)

这证明 decode **可以**从 memory-hierarchy reuse 获益，因此“decode 本来完全不
命中 L2，所以没有下降空间”不成立。但即使 `B=256` 与 controlled prefill 的
`N=256` 对齐，decode 最大收益也只有 `1.165×`，明显低于 prefill 的
`1.475×`。更准确的解释是：普通 decode 各 sequence 的 unique selected-KV
数量从一开始就是 `B*2048`，而且不会随 history 增长，因此它没有一项会随
history 逐步消失的跨-query重用；prefill 的额外下降还包含 chunk 内 unique
working set 扩张以及 query-row并发访存组织。没有 counter 时不能把 observed
reuse 收益全部指定为 L2。完整分析见
[`assets/decode_kv_reuse/analysis.md`](assets/decode_kv_reuse/analysis.md)。

## Sparse Decode 延迟与历史 KV

下图只使用 native FlashMLA sparse FP8 decode。每张图固定一个 batch，横轴
为历史 KV 长度，纵轴为一次 decode 调用的 median 延迟（us），阴影为 30 次
测量的 p5-p95。灰色区域是 `skipped_memory_limit`，没有按零延迟处理。

从 4K 到各 batch 最大可运行 history，median 延迟变化范围为 `-1.54%` 到
`+2.66%`。因此在 `topk=2048` 固定后，延迟主要随 batch 增长，而没有随
历史 KV 长度显著增长：B=1 约 33 us，B=256 的可运行点约 242-245 us。

| B=1 | B=2 | B=4 |
|---|---|---|
| ![B=1 sparse decode latency](assets/figures/sparse_decode_latency_b1.png) | ![B=2 sparse decode latency](assets/figures/sparse_decode_latency_b2.png) | ![B=4 sparse decode latency](assets/figures/sparse_decode_latency_b4.png) |
| B=8 | B=16 | B=32 |
| ![B=8 sparse decode latency](assets/figures/sparse_decode_latency_b8.png) | ![B=16 sparse decode latency](assets/figures/sparse_decode_latency_b16.png) | ![B=32 sparse decode latency](assets/figures/sparse_decode_latency_b32.png) |
| B=64 | B=128 | B=256 |
| ![B=64 sparse decode latency](assets/figures/sparse_decode_latency_b64.png) | ![B=128 sparse decode latency](assets/figures/sparse_decode_latency_b128.png) | ![B=256 sparse decode latency](assets/figures/sparse_decode_latency_b256.png) |

每个 batch 也有独立 PDF。81 个目标点的数值和 skip reason 位于
`assets/raw/sparse_decode_latency_vs_history.csv`。

## Decode TFLOPS 随 Batch 变化

Decode 主图只以 batch size 为横轴。总览中的 3x3 子图分别固定九个历史
长度；每个 history 另有独立 PNG/PDF，文件名为
`assets/figures/tflops_vs_batch_q1_h{H}.{png,pdf}`。

![Decode TFLOPS vs batch](assets/figures/tflops_vs_batch_q1_overview.png)

## Prefill TFLOPS 随历史长度变化

每条曲线固定 `(B,Q)`，横轴为历史 `H=2K..512K`。下面十六张总览分别固定
新增上下文 `Q`，每张的 3x3 子图固定 batch；统一为每行两张。`Q=1` 是
decode，其余是 chunked prefill。Q>=2 的 135 组独立图位于
`assets/figures/prefill_tflops_vs_history_b{B}_q{Q}.{png,pdf}`。

| Q=1（decode） | Q=2 |
|---|---|
| ![Q=1 decode history scaling](assets/figures/history_scaling_q1.png) | ![Q=2 history scaling](assets/figures/history_scaling_q2.png) |
| Q=4 | Q=8 |
| ![Q=4 history scaling](assets/figures/history_scaling_q4.png) | ![Q=8 history scaling](assets/figures/history_scaling_q8.png) |
| Q=16 | Q=32 |
| ![Q=16 history scaling](assets/figures/history_scaling_q16.png) | ![Q=32 history scaling](assets/figures/history_scaling_q32.png) |
| Q=64 | Q=128 |
| ![Q=64 history scaling](assets/figures/history_scaling_q64.png) | ![Q=128 history scaling](assets/figures/history_scaling_q128.png) |
| Q=256 | Q=512 |
| ![Q=256 history scaling](assets/figures/history_scaling_q256.png) | ![Q=512 history scaling](assets/figures/history_scaling_q512.png) |
| Q=1K | Q=2K |
| ![Q=1K history scaling](assets/figures/history_scaling_q1024.png) | ![Q=2K history scaling](assets/figures/history_scaling_q2048.png) |
| Q=4K | Q=8K |
| ![Q=4K history scaling](assets/figures/history_scaling_q4096.png) | ![Q=8K history scaling](assets/figures/history_scaling_q8192.png) |
| Q=16K | Q=32K |
| ![Q=16K history scaling](assets/figures/history_scaling_q16384.png) | ![Q=32K history scaling](assets/figures/history_scaling_q32768.png) |

## 复现命令

```bash
cd /root/sparse_mla_benchmark
./bootstrap.sh
.venv/bin/python test_correctness.py

.venv/bin/python benchmark.py --stage full --cache-state steady --resume \
  --skip-anchors \
  --backend-roles decode-dense,decode-sparse,prefill-dense,prefill-sparse \
  --batches 1,2,4,8,16,32,64,128,256 \
  --histories 2048,4096,8192,16384,32768,65536,131072,262144,524288 \
  --prefill-lengths 2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768 \
  --time-budget-minutes 60

.venv/bin/python profile_figure2.py --repeat 10 --resume
.venv/bin/python plot_results.py

# SGLang native SM90 Q8xKV8 sparse prefill
.venv/bin/python benchmark.py --stage full --cache-state steady --resume \
  --skip-anchors --backend-roles prefill-sparse \
  --prefill-sparse-kernel q8kv8 --time-budget-minutes 60 \
  --output-dir assets/q8kv8
.venv/bin/python plot_q8kv8_results.py

# No-NCU controlled cache-locality experiment
.venv/bin/python cache_locality_experiment.py --stage full \
  --warmup 10 --repeat 30 --output-dir assets/cache_locality_clean
.venv/bin/python analyze_cache_locality.py \
  --input assets/cache_locality_clean/raw/results.jsonl \
  --output-dir assets/cache_locality_clean

# Q8KV8 prefill adjacent-overlap matched/inverted attribution
.venv/bin/python overlap_attribution_experiment.py --stage smoke \
  --warmup 10 --repeat 30 --output-dir assets/overlap_attribution_smoke
.venv/bin/python overlap_attribution_experiment.py --stage full --resume \
  --warmup 10 --repeat 30 --time-budget-minutes 55 \
  --output-dir assets/overlap_attribution
.venv/bin/python analyze_overlap_attribution.py \
  --input assets/overlap_attribution/raw/results.jsonl \
  --output-dir assets/overlap_attribution

# Native decode within-invocation selected-KV reuse
.venv/bin/python decode_kv_reuse_experiment.py --stage full \
  --warmup 5 --repeat 30 --output-dir assets/decode_kv_reuse
.venv/bin/python decode_kv_reuse_experiment.py --stage full \
  --batches 1,64 --histories 65536,262144,524288 \
  --warmup 5 --repeat 50 --output-dir assets/decode_kv_reuse_rerun
.venv/bin/python decode_kv_reuse_experiment.py --stage full \
  --batches 128,256 --histories 4096,32768,262144 \
  --warmup 5 --repeat 30 --output-dir assets/decode_kv_reuse_large_batch
.venv/bin/python analyze_decode_kv_reuse.py \
  --input assets/decode_kv_reuse/raw/results.jsonl \
  --rerun-input assets/decode_kv_reuse_rerun/raw/results.jsonl \
  --supplement-input assets/decode_kv_reuse_large_batch/raw/results.jsonl \
  --output-dir assets/decode_kv_reuse
```

## 数据文件

- `assets/raw/history_scaling_results.jsonl`：本 README 对应的 2,592 行完整记录
- `assets/raw/history_scaling_results.csv`：同一数据的 CSV 版本
- `assets/raw/history_scaling_coverage.csv`：2,592 行 decode/prefill history 轴明细
- `assets/raw/decode_tflops_vs_batch.csv`：固定 history 的 decode batch 曲线
- `assets/raw/figure2_metrics.jsonl`：2,309 个唯一成功点的追加式 profiler 指标
- `assets/raw/figure2_metrics.csv`：同一 Figure 2 指标的 CSV 版本
- `assets/raw/sparse_decode_latency_vs_history.csv`：按 batch/history 的延迟明细
- `assets/raw/results.jsonl`：所有阶段、anchor 和早期诊断的追加式总账
- `assets/raw/environment.json`：GPU、driver、版本、commit、wheel hash、pip freeze
- `assets/raw/dense_sparse_comparison.csv`：同 shape dense/sparse 对照
- `assets/raw/platform_summary.csv`：平台最大值和 batch threshold 汇总
- `assets/q8kv8/raw/results.jsonl`：Q8×KV8 的 1,215 行完整 prefill 网格
- `assets/q8kv8/raw/results.csv`：同一 Q8×KV8 数据的 CSV 版本
- `assets/q8kv8/raw/q8kv8_vs_bf16_prefill.csv`：1,215 点 BF16/Q8 对照与吞吐比
- `assets/q8kv8/raw/environment.json`：含 SGLang commit 和 8 个 source blob SHA1
- `assets/q8kv8/figures/`：Q8×KV8 对照图的 PNG/PDF
- `assets/cache_locality_clean/raw/results.jsonl`：204 个受控实验原始 case
- `assets/cache_locality_clean/raw/aggregate.csv`：升降序聚合后的 102 个点
- `assets/cache_locality_clean/raw/design.json`：局部性实验设计和固定参数
- `assets/cache_locality_clean/analysis.md`：机制分析、比值和适用边界
- `assets/cache_locality_clean/figures/`：五组受控实验 PNG/PDF
- `assets/decode_kv_reuse/raw/results.jsonl`：B<=64 的 160 个原始 shared/independent case
- `assets/decode_kv_reuse_rerun/raw/results.jsonl`：三个异常区域的 24 个 50-repeat 重测 case
- `assets/decode_kv_reuse_large_batch/raw/results.jsonl`：B=128/256 的 12 个补测 case
- `assets/decode_kv_reuse/raw/paired.csv`：最终覆盖规则合并后的 46 个配对点
- `assets/decode_kv_reuse/raw/merge_manifest.json`：base/rerun/supplement 合并优先级
- `assets/decode_kv_reuse/analysis.md`：native decode 重用实验的完整归因

SGLang 的服务端 backend 选择和 FP8 KV 支持范围仍应参考
[SGLang attention backend 文档](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md)。本次结果只对应上面固定 commit 的
独立 Q8×KV8 prefill kernel 调用，不外推为端到端 SGLang 性能。
