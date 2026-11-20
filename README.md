# DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw) on two DGX Spark (GB10), TP = 2

> **Archived.** This configuration (a third-party 2.9 bpw quant on the TensorFold 0.6.6 line) is superseded by [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10). This repository is read-only.

A working two-node configuration for DeepSeek's 1017 update, with the measured rows from our own pair.

> **Weights:** [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
> at revision `afc7aa0a` (MIT), an EXL3 2.9 bpw quantization of
> [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision
> `1bae0e10` (MIT, © DeepSeek). We did not make or modify these weights; this repository holds no weights. The Engram
> tables are read at run time from DeepSeek's original shards 47 and 48.
>
> **Engine:** TensorFold with the DeepSeek-V4.1 family written by Bertholomus AI, at the pinned public build
> `07aa02585f64` (TensorFold 0.6.6 line plus the V4.1 engine and the agent-facing changes in
> [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold), Apache-2.0; credited below).
> **Scripts:** the launcher scripts from the pinned public two-node recipe
> [Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark)
> (Apache-2.0). This repository adds the configuration we verified and the numbers we measured.

## What you need

- Two NVIDIA DGX Spark (GB10) nodes, joined by one QSFP cable between their ConnectX-7 ports.
- About 500 GB free disk on each node (the weights use ~196 GiB; the Engram shards ~95 GiB each; the first start
  writes a rank cache).
- About 100 GB free memory on each node. On GB10 host memory is GPU memory: nothing else may hold the pair.
- ssh from the head node to the worker, and fixed addresses on the cable link.

## Configuration we verified

`.env` in this repository is the file that ran. The values that differ from the upstream `.env.example`:

| Setting | Value | Why |
|---|---|---|
| `TF_COMMIT` | `07aa02585f64` | the engine build we measured |
| `CONTEXT` | `262144` | a conservative value we shipped for a stable first lane; the engine supports the full 1M window. See the memory note |
| `MEM_FLOOR_GIB` | `0` | see the memory note below |
| `PARALLEL` | `4` | four concurrent requests |
| `MTP_DRAFTS` | `5` | DSpark draft depth |
| `BIND_HOST` | `0.0.0.0` | serve the LAN, not only loopback |

Paths in the file use `$HOME`; set `HEAD_IP`, `WORKER_IP`, `WORKER_SSH` and the RoCE device names for your pair
(`ibdev2netdev` and `ip -br addr` show them).

## Run it

On the head node, in a clone of the pinned scripts repository:

```sh
cp .env.example .env          # then edit the addresses and paths
scripts/download.sh           # weights + Engram, sha256-checked, then copied to the worker
scripts/patch_template.py --worker
docker/build.sh               # builds the engine image and ships it to the worker
scripts/host_tuning.sh        # swap off for the engine, THP off, earlyoom; needs sudo on both nodes
./start.sh
./verify.sh                   # five real requests
```

Replace that clone's `.env` with the one in this repository, and keep `TF_COMMIT`/`IMAGE` as they are.

## Measured rows

One pair of DGX Spark (GB10), TP = 2, engine `07aa02585f64`, the configuration above, weights at `afc7aa0a`.
Decode counts completion tokens per second after the first token, temperature 0, thinking off unless stated.

| Row | tok/s |
|---|---|
| Decode, 1 stream: code / prose / structured | **97.4 / 59.2 / 143.5** |
| Decode, 1 stream, second prompt set: code / prose / structured | 86.7 / 60.0 / 105.9 |
| Aggregate decode, 2 streams / 4 streams | 83.4 / 117.9 |
| Aggregate decode, 4 streams sustained for 90 s | **120.9** |
| Prefill, cold prompt: 8K / 32K / 64K / 128K | 2,079 / 1,779 / 2,086 / 1,979 |

`verify.sh`: 5 of 5 passed (plain reply, thinking reply, strict JSON schema, `tool_choice: required`, regex).
Boot: about 40 s once the rank cache exists.

Method: the two benchmark clients in the pinned scripts repository, single run each, idle server, warm engine.
Run-to-run spread on this class of lane is about 10 %. The 32K prefill row is one cold prompt; a second run moved
it by several percent.

## The GB10 memory note (read this before you raise `CONTEXT`)

The engine sets a torch per-process memory ceiling after its warm-up:

```
ceiling = reserved + max(0, host MemAvailable - TF_DS_MEM_FLOOR_GIB)
```

The ceiling counts non-PyTorch memory too (NCCL buffers, the driver context, about 2 GiB on this build). On a lean
pair the post-boot `MemAvailable` is only 2-3 GiB, so with any floor above that the ceiling lands at or below what
the process already holds, and the first transient allocation fails with `CUDA out of memory ... Tried to allocate
2.00 MiB` while several GiB look free. We hit that three times: at floors of 4, 8 and 16 GiB.

What worked: `MEM_FLOOR_GIB=0` (no self-ceiling; a real out-of-memory then fails one request instead of the lane).
We also set `CONTEXT=262144` at that point, as the safe value for the window we had; that is a choice, not a limit.
The engine takes the full 1,048,576 positions, and the upstream two-node recipe ships that value.

The KV pool is sized from the context at about 1.4 KiB per position, so a 1M window costs roughly 2.5 GiB more than
262K. With the ceiling off that memory comes straight out of the free device memory (about 3 GiB at idle on our
pair), so 1M is tight but plausible. It is untested here: our 1M runs are the ones that failed, but they failed
under a floor, before we found the ceiling trap. If you want 1M, raise `CONTEXT` and hold it with a long-prompt run
that reads `MemAvailable` afterwards; drop back to 262K if it does not hold.

`host_tuning.sh` is not optional on this class of node: without it the kernel swaps and compacts under a long
prefill and the same request takes 10-30 % longer.

## What this is not

- Shipped at 262,144 positions. The engine supports 1M; we have not verified 1M on this configuration (see the
  memory note).
- Not measured on your hardware. The rows above are one pair, one day, one run each.
- The vision tower is on; image requests work but are not in the rows.

## License and attribution

- DeepSeek for DeepSeek-V4.1-Flash-1017 (MIT) and its Engram tables.
- LibertAI for the EXL3 2.9 bpw quantization (MIT). We did not make or modify these weights.
- TensorFold (ashhart) for the engine, Apache-2.0.
- Bertholomus AI for the DeepSeek-V4.1 engine family inside TensorFold (Apache-2.0).
- Elevatormusic for the pinned build and the two-node launcher scripts this configuration runs on (Apache-2.0).
- The configuration, the measurements and the memory note above are ours.

Scripts in the pinned repository are Apache-2.0; this repository's text and `.env` are Apache-2.0. The model and the
weights keep their own licenses.

Not affiliated with or endorsed by DeepSeek, NVIDIA, LibertAI, TensorFold or Elevatormusic.
