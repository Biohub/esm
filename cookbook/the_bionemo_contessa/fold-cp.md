# `wrap_model_with_cp` — running ESMFold2 across several GPUs

## TL;DR

`wrap_model_with_cp(model, dm, …)` takes a normal `ESMFold2Model` and rewires it,
in place, to **spread one fold across several GPUs** so you can fold **longer
proteins** than fit on a single card. It does not make a new kind of model — it's
the same object, with a few internal pieces swapped for versions that split their
work across the GPUs. Don't wrap it, and it runs exactly as before on one GPU.

If you just want to use it, skip to [Minimal usage](#minimal-usage). The rest of
this doc explains *why* it's built the way it is.

## Why this exists: the "every pair of residues" table

A protein is a chain of building blocks called **residues** (`L` = how many). To
predict its shape, ESMFold2 keeps a big table with **one cell for every pair of
residues** — cell `(i, j)` is what the model believes about how residue `i` relates
to residue `j`. This is the **pair representation** (or just `z`), and it's what
dominates memory, because `L` residues means `L × L` cells:

- 500 residues → 250,000 cells
- 2,000 residues → 4,000,000 cells (16× bigger)

Each cell holds ~256 numbers, and the model builds and refines several of these
tables. On one GPU the `L × L` table fills memory first — that's why a single card
caps out at a certain length. (The limiter is **per-GPU peak memory**: the most any
one GPU needs at once.)

**Context parallelism (CP)** is the fix: instead of every GPU holding the whole
table, **cut it into blocks, one per GPU** — 4 GPUs hold a quarter each, 16 a
sixteenth. More GPUs → each holds less → longer proteins fit. Cutting a table into
per-GPU blocks is called **sharding** (the opposite, a full copy on every GPU, is
**replicated**), and it's the one idea this whole doc is about.

## How CP splits the work: a grid of GPUs

CP arranges the GPUs into a **square grid** (so the GPU count must be a perfect
square: 1, 4, 9, 16, …), set up by a helper called `DistributedManager`. Each GPU
process is a **rank**, and the total count is the **world size**. Picture the
`L × L` table as a checkerboard: **the GPU at grid position `(r, c)` owns the block
of rows `r` and columns `c`.** PyTorch tracks this with a **`DTensor`** — an ordinary
tensor that also knows it's split across GPUs and which block lives where.

Two things stay small and simple:

- **Model weights are replicated** — every GPU keeps a full copy of the network's
  parameters (small compared to the `L × L` activations).
- Only the big `L × L` **activations get sharded**, so each GPU holds `L² / P` of
  them (`P` = number of GPUs). That's the whole point: per-GPU memory *drops* as you
  add GPUs, instead of every GPU carrying the full table.

## The design pattern: swap pieces in, keep the same model

The base model file (`modeling_esmfold2.py`) has **no idea CP exists** — it imports
nothing from the distributed code. Instead its `forward` has a few **optional
checks**: `getattr(self, "_cp_*", None)` lookups that ask "has a distributed helper
been plugged in here?"

```python
_cp_pair_init = getattr(self, "_cp_pair_init", None)      # sharded pair init (z_init, rel_pos, token_bonds, mask)
_cp_engine    = getattr(self, "_cp_recycle_engine", None) # recycle loop
_cp_lm        = getattr(self, "_cp_language_model", None) # language-model pair builder
_cp_conf      = getattr(self, "_cp_confidence_head", None)# confidence head
_cp_disto     = getattr(self, "_cp_distogram_head", None) # distogram head
```

- **Attribute absent (plain model):** the check returns `None` and the forward runs
  the original code — bit-for-bit the stock model.
- **Attribute present (after wrapping):** the forward hands that step to the
  distributed version.

So `wrap_model_with_cp` parallelizes the fold in two ways:

1. **Swaps a submodule** for a same-shaped version that runs across the grid:
   - the **MSA encoder**
   - every **`FoldingTrunk`** in the model (the recycle trunk, the LM encoder,
     `parcae_coda`, and the confidence head's inner trunk)
   - the **structure head**'s diffusion module (only if `wrap_structure=True`)
   - ESM-C's **MLP** (only if `tp_esmc=True`)
2. **Plugs in a helper** — one of the `_cp_*` attributes the checks above look for:
   - **`_cp_pair_init`** — the sharded pair init (`z_init`, rel_pos, token_bonds, the
     starting pair table, and the pair mask)
   - **`_cp_recycle_engine`** — the recycle loop
   - **`_cp_language_model`** — the language-model pair builder (`lm_z`, the `L×L`
     pair table built from ESM-C's per-residue embeddings)
   - **`_cp_distogram_head`** — the distogram head
   - **`_cp_confidence_head`** — the confidence head

Because it edits the same object, `type(model)` is still `ESMFold2Model`, and
`from_pretrained` / `fold()` / the output dictionary all behave exactly as before.
You can even switch one piece back off at runtime
(`model._cp_confidence_head = None`), which makes it easy to turn a single piece off
and compare.

## The fold, end to end (in plain terms)

Before the stage table, here's the whole pipeline in one pass, naming each part:

1. **`inputs_embedder`** turns the raw atoms into per-residue features.
2. **ESM-C** — a large protein language model (6 billion parameters) — reads the
   sequence and produces rich embeddings.
3. Those feed the first `L × L` **pair table** (the "pair init"), optionally combined
   with an **MSA** (a stack of evolutionarily-related sequences that gives extra
   hints — skipped if you don't have one).
4. The **recycle loop** refines that table over a few passes, each pass running the
   main refining network (the **trunk**, a "Pairformer") and feeding the result back
   in.
5. The refined table becomes a **distogram** (the model's predicted histogram of
   residue-to-residue distances) and then goes to the **structure head**, which uses
   **diffusion** to turn it into actual 3D coordinates.
6. Finally the **confidence head** predicts how trustworthy the result is (the
   pLDDT / pTM / ipTM scores you get back).

## Stage-by-stage: what runs where

The base model runs every stage at full length `L` on every GPU, with the pair table
full-size. The table below walks that same pipeline; **a `*` means
`wrap_model_with_cp` splits that stage across the grid** (everything else stays
replicated).

| Stage (base ESMFold2Model.forward) | Split across GPUs? | What wrapping does |
|---|---|---|
| `inputs_embedder` (atom features) | — (replicated) | unchanged; its cost grows only **linearly** with size (it looks at a small window of atoms at a time), so it stays a small, fixed floor |
| ESM-C 6B language model → embeddings | `*` MLP only, if `tp_esmc=True` | splits ESM-C's big MLP across GPUs; the rest stays replicated; ESM-C is moved to CPU right after it's used, to free room |
| `language_model` → `lm_z` (`L×L`) | `*` | `_cp_language_model` builds this `L×L` table already sharded (the full thing is never assembled on any GPU) |
| pair init (`z_init`, rel_pos, token_bonds, initial `z`, pair mask) | `*` | `_cp_pair_init` builds each `L×L` table one block per GPU — no full copy anywhere. (This *used* to be built full then cut up, which is what ran short proteins out of memory.) |
| recycle loop (`_run_one_loop`) | `*` | `_cp_recycle_engine` keeps the pair table sharded the whole time — it never gathers the full table between passes |
| `parcae_readout` + `parcae_coda` | `*` | runs on each GPU's block plus a shared trunk |
| `distogram_head(z + zᵀ)` | `*` | works on the sharded table and gathers only the small result |
| `structure_head.sample` (diffusion → 3D coords) | `*` (if `wrap_structure`) | runs the diffusion with the pair table kept sharded |
| `confidence_head` (pLDDT/PAE/PDE/pTM/ipTM scores) | `*` | every `L×L` input (pair, rel_pos, token_bonds, distance bins, mask) stays sharded — nothing full-size is rebuilt; only the small final scores are gathered |

**Net effect:** after wrapping, the pair table stays cut-into-blocks from start to
finish (recycle → parcae → distogram → structure → confidence). Only small,
per-residue results and the final outputs are ever reassembled.

## Inside the recycle loop

"Recycling" just means: run the refining network a few times, each pass taking the
previous pass's table as a starting point. The loop is
`for _ in range(total_steps)` with `total_steps = num_loops + 1` (so `num_loops=3`
→ 4 passes). A leading **`*`** marks steps split across the grid.

- **Before the loop (done once, reused by every pass)**
  - `inputs_embedder` → per-residue features  *(replicated — grows only linearly with size, so it's a small fixed floor, not a wall)*
  - **\*** ESM-C 6B → embeddings (then moved to CPU) — **MLP split only if `tp_esmc=True`**
  - **\*** `language_model` → `lm_z` (the `L×L` language-model table)  *(built already sharded)*
  - **\*** `z_init`, rel_pos, token_bonds, the starting pair table, and the pair mask  *(each built one block per GPU by `_cp_pair_init` — never full-size anywhere)*
  - `a`, `b_mat`, MSA column mask  *(tiny, replicated)*
- **The loop itself**
  - **\*** the one thing carried from pass to pass is **`z`** (the pair table being refined)  *(stays sharded the whole time)*
  - each pass does, in order:
    1. **\*** **LM dropout** — randomly drop part of the language-model table, a fresh pattern each pass  *(drawn per-block)*
    2. **\*** **LM encoder** — refine that table
    3. reset the injection base — `z_inject = z_init`  *(just reusing the sharded starting table; no real compute)*
    4. **\*** **MSA encoder** — re-sample the MSA and fold it in (skipped entirely if you have no MSA)
    5. **\*** add the LM output into `z_inject`  *(elementwise, on each GPU's block)*
    6. **\*** **parcae step** — blend the previous table with the new one  *(elementwise, on each block)*
    7. **\*** **trunk (Pairformer)** — the main refining network
  - the three heavy steps per pass: **\*** LM encoder, **\*** MSA encoder, **\*** trunk
- **After the loop (once)**
  - **\*** `parcae_readout` → **\*** `parcae_coda` → **\*** `distogram_head` → **\*** `structure_head.sample` (→ 3D coords, if `wrap_structure`) → **\*** `confidence_head`
- **Redrawn each pass vs fixed:**
  - redrawn: **\*** `z`, **\*** the LM dropout pattern, the MSA re-sample, **\*** the three heavy steps
  - fixed (built once, reused): `lm_z`, `z_init`, `pair_mask` (all sharded), `a`, `b_mat`, MSA data + column mask
- **How the `*` steps actually run across GPUs (`CPRecycleEngine.run_loop`):**
  - keeps `z` sharded across all passes
  - calls the sharded versions of the LM encoder / MSA encoder / trunk (no full-table gather between passes)
  - the blend + dropout run on each GPU's local block

**Stays replicated on every GPU (not split):** `inputs_embedder` (but it grows only
linearly, so it never dominates at large `L`), the bulk of the ESM-C forward (unless
`tp_esmc=True`), and the tiny per-residue/scalar bits (`a`, `b_mat`, the per-residue
masks, MSA sampling prep). Note the `L×L` **pair** mask is *not* replicated — it's
built one block per GPU, like the rest of the pair init.

## Memory: what shrinks, what doesn't

- **Grows with `L²` (the `L×L` pair table) → now sharded:** pair init, recycle loop,
  parcae, distogram, structure, confidence. Each is built one block per GPU and never
  assembled full, so **adding GPUs raises the longest protein you can fold**. (The
  pair init was the last holdout — it used to build these tables full on every GPU
  and only cut them up later, so short proteins could still run out of memory during
  setup. Now they start sharded.)
- **Grows only with `L` (atom-level work):** `inputs_embedder` — a slow-growing,
  replicated floor. Splitting it would shave the baseline but wouldn't change the
  ceiling, so it's left alone.
- **Roughly fixed:** ESM-C's weights (split by `tp_esmc`); ESM-C's own activations are
  the remaining floor (a different technique would be needed to shrink those, not yet
  built).

## One hardware caveat: you need a fast GPU interconnect

Splitting the table trades memory for **communication**: because each GPU holds only
part of it, the GPUs constantly shuffle blocks back and forth — many times per fold —
and they wait while that traffic moves. So CP is only fast with a fast link between
GPUs: **NVLink** (NVIDIA's direct GPU-to-GPU link) within a machine, and
**InfiniBand** (a high-speed network fabric) between machines.

Without them — GPUs on plain PCIe, or nodes on ordinary Ethernet — communication
dominates: you still get the memory savings (longer proteins fit), but each fold can
be much slower. Treat NVLink-within-a-node and InfiniBand-between-nodes as the
intended setup, not an optimization.

## Comparison to the single-GPU implementation

The API and outputs are identical (`from_pretrained`, `fold()`, output keys,
`type(model)`), and almost every step matches the single-GPU result — exactly, or
within tiny bf16 rounding — checked by folding real proteins and comparing. Two
differences are expected, both small:

- **LM dropout** picks a different (still valid) random pattern than the single-GPU
  run, because the sharded table can't cheaply reproduce the exact full-table draw.
  This is the main reason the pTM/ipTM scores can wiggle slightly.
- **Rounding** differs at about 1e-3, because numbers are summed across GPUs in a
  different order.

See `fold-cp_caveats.md` for how to tell these harmless differences from a real bug.

## Requirements the wrapped model adds

- `torch.distributed` set up via `DistributedManager` with a **square** number of GPUs
  (1, 4, 9, 16 …).
- `num_diffusion_samples == 1` (the distributed diffusion path).
- `comm="gather"` (exact) or `comm="ring"` (cheaper at scale) for the MSA encoder;
  `tp_esmc=True` needs ESM-C's `ffn_hidden=6912` to divide evenly by the number of
  GPUs (4/9/16 all work).

## Minimal usage

```python
from transformers.models.esmfold2.distributed import DistributedManager, wrap_model_with_cp
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

# number of GPUs must be a perfect square; launch with torchrun --nproc-per-node=<P>
DistributedManager.initialize(OrderedDict([("dp", 1), ("cp", (n, n))]),
                              device_type="cuda", backend="nccl")
dm = DistributedManager()
model = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16").cuda().eval()

wrap_model_with_cp(model, dm, comm="ring", tp_esmc=True)   # rewires model in place
# still an ESMFold2Model; fold() / outputs unchanged, now spread across the GPUs.
```
