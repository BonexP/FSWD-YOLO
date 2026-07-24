# Vitis Inspector Diagnostics and SiLU Experiment Design

## Goal

Turn the successful Vitis AI Inspector run into an accurate compatibility assessment and test one semantics-preserving SiLU graph form without mixing it with other model rewrites.

## Evidence From the Prepared Run

The run for `DPUCZDX8G_ISA1_B4096` completed successfully and preserved the raw prediction tensor exactly. However, `status: ok` only means that Inspector completed. It does not mean that the complete model was assigned to the DPU.

The generated report contains three textual copies of each finding. After deduplication by node name, operator, and reason, the graph has 322 CPU findings and the following direct blockers:

- 77 `aten::silu` nodes that cannot be converted to XIR
- 24 `nndct_strided_slice` nodes that cannot be assigned to the DPU
- attention-path blockers in `C2PSFCA`, including `nndct_conv1d`, `nndct_expand_as`, `nndct_matmul`, `nndct_sum`, `nndct_softmax`, and shape-constrained `nndct_sigmoid`

The previous preparation changed `aten::silu_` into `aten::silu`, so counting only `aten::silu_` incorrectly suggested success. The previous `C2f.forward_split` binding also changed `chunk` into `split` without reducing the 24 unique `nndct_strided_slice` nodes. Both observations come from the target-specific Inspector output rather than an assumed operator list.

## Scope

This iteration will:

- parse the generated Inspector text report after every successful run
- deduplicate repeated report rows
- distinguish direct unsupported operators from downstream CPU-placement effects
- record the compatibility assessment in `inspection_manifest.json`
- replace Inspector-only SiLU execution with the mathematical form `x * sigmoid(x)`
- preserve the deterministic prediction-equivalence gate
- remove the ineffective Inspector-only `C2f.forward_split` binding

This iteration will not:

- alter training-time model definitions or checkpoint contents
- install or update Vitis AI packages
- claim hardware execution or performance without a board
- rewrite C2f channel splitting
- rewrite `C2PSFCA` attention
- compile or quantize the model

## Report Analysis

The existing Inspector CLI will locate the generated `inspect_*.txt` file after `Inspector.inspect()` returns. A focused parser will recognize hardware-constraint rows with these fields:

1. node name
2. operator type
3. reason

Rows will be deduplicated by all three fields because the Vitis AI 3.5 report repeats the same table in multiple sections. Reasons will be classified as:

- `direct_unsupported`: contains `can't be converted to XIR`, `can't be assigned to DPU`, `Try to assign`, or `Convert nndct graph to XIR failed`
- `downstream_cpu`: indicates that inputs, children, or inserted transpose operations are already on CPU
- `other_cpu_constraint`: any parsed CPU constraint that does not match the known direct or downstream forms

The manifest will add an `inspection_summary` object containing:

- report path
- total parsed rows and total unique rows
- unique node count
- repeated-row count
- category counts
- operator counts within each category
- representative node names and reasons for each direct blocker operator
- `requires_cpu_fallback`, which is true when the report contains any unique CPU finding

The top-level `status` remains an execution status. `status: ok` means that tracing and report generation completed. It must not be used as a DPU-compatibility verdict.

If no report is generated, more than one newly generated report is ambiguous, or a non-empty report cannot be parsed, the CLI will fail and write the reason to the manifest. A genuinely empty constraint table is recorded as zero findings rather than silently treated as a parser error.

## SiLU Experiment

The Inspector model will retain the loaded weights and module topology, but each `torch.nn.SiLU` instance will receive an Inspector-only forward method equivalent to:

```python
return x * torch.sigmoid(x)
```

This tests whether NNDCT can lower the primitive multiplication and sigmoid graph even though it cannot convert `aten::silu`. The transformation does not assert that the target supports the expanded form; only the next target-specific Inspector report can establish that.

The preparation manifest will replace the misleading `silu_inplace_disabled` and `c2f_forward_split_enabled` interpretation with explicit counters:

- `silu_decomposed`
- `c2f_forward_split_enabled: 0`

The original and prepared models will run on the same seeded FP32 tensor. Inspection proceeds only when output shapes match and `torch.allclose` succeeds with the existing `rtol=1e-5` and `atol=1e-6`. Maximum and mean absolute error remain recorded. A mismatch is a hard failure before NNDCT inspection.

## Single-Variable Comparison

The next Vitis AI run must use a new output directory. Its report will be compared with the current prepared baseline using unique-node summaries, not raw string counts.

The SiLU hypothesis is supported only if:

- prediction parity passes
- unique `aten::silu` direct blockers decrease from 77, ideally to zero
- no new direct blocker replaces SiLU at comparable scale

The hypothesis is rejected if `aten::silu` remains, the expanded sigmoid/multiply nodes become direct blockers, or prediction parity fails. In that case the repository will retain the diagnostic parser while the SiLU experiment will not be presented as a deployment solution.

## Later Iterations

After the SiLU result is isolated, channel slicing can be tested independently by replacing each split-producing projection with equivalent output-channel convolution branches. That transformation needs its own design because it changes module topology and can affect quantization even when FP32 predictions match.

`C2PSFCA` is a separate architectural compatibility problem. Its attention operators should be assessed only after common activation and channel-split blockers are isolated, because changing it could require retraining or a DPU-oriented approximation.

## Verification

Dependency-free unit tests will cover report deduplication, reason classification, repeated rows, empty reports, malformed non-empty reports, manifest summary shape, SiLU method binding, and preparation counters. Existing prediction-parity tests will remain. Full local tests and static checks will run before commit, while the Vitis AI 3.5 host remains the required integration environment for the target-specific result.
