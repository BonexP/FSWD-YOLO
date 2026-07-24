# Vitis Inspector Diagnostics and Activation Experiments Design

## Goal

Turn a successful Vitis AI Inspector invocation into an accurate compatibility assessment, then compare explicit DPU-oriented activation experiments without presenting a modified graph as the original FSWD-YOLO model.

## Evidence From the Prepared Run

The run for `DPUCZDX8G_ISA1_B4096` completed and preserved the raw prediction tensor exactly. However, `status: ok` only means that Inspector completed. It does not mean that the complete model was assigned to the DPU.

After deduplicating repeated report rows by node name, operator, and reason, the graph has 322 unique CPU findings. Its direct blockers include:

- 77 `aten::silu` nodes that cannot be converted to XIR
- 24 `nndct_strided_slice` nodes that cannot be assigned to the DPU
- `C2PSFCA` attention operators including `nndct_conv1d`, `nndct_expand_as`, `nndct_matmul`, `nndct_sum`, `nndct_softmax`, and shape-constrained `nndct_sigmoid`

The previous preparation changed `aten::silu_` into `aten::silu`, so counting only `aten::silu_` incorrectly suggested success. Binding `C2f.forward_split` changed `chunk` into `split` without reducing the 24 unique `nndct_strided_slice` nodes.

## Documentation Findings

The Vitis AI 3.5 PyTorch operator table does not list SiLU. It lists Hardswish and Hardsigmoid, states that ordinary tensor slices become `aten::slice`, and states that a slice is compiled for the CPU unless it can be removed or fused. It also states that Softmax is CPU-only in this PyTorch path and that Matmul reaches the DPU only when it can be transformed into Conv2d.

The DPUCZDX8G product guide lists ReLU, ReLU6, LeakyReLU, Hard Sigmoid, and Hard Swish as supported activation capabilities. It also makes support dependent on DPU configuration and identifies the generated `arch.json` as the compiler description of the final hardware.

The third-party YOLOv11 experiment follows the same pattern: it replaces SiLU, replaces Chunk with a convolutional operation, and removes or approximates unsupported attention and post-processing operations. Its Hardswish graph still produced multiple DPU subgraphs, so it ultimately used Hardsigmoid and retrained. This is useful experimental evidence, not a guarantee for FSWD-YOLO or the selected target.

The supplied Vitis AI 1.4.1 custom-layer page describes TensorFlow2 `custom_objects` and custom quantization strategies. It does not establish PyTorch DPU execution. Vitis AI 3.5 Custom OP support can implement operations as CPU OPs for GraphRunner, but that is a heterogeneous deployment route rather than an all-DPU conversion.

## Design Principles

The tool will separate two questions:

1. **Original-model audit:** What can the unmodified checkpoint execute on for the selected target?
2. **Deployment experiment:** What graph partition results from an explicitly selected, non-equivalent activation replacement?

The default command must answer the first question. It must never apply an accuracy-changing activation replacement implicitly. An experimental result must identify itself as a deployment experiment in the manifest and console output.

## Scope

This iteration will:

- parse and deduplicate the generated Inspector text report
- classify direct blockers separately from downstream CPU effects
- record a compatibility summary in `inspection_manifest.json`
- stop binding `C2f.forward_split`, because the target report proved it ineffective
- retain the semantics-preserving change from in-place to non-in-place SiLU for tracing
- add `--activation-experiment {none,hardswish,hardsigmoid}` with `none` as the default
- measure prediction drift for every experiment
- run Inspector for an explicitly requested non-equivalent experiment even though strict parity is not expected

This iteration will not:

- alter training-time model definitions or checkpoint files
- claim that Hardswish or Hardsigmoid preserves model accuracy
- select a deployment activation before comparing target-specific reports
- rewrite C2f channel splitting
- rewrite `C2PSFCA`
- install or update dependencies
- quantize or compile the model
- claim final board compatibility without the board's target fingerprint or `arch.json`

## Original-Model Audit

With `--activation-experiment none`, the CLI will disable only in-place behavior on existing `torch.nn.SiLU` modules. This preserves SiLU mathematics while allowing NNDCT to trace the graph. It will not bind `C2f.forward_split`.

The CLI will compare deterministic raw predictions before and after preparation. Equal shapes and `torch.allclose` with `rtol=1e-5` and `atol=1e-6` remain a hard gate before Inspector. The manifest will record:

- `mode: original_model_audit`
- `semantic_equivalence_required: true`
- preparation counters
- shape, allclose, maximum error, and mean error

## Activation Experiments

With `--activation-experiment hardswish` or `hardsigmoid`, the CLI will first apply the semantics-preserving tracing preparation, then replace each unique SiLU module's forward operation in memory with the selected hard activation. It will not write a checkpoint.

The CLI will evaluate the experimental graph on the same deterministic input and record the same drift metrics. A shape mismatch remains a hard failure. A numerical mismatch is expected and will not stop Inspector because the user explicitly selected a non-equivalent experiment.

The manifest will record:

- `mode: deployment_activation_experiment`
- `activation_experiment`
- `semantic_equivalence_required: false`
- `checkpoint_modified: false`
- unique module replacement count
- prediction drift metrics

The console must print a clear warning that the resulting Inspector report says nothing about retained model accuracy and that retraining or fine-tuning is required before deployment use.

The two activations must be tested in separate output directories. The experiment is useful only for graph selection:

- direct activation blocker counts
- new blocker types introduced by the activation
- downstream CPU-node reduction
- DPU partition continuity visible in the Inspector report

## Report Analysis

After `Inspector.inspect()` returns, the CLI will locate the newly generated `inspect_*.txt` report. A focused parser will recognize rows containing a node name, operator type, and hardware-constraint reason.

Rows will be deduplicated by all three fields because Vitis AI 3.5 repeats the findings table. Reasons will be classified as:

- `direct_unsupported`: contains `can't be converted to XIR`, `can't be assigned to DPU`, `Try to assign`, or `Convert nndct graph to XIR failed`
- `downstream_cpu`: indicates that inputs, children, concat operands, reshape inputs, or inserted transpose operations are already on CPU
- `other_cpu_constraint`: any parsed CPU constraint not covered above

The manifest will add `inspection_summary` containing:

- report path
- parsed and unique row counts
- repeated-row count
- unique node count
- category counts
- operator counts per category
- representative node names and reasons for each direct blocker operator
- `requires_cpu_fallback`, true whenever unique CPU findings exist

The top-level `status` remains an execution status. `status: ok` must not be interpreted as a full-DPU verdict.

If no new report is generated, multiple new reports are ambiguous, or a non-empty report cannot be parsed, the CLI will fail and write the error to the manifest. An empty hardware-constraint table is recorded as zero findings.

## Decision Gates

The next Vitis host runs will use three new output directories for `none`, `hardswish`, and `hardsigmoid`. The report summaries will determine the activation candidate, but neither hard activation becomes a deployment solution until a separately trained model recovers acceptable validation and test accuracy.

After activation selection, later work will proceed as separate experiments:

1. Replace split-producing projections with equivalent dual convolution branches and require FP32 parity.
2. Design a DPU-oriented `C2PSFCA` alternative using supported Conv2d, pooling, elementwise, and hard-activation operations.
3. Create an explicitly named FSWD-YOLO-DPU training configuration.
4. Initialize reusable weights, retrain or fine-tune, and compare accuracy with the original model.
5. Quantize and compile with the final board's actual fingerprint or `arch.json`.

The original checkpoint and the DPU-oriented trained variant must remain distinct artifacts in reports and manuscript claims.

## Verification

Dependency-free unit tests will cover report parsing, repeated-row deduplication, reason classification, empty and malformed reports, activation-mode argument validation, original-mode parity enforcement, experiment-mode drift recording, and preparation counters. Existing deployment-tool tests and static checks will run locally. The Vitis AI 3.5 host remains the required integration environment for target-specific graph results.
