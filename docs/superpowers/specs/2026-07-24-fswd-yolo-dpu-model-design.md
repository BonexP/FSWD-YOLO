# FSWD-YOLO DPU-Oriented Model Design

## Goal

Define a trainable FSWD-YOLO-DPU variant before starting another full training run. The variant must preserve the recognizable FSWD-YOLO structure, especially the parallel spatial and channel attention roles of `C2PSFCA`, while removing graph forms already proven incompatible with `DPUCZDX8G_ISA1_B4096`.

This design intentionally separates the original research model from the deployment-oriented variant:

- `fswd-yolo.yaml` and `C2PSFCA` remain unchanged.
- A new `fswd-yolo-dpu.yaml` and `C2PSFCADPU` represent the trainable deployment variant.
- Inspector screening happens before formal training.
- A successful screening result for `DPUCZDX8G_ISA1_B4096` is not a final board-compatibility claim. Quantization and compilation must later use the selected board's real `arch.json` or fingerprint.

## Evidence

The controlled activation experiments used the same checkpoint, input shape, seed, target, and repository commit. Their manifests reported:

| Graph | Direct blockers | Downstream CPU nodes | Total CPU findings | DPU-assigned nodes |
| --- | ---: | ---: | ---: | ---: |
| Original SiLU audit | 123 | 199 | 322 | 131 |
| Hardswish experiment | 46 | 43 | 89 | 216 |
| Hardsigmoid experiment | 46 | 43 | 89 | 216 |

Both hard activations removed all 77 `aten::silu` blockers. Hardswish produced substantially less drift on the deterministic diagnostic input than Hardsigmoid and is the selected trainable activation. The diagnostic input is not accuracy evidence; retraining remains required.

The 46 remaining direct blockers in the Hardswish graph divide into:

- 14 backbone or neck `nndct_strided_slice` nodes.
- 20 `C2PSFCA` nodes.
- 12 Detect/decode nodes that will be moved to host post-processing.

The `C2PSFCA` blockers map to three top-level channel slices, six PSA operations, and eleven FCA operations:

- PSA: three q/k/v slices, two matrix multiplications, and one softmax.
- FCA: two Conv1d operations, two reductions, three sigmoids, two expands, one matrix multiplication, and one subtraction.

PG338 v4.1, Table 7 on PDF pages 19-21, lists convolution, depthwise convolution, average pooling, Hard Sigmoid, Hard Swish, elementwise sum, elementwise multiplication, and concatenation for DPUCZDX8G. Support is configuration dependent. Although PG338 also describes an optional Softmax accelerator, the selected Inspector target explicitly rejected this graph's softmax nodes. Target-specific Inspector evidence takes precedence over the product-level capability list.

## Design Decisions

### Model Identity

The deployment model retains these FSWD-YOLO characteristics:

- A three-path `C2PSFCA` shell: preserved path, spatial-attention path, and channel-attention path.
- A spatial path with attention residual and feed-forward residual behavior.
- A channel path driven by global statistics and channel recalibration.
- Concatenation followed by pointwise convolution for path fusion.
- The existing C3k2GhostSimAMinner, neck, and three-scale detection organization.

The internal attention mathematics may change where the original operation cannot execute on the target DPU. The deployment variant must have a distinct name in code, checkpoints, reports, and manuscript claims.

### Default Activation

`fswd-yolo-dpu.yaml` will use the existing Ultralytics model-level activation mechanism:

```yaml
activation: torch.nn.Hardswish()
```

This makes Hardswish part of the trainable model and checkpoint. It replaces the Inspector-only forward binding used for graph experiments. Explicit gates inside attention modules use `torch.nn.Hardsigmoid`.

The implementation must not change `Conv.default_act` permanently for unrelated models in the same test process. Tests that construct both original and DPU models must isolate or restore this class-level setting.

## C2PSFCADPU Architecture

### Interface

`C2PSFCADPU(c1, c2, n=1, e=0.5, channel_reduction=4)` preserves the original constructor contract and output shape. It requires `c1 == c2` and computes `c = int(c1 * e)`.

For input `x` with shape `[B, c1, H, W]`, the output shape is `[B, c2, H, W]`.

### Split-Free Projections

The original `Conv(c1, 3*c, 1)` followed by `split` is replaced by three independent projections:

```text
a = keep_proj(x)       # c1 -> c
b = spatial_proj(x)    # c1 -> c
c = channel_proj(x)    # c1 -> c
```

The three projection convolutions have the same total convolution weight count as one `c1 -> 3*c` convolution. Separate normalization modules add only a small parameter overhead. This removes the three top-level strided slices.

### Spatial Branch

The spatial branch preserves the two residual stages of `PSABlock` while replacing global self-attention with a DPU-oriented spatial gate:

```text
spatial_gate = Hardsigmoid(PointwiseConv(DepthwiseConv5x5(b)))
b = b + b * spatial_gate
b = b + FFN(b)
```

The FFN remains two pointwise convolutions with expansion factor 2. Its first activation follows the model default Hardswish; its second convolution has no activation, matching the original residual FFN form.

The branch contains no q/k/v split, transpose, matrix multiplication, or softmax. The 5x5 depthwise kernel gives a wider local receptive field than the original positional 3x3 convolution while staying within the PG338 depthwise-convolution capability. Inspector remains the authority on the target-specific channel and bank constraints.

### Channel Branch

The channel branch retains global pooling and multiplicative channel recalibration:

```text
channel_gate = GlobalAveragePool(c)
channel_gate = Conv1x1Reduce(channel_gate)
channel_gate = Hardswish(channel_gate)
channel_gate = Conv1x1Expand(channel_gate)
channel_gate = Hardsigmoid(channel_gate)
c = c * channel_gate
```

The hidden channel count is `max(1, c // channel_reduction)`, with a default reduction of 4. Both gate convolutions are Conv2d. The branch contains no Conv1d, squeeze/transpose chain, matrix multiplication, reduction sum, explicit expand, subtraction, or ordinary Sigmoid.

The existing Inspector report assigned the final broadcast elementwise multiplication in FCA to the DPU. The new branch must still be verified because support depends on tensor layout and the final target configuration.

### Fusion

The output is:

```text
out = fuse(concat(a, b, c))
```

`fuse` is a `Conv(3*c, c2, 1)`. This preserves the original three-path fusion structure.

## Split-Free Backbone And Neck

The Hardswish graph still contains 14 `nndct_strided_slice` nodes in C2f-derived backbone and neck modules. Formal training must not begin with these known blockers.

Each affected DPU variant will replace:

```text
Conv(c1, 2*c, 1) -> chunk/split -> branch 0, branch 1
```

with two independent `Conv(c1, c, 1)` projections. Convolution weights and every BatchNorm vector can be copied by output-channel slice from the original projection. With the same activation, this transformation is mathematically equivalent and must pass an FP32 allclose test before the model-level Hardswish change is considered.

Only affected classes receive explicitly named DPU variants. The original C2f, C3k2, and C3k2GhostSimAMinner implementations remain unchanged.

## Training And Deployment Heads

Training keeps the existing Detect behavior so the standard Ultralytics detection loss and label assignment remain unchanged.

Deployment uses an adapter that returns the three raw detection feature maps before DFL, anchor/grid decoding, final Sigmoid, coordinate arithmetic, and NMS. For the configured model, each output has channel count `nc + 4 * reg_max` and the P3, P4, and P5 spatial resolutions.

CPU post-processing consumes these three tensors and must reproduce the native FP32 decoded predictions within an agreed numerical tolerance before quantization work starts. The adapter has no trainable state and is not a separate checkpoint.

This boundary deliberately treats decode and NMS as host operations. It avoids forcing the twelve known Detect/decode blockers into the DPU graph while keeping the trainable detection head unchanged.

## Post-Inspection GSConv Amendment

The first random-weight DPU candidate inspection localized all 15 remaining CPU findings to the three `GSConv` instances inside layer 19 `VoVGSCSPC`. Each instance produced the same `reshape -> permute -> reshape` channel-shuffle pattern: the transpose was assigned to CPU and the surrounding layout operations followed it off the DPU. No other candidate-model node appeared in the hardware-constraints table.

The DPU YAML therefore uses `VoVGSCSPCDPU`, composed from `GSConvDPU` and `GSBottleneckCDPU`. `GSConvDPU` preserves the original standard-convolution branch, 5x5 depthwise branch, concatenation, and channel ordering. It replaces only the tensor-layout shuffle with a one-hot pointwise convolution initialized to the identical permutation.

The permutation weight remains an `nn.Conv2d` buffer rather than a trainable parameter. This keeps the operation visible to THOP, NNDCT, and Inspector as a standard convolution while preventing Ultralytics Trainer from enabling gradients and changing the permutation. The buffer is persistent in checkpoints and migration reports. Resource gates count all persistent deployment tensors, not only trainable parameters.

The transformation must pass isolated GSConv and complete `VoVGSCSPC` FP32 parity checks when source and destination use the same activation modules. The actual checkpoint migration is not source-to-candidate equivalent because the DPU YAML intentionally replaces SiLU and C2PSFCA; the migration report records this distinction explicitly. DPU support remains an Inspector question: training cannot start until the revised graph reports no CPU assignment for the fixed pointwise shuffle or any other backbone/neck node.

## Weight Initialization

A dedicated initialization tool will construct the DPU YAML model from the original checkpoint and produce a machine-readable migration report. It will never modify the source checkpoint.

Migration is divided into three classes:

1. Exact name-and-shape copies for unchanged layers.
2. Structured copies for split-free projections:
   - slice original convolution output channels into the independent branch convolutions;
   - slice BatchNorm weight, bias, running mean, and running variance identically;
   - copy the original `C2PSFCA.cv2` into `C2PSFCADPU.fuse`;
   - copy compatible PSA FFN parameters.
3. Exact name-and-shape copies for `VoVGSCSPC -> VoVGSCSPCDPU`, plus deterministic one-hot initialization for its three new fixed shuffle buffers.
4. New initialization for the spatial gate and channel bottleneck parameters whose semantics or shapes differ.

The report records source and destination checkpoint hashes, exact-copy keys, transformed-copy keys, newly initialized keys, skipped keys, tensor counts, parameter counts, and migration percentages. Unexpected same-name shape mismatches are errors, not silent skips.

## Graph-Screening Workflow

Formal training starts only after this sequence:

1. Construct `fswd-yolo-dpu.yaml` without trained weights.
2. Run a CPU forward and training-loss smoke test.
3. Record model parameters and FLOPs.
4. Inspect the model-config graph on the Vitis AI host using `DPUCZDX8G_ISA1_B4096`.
5. Parse the generated report with the existing structured report parser.
6. Iterate on graph structure without launching full training if any backbone, neck, or `C2PSFCADPU` direct blockers remain.
7. Inspect the raw-head deployment adapter separately.
8. Run the migration tool and smoke-test the partially initialized model.
9. Obtain explicit approval of the graph report and training protocol.
10. Launch one formal retraining campaign containing all approved DPU-oriented changes.

The Inspector CLI will accept either a `.pt` checkpoint or a model YAML as the graph source. The manifest distinguishes `checkpoint_model` from `model_config_candidate`, hashes the selected source, and records whether weights are random, migrated, or trained.

## Acceptance Gates

### Module Gates

- Output shape matches the original module contract.
- Forward and backward passes produce finite values.
- No `split`, `chunk`, matmul, softmax, Conv1d, explicit expand, or reduction sum exists in `C2PSFCADPU`.
- Inspector reports zero direct blockers inside `C2PSFCADPU`.
- All internal spatial and channel gate nodes are DPU assigned.

### Full-Graph Gates

- The DPU training YAML contains actual Hardswish modules and no SiLU modules.
- Backbone and neck contain no `nndct_strided_slice` direct blockers.
- The raw-head deployment graph contains no CPU fallback inside the backbone, neck, attention module, or convolutional detection head.
- Any remaining CPU work belongs only to the explicitly external decode/NMS pipeline.
- Inspector console output and parsed manifests are retained, including official device and DPU subgraph counts.
- Evidence runs use a clean Git worktree.

### Resource Gates

- Total persistent deployment tensors, including fixed buffers, do not exceed the original FSWD-YOLO by more than 10%.
- Total FLOPs do not exceed the original FSWD-YOLO at 640x640.
- The report includes trainable parameters, persistent buffers, total deployment tensors, and model-level parameter/FLOP deltas.

### Training Gate

This design does not invent an accuracy-loss threshold before a candidate graph exists. Formal training is authorized only after the graph and resource gates pass and the user approves the experiment protocol. Accuracy evaluation then follows the paper workspace evidence rules: independent test-set reporting and multi-seed evidence are preferred over a single validation run.

## Testing Strategy

Local dependency-light tests cover:

- YAML parsing and class registration.
- Original model construction after DPU model construction, guarding activation state leakage.
- `C2PSFCADPU` output shape, finite forward values, and finite backward gradients.
- Spatial and channel branch tensor shapes.
- Parameter and FLOP budget reporting.
- Exact FP32 parity for isolated split-free projection conversion with the same activation.
- Weight-migration accounting and failure on unexpected mismatches.
- Raw-head output contracts and FP32 decode parity with native Detect.
- Inspector CLI argument validation and candidate-source manifest fields.

Target-specific integration remains on the Vitis AI 3.5 host. No local unit test may claim DPU compatibility in place of Inspector evidence.

## Failure Handling

- A candidate with any new unsupported operator returns to graph design before training.
- A projection conversion that fails exact parity is treated as an implementation defect.
- A raw-head post-processor that fails decoded-output parity blocks deployment work.
- A candidate exceeding either resource gate is rejected or explicitly redesigned; it is not silently accepted because training accuracy might improve.
- A target result from a dirty worktree is diagnostic only and must be repeated cleanly before it becomes experiment evidence.

## Non-Goals

This design does not:

- modify or rename the original FSWD-YOLO model;
- claim that Hardswish preserves the original checkpoint's accuracy;
- start formal training before graph screening;
- quantize or compile without the final platform configuration;
- require Detect decode or NMS to execute on the DPU;
- treat PG338's product-level operator list as proof that a specific graph is supported.
