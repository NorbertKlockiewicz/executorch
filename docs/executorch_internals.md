# ExecuTorch Internals — Deep-Dive Reference

> A technical reference for working with ExecuTorch's export pipeline, backend delegation,
> quantization, and operator/kernel systems. Assumes familiarity with PyTorch and FX graphs.

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [IR Levels — The Three Dialects](#2-ir-levels--the-three-dialects)
3. [Export Pipeline (Python Side)](#3-export-pipeline-python-side)
   - [Stage 0: torch.export → ATen Dialect](#stage-0-torchexport--aten-dialect)
   - [Stage 1: to_edge() → Edge Dialect](#stage-1-to_edge--edge-dialect)
   - [Stage 1b: to_edge_transform_and_lower() — Recommended Path](#stage-1b-to_edge_transform_and_lower--recommended-path)
   - [Stage 2: to_executorch() → ExecuTorch Dialect](#stage-2-to_executorch--executorch-dialect)
   - [Stage 3: Serialization to .pte](#stage-3-serialization-to-pte)
4. [Backend Delegation](#4-backend-delegation)
   - [Python Backend Interface](#41-python-backend-interface)
   - [Partitioning](#42-partitioning)
   - [to_backend() Dispatch](#43-to_backend-dispatch)
   - [C++ Runtime Interface](#44-c-runtime-interface)
   - [Concrete Backend Examples](#45-concrete-backend-examples)
5. [Quantization](#5-quantization)
   - [PT2E Quantization Flow](#51-pt2e-quantization-flow)
   - [Quantizer Interface](#52-quantizer-interface)
   - [Annotation System](#53-annotation-system)
   - [Backend Quantizers](#54-backend-quantizers)
   - [Quantized Kernels](#55-quantized-kernels)
6. [Operator Registration & Kernels](#6-operator-registration--kernels)
   - [YAML Operator Definitions](#61-yaml-operator-definitions)
   - [C++ Kernel Pattern](#62-c-kernel-pattern)
   - [Runtime Kernel Registry](#63-runtime-kernel-registry)
   - [Adding a New Custom Operator](#64-adding-a-new-custom-operator)
7. [Pass Infrastructure](#7-pass-infrastructure)
8. [Memory Planning](#8-memory-planning)

---

## 1. Project Architecture Overview

```
executorch/
├── exir/                   # Python-side compiler infrastructure (IR, passes, serialization)
│   ├── program/            # Main pipeline entry points (to_edge, to_executorch, managers)
│   ├── backend/            # Backend delegation API (Partitioner, BackendDetails, to_backend)
│   ├── passes/             # Built-in compiler passes (memory planning, spec prop, sym shape)
│   ├── dialects/           # Edge and backend op definitions (edge.yaml, _ops.py)
│   ├── emit/               # Graph → Program schema emitter
│   ├── _serialize/         # FlatBuffers serialization helpers
│   ├── serde/              # Graph module serialization/deserialization
│   ├── capture/            # CaptureConfig, EdgeCompileConfig, ExecutorchBackendConfig
│   ├── verification/       # IR dialect verifiers (ATen, Edge)
│   └── lowered_backend_module.py  # LoweredBackendModule (post-delegation wrapper)
│
├── schema/                 # FlatBuffers schema for .pte file format (program.fbs)
│
├── runtime/                # C++ on-device runtime
│   ├── backend/            # BackendInterface, register_backend, get_backend_class
│   ├── executor/           # Graph executor, instruction dispatch
│   ├── kernel/             # Kernel registry, KernelRuntimeContext, operator_registry
│   └── core/               # EValue, Tensor, Error, memory allocators, NamedDataMap
│
├── kernels/                # Kernel implementations (Python-registered, C++ bodies)
│   ├── portable/           # Pure C++ fallback kernels for all Core ATen ops
│   │   ├── cpu/            # op_*.cpp implementations
│   │   ├── functions.yaml  # ATen-compatible op → kernel mapping
│   │   └── custom_ops.yaml # Non-ATen custom ops
│   ├── optimized/          # Platform-optimized kernel variants
│   └── quantized/          # Quantized decomposed ops (quantized_decomposed::*)
│       ├── cpu/            # op_quantize.cpp, op_dequantize.cpp, …
│       └── quantized.yaml
│
├── backends/               # Hardware backend implementations
│   ├── xnnpack/            # XNNPACK (CPU, mobile)
│   ├── vulkan/             # Vulkan (Android GPU)
│   ├── apple/              # CoreML + MPS (Apple)
│   ├── qualcomm/           # QNN (Qualcomm NPU)
│   ├── arm/                # TOSA / Ethos-U
│   └── transforms/         # Shared graph transforms used across backends
│
├── extension/              # Higher-level utilities (LLM runner, training, mobile SDKs)
│   └── llm/runner/         # C++ multimodal/LLM runner infrastructure
│
├── codegen/                # Code generation tooling (generates NativeFunctions.h from YAML)
└── examples/               # Example export scripts and model wrappers
```

**Python vs. C++ boundary:** Everything in `exir/` runs at export time (Python). Everything in
`runtime/` runs on-device (C++). The `kernels/` C++ bodies are compiled into a library that
the runtime links against. The `backends/` directory is split — Python `preprocess()` runs at
export time, while the C++ `BackendInterface` subclass runs at inference time.

---

## 2. IR Levels — The Three Dialects

ExecuTorch uses a three-stage intermediate representation. Each stage is still an FX
`GraphModule` / `ExportedProgram`, but the allowed operator set and graph properties
change at each level.

### ATen Dialect

- **What it is:** The direct output of `torch.export.export()`.
- **Operator set:** Full PyTorch ATen operator set (2000+ ops via `torch._ops`).
  Includes non-core ops, in-place ops, view ops.
- **Symbolic shapes:** Present — `torch.SymInt` in tensor sizes.
- **Used by:** Backends can inspect but not lower from this dialect directly.
- **Verifier:** `EXIRATenDialectVerifier` in
  [exir/verification/verifier.py](../exir/verification/verifier.py)

### Edge Dialect

- **What it is:** Output of `to_edge()`. A restricted, compilable IR.
- **Operator set:** Core ATen ops only — a curated subset of ~200 ops that have
  well-defined semantics and decompositions. Operators are represented as
  `EdgeOpOverload` instances defined in
  [exir/dialects/edge/_ops.py](../exir/dialects/edge/_ops.py) with schemas in
  [exir/dialects/edge/edge.yaml](../exir/dialects/edge/edge.yaml).
- **Key constraints:**
  - No non-Core ATen ops (e.g. `torch.ops.aten._unsafe_view` → replaced by `view_copy`).
  - No `.item()` calls (no tensor-to-scalar extraction inside the graph).
  - View ops replaced by `view_copy` (explicit copy semantics).
  - Memory layout operators added (`MemoryFormatOpsPass`).
- **Symbolic shapes:** Still present until `ConstraintBasedSymShapeEvalPass`.
- **Verifier:** `EXIREdgeDialectVerifier` in
  [exir/verification/verifier.py](../exir/verification/verifier.py)

### ExecuTorch Dialect (Final)

- **What it is:** Output of `to_executorch()`. Ready for serialization and on-device execution.
- **Operator set:** Backend-specific lowered ops. Delegated subgraphs are replaced by
  `executorch_call_delegate()` nodes. Remaining un-delegated ops use `out`-variant ops
  (e.g. `aten.add.out`) that write into pre-allocated output tensors.
- **Key properties:**
  - All symbolic shapes resolved to concrete integers or bounds.
  - Memory layout fully planned (tensors assigned to memory arenas).
  - `view_copy` ops optionally converted back to lightweight `view` ops.
  - `out` parameters pre-allocated via `memory.alloc` nodes.
- **Verifier:** Runtime execution itself (structural validation at emit time).

### Dialect Transitions

```
torch.export.export()
        │
        ▼
  ATen Dialect
  (ExportedProgram)
        │
        │  to_edge()  ─── or ─── to_edge_transform_and_lower()
        ▼
  Edge Dialect
  (EdgeProgramManager)
        │
        │  .to_executorch()
        ▼
  ExecuTorch Dialect
  (ExecutorchProgramManager)
        │
        │  .buffer  (serialize)
        ▼
    .pte file
```

---

## 3. Export Pipeline (Python Side)

All pipeline functions live in
[exir/program/_program.py](../exir/program/_program.py).
Configuration dataclasses are in
[exir/capture/_config.py](../exir/capture/_config.py).

---

### Stage 0: torch.export → ATen Dialect

```python
import torch
from torch.export import export, Dim

# Capture with dynamic shapes
seq_len = Dim("seq_len", min=1, max=2048)
ep = export(
    model,
    args=(tokens, input_pos),
    dynamic_shapes={"tokens": {1: seq_len}, "input_pos": {0: seq_len}},
)
# ep is a torch.export.ExportedProgram in ATen dialect
```

**`CaptureConfig`** ([exir/capture/_config.py:23](../exir/capture/_config.py#L23)):

| Field | Default | Purpose |
|---|---|---|
| `enable_dynamic_shape` | `False` | Legacy flag, no effect when `enable_aot=True` |
| `enable_aot` | `False` | Enables automatic dynamic shapes via Dynamo |
| `enable_functionalization` | `True` | Converts in-place ops to functional equivalents |
| `_dynamo_config` | `ExirDynamoConfig()` | Dynamo tracing options |

> **Note:** In modern usage you call `torch.export.export()` directly rather than the
> legacy `exir.capture()`. The `CaptureConfig` is kept for backward compatibility.

---

### Stage 1: to_edge() → Edge Dialect

**Entry point** ([exir/program/_program.py:1413](../exir/program/_program.py#L1413)):

```python
from executorch.exir import to_edge, EdgeCompileConfig

edge_manager = to_edge(
    {"forward": ep},               # dict of method_name → ExportedProgram
    compile_config=EdgeCompileConfig(
        _check_ir_validity=True,   # run verifier after conversion
        preserve_ops=[torch.ops.aten.linear.default],  # skip decomposition for these
    ),
)
# Returns EdgeProgramManager
```

**Pass sequence inside `_generate_edge_program()`**
([exir/program/_program.py:857](../exir/program/_program.py#L857)):

1. `remove_unused_parameters_pass` — drops dead `placeholder` nodes
2. `RemoveNonCoreAtenOpGraphAssertsPass` — removes `assert` nodes using non-Core ops
3. `ReplaceViewOpsWithViewCopyOpsPass` — `view` → `view_copy` (explicit copy semantics)
4. Pre-op replace passes (custom, from `EdgeCompileConfig`)
5. `OpReplacePass` — maps ATen ops to their Edge dialect equivalents
6. `MemoryFormatOpsPass` — inserts memory-layout-specific ops
7. Post-op replace passes (custom)
8. `run_decompositions()` — decomposes ops not in Core ATen
9. `lift_constant_tensor_pass` — lifts inline constant tensors to `get_attr` nodes
10. `EXIREdgeDialectVerifier` — validates the result (if `_check_ir_validity=True`)

**`EdgeCompileConfig`** ([exir/capture/_config.py:37](../exir/capture/_config.py#L37)):

| Field | Default | Purpose |
|---|---|---|
| `_check_ir_validity` | `True` | Run `EXIREdgeDialectVerifier` after conversion |
| `_use_edge_ops` | `True` | Use Edge dialect ops (should stay True) |
| `_skip_dim_order` | `False` | Skip dim-order annotations (legacy) |
| `preserve_ops` | `[]` | List of ops to **not** decompose — used when a backend needs them intact |
| `_core_aten_ops_exception_list` | `[]` | Ops missing decompositions; skip their check only |

**`EdgeProgramManager`** exposes:

```python
edge_manager.exported_program("forward")  # get individual ExportedProgram
edge_manager.transform([MyPass()])         # apply custom passes to all methods
edge_manager.to_backend(XnnpackPartitioner())  # delegate subgraphs
```

---

### Stage 1b: to_edge_transform_and_lower() — Recommended Path

**Entry point** ([exir/program/_program.py:1291](../exir/program/_program.py#L1291)):

```python
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

edge_manager = to_edge_transform_and_lower(
    {"forward": ep},
    partitioner=[XnnpackPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
```

**Why it exists:**
Some backends (like XNNPACK) need to see ops like `torch.ops.aten.linear.default` *before*
decomposition into smaller ATen primitives. `to_edge()` would decompose everything eagerly.
`to_edge_transform_and_lower()` instead:

1. Asks each partitioner `ops_to_not_decompose()` for a set of ops to preserve.
2. Calls `_gen_edge_manager_for_partitioners()` which runs `to_edge()` with those ops
   added to `preserve_ops`.
3. Applies any user `.transform()` passes.
4. Iterates through partitioners calling `.to_backend(partitioner)`.
5. Verifies that all preserved ops ended up in the backend (not left un-lowered in the graph).

This is the correct API for most production use cases.

---

### Stage 2: to_executorch() → ExecuTorch Dialect

**Entry point** (on `EdgeProgramManager`):

```python
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

exec_manager = edge_manager.to_executorch(
    ExecutorchBackendConfig(
        sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        memory_planning_pass=MemoryPlanningPass(),
        emit_stacktrace=False,
        extract_delegate_segments=True,
        external_constants=False,
    )
)
# Returns ExecutorchProgramManager
```

**Pass sequence inside `edge_to_executorch_passes()`**
([exir/program/_program.py:837](../exir/program/_program.py#L837)):

1. Custom passes from `ExecutorchBackendConfig.passes`
2. `SpecPropPass` — propagates `TensorSpec` (dtype, layout, memory format) through the graph
3. `EdgeToBackendOpsPass` — converts Edge dialect ops to their ExecuTorch `out`-variant forms
4. `RemoveGraphAssertsPass` — drops remaining assert nodes
5. **Pre-memory planning:**
   - `NormalizeViewCopyBasePass` — normalizes base tensors for view ops
   - `dead_code_elimination_pass` — removes unused nodes
   - `ReplaceViewCopyWithViewPass` — replaces view_copy with lightweight view (if `remove_view_copy=True`)
   - `ConstraintBasedSymShapeEvalPass` — evaluates symbolic shapes to concrete integers using `Dim()` constraints
   - `ToOutVarPass` — rewrites ops from functional to `out`-param form
6. `memory_planning_pass` — assigns each tensor to a memory arena and offset

**`ExecutorchBackendConfig`** ([exir/capture/_config.py:57](../exir/capture/_config.py#L57)):

| Field | Default | Purpose |
|---|---|---|
| `passes` | `[]` | Custom passes to run before built-in ones |
| `memory_planning_pass` | `MemoryPlanningPass()` | Strategy for tensor memory layout |
| `sym_shape_eval_pass` | `ConstraintBasedSymShapeEvalPass()` | How to resolve symbolic shapes |
| `emit_stacktrace` | `False` | Embed Python stack traces in debug info |
| `extract_delegate_segments` | `True` | Store backend blobs in separate segments (frees after init) |
| `segment_alignment` | `128` | Byte alignment for segments |
| `external_constants` | `False` | Store weight tensors in a separate `.ptd` file |
| `external_mutable_weights` | `False` | Store trainable weights externally |
| `do_quant_fusion_and_const_prop` | `False` | Run quantization fusion + constant folding |
| `remove_view_copy` | `True` | Replace view_copy with lighter view ops |

**`ConstraintBasedSymShapeEvalPass`:**
This is the key pass for dynamic shapes. Instead of requiring all shapes to be static,
you express constraints via `torch.export.Dim()`:

```python
seq_len = Dim("seq_len", min=1, max=2048)
```

The pass evaluates symbolic expressions using these bounds, replacing `SymInt` nodes
with their upper-bound values for memory planning purposes, while preserving runtime
dynamic dispatch.

> **Critical note (from LFM2 experience):** Using `.item()` inside model forward (e.g.
> in `rope.get_freqs`) causes `ConstraintBasedSymShapeEvalPass` to fail with
> `Cannot cast FakeTensor to number`. Fix: avoid `.item()` in the graph entirely and
> use tensor indexing instead.

---

### Stage 3: Serialization to .pte

**Triggering serialization:**

```python
with open("model.pte", "wb") as f:
    f.write(exec_manager.buffer)
# Or for multi-method PTE:
exec_manager.write_to_file("model.pte")
```

**Emit phase** ([exir/emit/_emit_program.py:118](../exir/emit/_emit_program.py#L118)):

`emit_program()` walks each method's FX graph and constructs a Python `Program` object
matching the FlatBuffers schema. Each FX node becomes an `Instruction`:

- `call_function` → an `Operator` instruction with resolved kernel index
- `executorch_call_delegate()` → a `DelegateCall` instruction with backend blob reference
- `memory.alloc` → an `Alloc` instruction with arena + offset

**Serialization phase** ([exir/_serialize/](../exir/_serialize/)):

1. `FlatTensorSerializer` packs tensor data into contiguous segments.
2. `serialize_for_executorch()` converts the Python `Program` to FlatBuffers binary.
3. Result: raw bytes written to `.pte` file.

**`.pte` file schema** ([schema/program.fbs](../schema/program.fbs)) key tables:

| Table | Purpose |
|---|---|
| `Program` | Root: version, list of `ExecutionPlan`s, constant buffers, segments |
| `ExecutionPlan` | One per named method: instructions, inputs/outputs, chain of delegates |
| `Instruction` | Single operation: `KernelCall`, `DelegateCall`, `Alloc`, `JumpFalseCall`, `MoveCall` |
| `Tensor` | Metadata: dtype, shape, strides, dim order, allocation info |
| `AllocationDetails` | Arena index + byte offset in that arena |
| `BackendDelegate` | Backend ID string + reference to compiled blob |
| `BackendDelegateInlineData` | Backend blob embedded in the flatbuffer |
| `BackendDelegateDataReference` | Reference to external segment (when `extract_delegate_segments=True`) |

**Multi-method PTE:**
Pass a `dict` of `{method_name: ExportedProgram}` to `to_edge()`. Each method becomes
a separate `ExecutionPlan` in the `Program`. Constant methods (like `get_max_seq_len`)
are passed via the `constant_methods` argument:

```python
edge_manager = to_edge(
    {"vision_encoder": ep_vis, "token_embedding": ep_emb, "text_decoder": ep_dec},
    constant_methods={"get_max_seq_len": 2048, "get_eos_ids": [7]},
)
```

Constant methods are serialized as small `ExecutionPlan`s that return a literal value.

---

## 4. Backend Delegation

Backend delegation is the mechanism by which a subgraph of the FX graph is compiled
by a hardware-specific backend (XNNPACK, CoreML, QNN, etc.) and stored as an opaque
blob in the `.pte` file.

---

### 4.1 Python Backend Interface

**File:** [exir/backend/backend_details.py](../exir/backend/backend_details.py)

```python
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export import ExportedProgram
from typing import List

class MyBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        # Inspect edge_program.graph_module
        # Compile to backend binary
        blob: bytes = compile_to_binary(edge_program)
        return PreprocessResult(
            processed_bytes=blob,
            debug_handle_map=None,   # optional: {node_id: debug_handle} for profiling
            data_store_output=None,  # optional: shared named data across partitions
        )
```

**Design constraints:**
- `BackendDetails` subclasses cannot themselves be subclassed (enforced by
  `__init_subclass__` at [backend_details.py:60](../exir/backend/backend_details.py#L60)).
  Each backend must be a final, concrete implementation.
- `preprocess()` receives the `ExportedProgram` for the **subgraph** being delegated,
  not the full model graph.
- The returned `processed_bytes` are an opaque blob — ExecuTorch never interprets them.
  They are stored in the `.pte` file and passed verbatim to the C++ `init()` at runtime.

**`PreprocessResult`** ([backend_details.py:24](../exir/backend/backend_details.py#L24)):

| Field | Purpose |
|---|---|
| `processed_bytes: bytes` | Compiled backend blob (stored in .pte, passed to C++ `init()`) |
| `debug_handle_map` | Maps delegate node IDs to debug handles (for profiling) |
| `data_store_output` | Shared named tensors/data for cross-partition sharing |
| `_delegate_info_meta` | Backend-specific metadata stored in `LoweredBackendModule.meta` |

**`CompileSpec`** ([exir/backend/compile_spec_schema.py](../exir/backend/compile_spec_schema.py)):

```python
@dataclass
class CompileSpec:
    key: str     # e.g. "storage_type_override", "force_fp16"
    value: bytes # serialized config value
```

Backends parse these in `preprocess()` to enable features. The same specs are stored in
the `.pte` and passed to the C++ `init()` so the runtime can apply matching options.

**`preprocess_multimethod()`** ([backend_details.py:104](../exir/backend/backend_details.py#L104)):

An optional override that receives **all** partitions across **all** methods at once.
Useful when a backend needs to share weights or data between, e.g., a vision encoder
partition and a text decoder partition. The default implementation just calls
`preprocess()` for each partition independently.

**`LoweredBackendModule`** ([exir/lowered_backend_module.py](../exir/lowered_backend_module.py)):

After `preprocess()`, ExecuTorch wraps the result in a `LoweredBackendModule`:

```python
class LoweredBackendModule(torch.nn.Module):
    _backend_id: str             # e.g. "XnnpackBackend"
    _processed_bytes: bytes      # blob from preprocess()
    _compile_specs: List[CompileSpec]
    _original_exported_program: ExportedProgram  # original subgraph, for inspection
    meta: Optional[Dict[str, Any]]  # includes debug_handle_map, _delegate_info_meta
```

This module is attached to the parent graph module as an `nn.Module` attribute and
referenced via `get_attr` + `executorch_call_delegate()` call in the FX graph.

---

### 4.2 Partitioning

Rather than delegating an entire model to a single backend, you typically use a
**Partitioner** to select which subgraph(s) to delegate.

**File:** [exir/backend/partitioner.py](../exir/backend/partitioner.py)

```python
from executorch.exir.backend.partitioner import Partitioner, PartitionResult, DelegationSpec
from executorch.exir.backend.compile_spec_schema import CompileSpec

class MyPartitioner(Partitioner):
    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partition_tags = {}
        for node in exported_program.graph.nodes:
            if self._can_delegate(node):
                tag = f"my_tag_{node.name}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = DelegationSpec(
                    backend_id="MyBackend",
                    compile_specs=[CompileSpec("key", b"value")],
                )
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )

    def ops_to_not_decompose(self, ep):
        # Return ops that should stay un-decomposed for this backend to handle
        return ([torch.ops.aten.linear.default], None)
```

**Partitioning rules:**
- A node is tagged by setting `node.meta["delegation_tag"] = "some_tag"`.
- All nodes sharing a tag form one delegated submodule.
- Tags must map to a `DelegationSpec` in `partition_tags`.
- The partitioner **must not** modify the graph structure — only add metadata.
  ExecuTorch validates this with `is_identical_graph()`.
- Constant nodes shared across partitions are automatically duplicated.
- Output nodes (`output` in FX) can never be tagged.

**`DelegationSpec`** ([partitioner.py:19](../exir/backend/partitioner.py#L19)):

```python
class DelegationSpec(NamedTuple):
    backend_id: str              # must match the name used in register_backend() on C++ side
    compile_specs: List[CompileSpec]
```

**`ops_to_not_decompose()`** ([partitioner.py:97](../exir/backend/partitioner.py#L97)):

Returns a list of `torch._ops.OpOverload` that should **not** be decomposed by
`to_edge_transform_and_lower()`. This is how XNNPACK keeps `aten.linear.default`
intact instead of having it decomposed into `mm + add`.

An optional filter function `Callable[[Node], bool]` can be returned as a second value
to further refine which specific nodes of that op should remain un-decomposed.

---

### 4.3 to_backend() Dispatch

**File:** [exir/backend/backend_api.py](../exir/backend/backend_api.py)

Two call signatures:

**Variant 1 — Direct (whole subgraph)**:
```python
from executorch.exir.backend.backend_api import to_backend

lowered: LoweredBackendModule = to_backend(
    "MyBackend",          # backend_id string
    edge_program,         # the ExportedProgram to compile
    [CompileSpec(...)],   # compile specs
)
```
Finds the `BackendDetails` subclass registered under `"MyBackend"`, calls its
`preprocess()`, and returns a `LoweredBackendModule`.

**Variant 2 — Partitioner-based**:
```python
edge_manager = edge_manager.to_backend(MyPartitioner())
```
Internally:
1. Runs `partitioner.partition(exported_program)` to get tagged graph.
2. Validates partitioner didn't modify graph structure.
3. Groups nodes by tag, extracts each group as a submodule.
4. Calls Variant 1 `to_backend()` for each submodule.
5. Replaces each submodule `call_function` with `executorch_call_delegate(lowered_module, *args)`.

The resulting graph has `executorch_call_delegate` nodes where partitioned subgraphs
were, with `LoweredBackendModule` instances carrying the compiled blobs.

---

### 4.4 C++ Runtime Interface

**File:** [runtime/backend/interface.h](../runtime/backend/interface.h)

```cpp
namespace executorch {

class BackendInterface {
 public:
  // Returns true if this backend can run on the current device.
  virtual bool is_available() const = 0;

  // Called once when the .pte program is loaded.
  // processed: the blob from Python preprocess(), stored in .pte
  // compile_specs: same specs used at export time
  // Returns an opaque DelegateHandle* owned by the backend.
  virtual Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const = 0;

  // Called for every inference pass.
  // handle: the DelegateHandle* from init()
  // args: EValue* array containing input + output tensors
  virtual Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const = 0;

  // Optional: update runtime backend options.
  virtual Error set_option(BackendOptionContext&, Span<BackendOption>&);
  virtual Error get_option(BackendOptionContext&, Span<BackendOption>&);

  // Called when the program is destroyed. Release resources.
  virtual void destroy(DelegateHandle* handle) const {}
};

// Registration (call once, typically in a static initializer):
Error register_backend(const Backend& backend);  // Backend = {name, BackendInterface*}
BackendInterface* get_backend_class(const char* name);
}
```

**Lifecycle:**

```
Program load
    │
    ▼
backend->is_available()       ← check device support
    │
    ▼
backend->init(blob, specs)    ← parse blob, allocate GPU memory, load kernels
    │                           returns DelegateHandle*
    ▼
  [inference loop]
    │
    ▼
backend->execute(handle, args) ← run the delegated subgraph
    │
    ▼
  [program destroyed]
    │
    ▼
backend->destroy(handle)       ← free GPU memory, release handles
```

**`FreeableBuffer`:** The `processed` parameter in `init()` wraps the blob from
`preprocess()`. If the backend doesn't need the raw bytes after `init()` (e.g. after
loading into GPU memory), it should call `processed->Free()` to allow ExecuTorch to
reclaim that memory.

**`EValue`:** A tagged union (`Tensor | int | float | bool | string | ...`) used
throughout the runtime for passing values between kernels and delegates.

---

### 4.5 Concrete Backend Examples

**XNNPACK** ([backends/xnnpack/xnnpack_preprocess.py](../backends/xnnpack/xnnpack_preprocess.py)):
- Walks each FX node with a visitor pattern (`node_visitors` dict mapping op → handler).
- Builds an `XNNGraph` (a custom flatbuffer structure describing the XNNPACK op graph).
- For quantized models, applies `ConvertToLinearPass` beforehand.
- `processed_bytes` = serialized `XNNGraph` flatbuffer.

**Vulkan** ([backends/vulkan/vulkan_preprocess.py](../backends/vulkan/vulkan_preprocess.py)):
- Runs an extensive in-preprocess pass pipeline:
  `FuseBatchNormPass → FusePatternsPass → FuseClampPass → AddmmToLinearTransform →
  RemoveRedundantOpsTransform → SpecPropPass → ConstraintBasedSymShapeEvalPass →
  MemoryPlanningPass`
- Uses `VkGraphBuilder` to produce a Vulkan compute graph.
- Supports `CompileSpec` options: `storage_type_override`, `memory_layout_override`,
  `force_fp16`.

**MPS / CoreML** ([backends/apple/mps/mps_preprocess.py](../backends/apple/mps/mps_preprocess.py)):
- Processes placeholder (input) and `call_function` nodes separately.
- Serializes to a flatbuffer with a custom 8-byte header: `"MP00"` magic + offsets.
- Constant data padded to 16-byte alignment.
- Optional `CompileSpec` `"use_fp16"` for Metal fp16 execution.

---

## 5. Quantization

ExecuTorch uses PyTorch's **PT2E** (Post-Training Quantization 2.0 Export) flow, built
on top of `torchao.quantization.pt2e`. This approach quantizes at the FX graph level,
producing a QDQ (Quantize-Dequantize) decomposed graph that backends can then
efficiently lower.

---

### 5.1 PT2E Quantization Flow

```python
from torchao.quantization.pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

# 1. Create and configure quantizer
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

# 2. Export to FX graph (ATen dialect)
model_export = torch.export.export(model, sample_inputs).module()

# 3. Prepare: insert fake-quantize / observer nodes
prepared = prepare_pt2e(model_export, quantizer)

# 4. Calibrate: run representative data to collect statistics
for batch in calibration_data:
    prepared(*batch)

# 5. Convert: replace observers with actual quantize/dequantize ops
quantized_model = convert_pt2e(prepared)

# 6. Now export the quantized model through the normal pipeline
ep = torch.export.export(quantized_model, sample_inputs)
edge_manager = to_edge_transform_and_lower(
    {"forward": ep},
    partitioner=[XnnpackPartitioner()],
)
```

**Static vs. dynamic quantization:**
- **Static:** calibrate with data → fixed scale/zero_point per tensor.
  Use `is_dynamic=False` in `QuantizationSpec`.
- **Dynamic:** scale/zero_point computed per-activation at runtime.
  Use `is_dynamic=True`.

---

### 5.2 Quantizer Interface

A `Quantizer` is a Python class that annotates an FX graph with quantization metadata.
It does **not** insert any ops itself — that's done by `prepare_pt2e()` using the annotations.

**`QuantizationConfig`** (from `torchao.quantization.pt2e`):

```python
@dataclass
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec]
```

**`QuantizationSpec`:**

```python
@dataclass
class QuantizationSpec:
    dtype: torch.dtype          # e.g. torch.int8, torch.uint8
    quant_min: int              # e.g. -128
    quant_max: int              # e.g. 127
    qscheme: torch.qscheme      # per_tensor_affine, per_channel_affine, etc.
    is_dynamic: bool = False
    ch_axis: Optional[int] = None  # channel axis for per-channel
    observer_or_fake_quant_ctr: Callable  # e.g. HistogramObserver, MinMaxObserver
```

**Helper function** ([backends/xnnpack/quantizer/xnnpack_quantizer.py:106](../backends/xnnpack/quantizer/xnnpack_quantizer.py#L106)):

```python
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config
)

# Produces a QuantizationConfig suitable for int8 symmetric XNNPACK inference
config = get_symmetric_quantization_config(
    is_per_channel=True,   # per-channel weight quantization
    is_dynamic=False,      # static (calibrated) activations
    is_qat=False,          # PTQ (not QAT)
)
```

---

### 5.3 Annotation System

Each backend's `Quantizer` subclass annotates specific subgraphs (patterns) with
`QuantizationAnnotation` objects placed in `node.meta[Q_ANNOTATION_KEY]`.

`XNNPACKQuantizer` uses a registry pattern in
[backends/xnnpack/quantizer/xnnpack_quantizer_utils.py](../backends/xnnpack/quantizer/xnnpack_quantizer_utils.py):

```python
OP_TO_ANNOTATOR: Dict[str, Callable] = {}

def register_annotator(op: str):
    def decorator(fn):
        OP_TO_ANNOTATOR[op] = fn
        return fn
    return decorator

@register_annotator("linear")
def annotate_linear(gm, quantization_config, filter_fn=None):
    # Find all linear subgraphs, annotate input/weight/bias/output nodes
    ...
```

**Supported fusion patterns** (from `xnnpack_quantizer.py`):

| Pattern | Fused nodes |
|---|---|
| `conv_bn_relu` | Conv2d + BatchNorm + ReLU |
| `conv_bn` | Conv2d + BatchNorm |
| `conv_relu` | Conv2d + ReLU |
| `conv` | Conv2d |
| `linear_relu` | Linear + ReLU |
| `linear` | Linear |
| `add_relu` | add + ReLU |
| `add` | add |
| `mul_relu` | mul + ReLU |
| `mul` | mul |
| `cat` | torch.cat |
| `adaptive_avg_pool2d` | AdaptiveAvgPool2d |

The annotation approach ensures that when `prepare_pt2e()` inserts observers, they
are placed at the optimal points (e.g., one observer shared between the output of
`conv` and the input of the fused `relu`).

---

### 5.4 Backend Quantizers

**`XNNPACKQuantizer`**
([backends/xnnpack/quantizer/xnnpack_quantizer.py](../backends/xnnpack/quantizer/xnnpack_quantizer.py)):
- Symmetric int8 activations, int8 per-channel weights.
- `quantizer.set_global(config)` — applies to all supported ops.
- `quantizer.set_module_type(Linear, config)` — per-module-type config.
- `quantizer.set_module_name("model.fc1", config)` — per-module-name config.
- `quantizer.set_operator_type(torch.ops.aten.linear.default, config)` — per-op config.
- `quantizer.set_filter_function(fn)` — custom node filter.

**`QnnQuantizer`** ([backends/qualcomm/quantizer/quantizer.py](../backends/qualcomm/quantizer/quantizer.py)):
- `QuantDtype` enum: `w8a8` (int8 weights + int8 activations), `w8a16`, `w4a8`, `w4a16`, etc.
- Supports both PTQ and QAT modes via `is_qat` flag.
- Per-block quantization for LLM weight compression.

**`ArmQuantizer` / `TOSAQuantizer` / `EthosUQuantizer`**
([backends/arm/quantizer/arm_quantizer.py](../backends/arm/quantizer/arm_quantizer.py)):
- Multiple backend targets with different precision requirements.
- `get_symmetric_quantization_config(is_per_channel, is_qat, is_dynamic)` helper.

**`CoreMLQuantizer`**
([backends/apple/coreml/quantizer/](../backends/apple/coreml/quantizer/)):
- Annotates at module and operation level.
- Targets Apple Neural Engine constraints.

---

### 5.5 Quantized Kernels

Quantized operations live in the `quantized_decomposed` namespace, defined in
[kernels/quantized/quantized.yaml](../kernels/quantized/quantized.yaml)
and implemented in [kernels/quantized/cpu/](../kernels/quantized/cpu/).

Key operations:

| Op | Description |
|---|---|
| `quantized_decomposed::quantize_per_tensor` | Float → int8/uint8 with scalar scale/zp |
| `quantized_decomposed::dequantize_per_tensor` | int8/uint8 → float |
| `quantized_decomposed::quantize_per_channel` | Float → int8 with per-channel scale/zp |
| `quantized_decomposed::dequantize_per_channel` | int8 per-channel → float |
| `quantized_decomposed::choose_qparams.Tensor` | Compute scale/zp from tensor (dynamic quant) |
| `quantized_decomposed::embedding_byte` | int8-quantized embedding lookup |
| `quantized_decomposed::embedding_4bit` | 4-bit quantized embedding lookup |
| `quantized_decomposed::embedding_2bit` | 2-bit quantized embedding lookup |
| `quantized_decomposed::mixed_mm` | int8 weight × float activation matmul |
| `quantized_decomposed::mixed_linear` | int8 weight × float activation linear |

These ops are recognized by the XNNPACK backend and mapped to optimized XNNPACK
kernels during `preprocess()`. If a fallback is needed, the portable C++ implementations
in [kernels/quantized/cpu/](../kernels/quantized/cpu/) handle them.

---

## 6. Operator Registration & Kernels

ExecuTorch uses a static, YAML-driven operator registration system. Unlike PyTorch's
dynamic dispatcher, kernels are registered at compile time and looked up at runtime
using a compact key based on tensor dtype and memory layout.

---

### 6.1 YAML Operator Definitions

**ATen-compatible ops** — [kernels/portable/functions.yaml](../kernels/portable/functions.yaml):

Defines the mapping from ATen operator schema to C++ kernel function. Each entry:

```yaml
- op: add.out
  kernels:
    - arg_meta: null          # null = fallback kernel (handles any dtype/layout)
      kernel_name: torch::executor::add_out
```

With dtype/layout specialization:
```yaml
- op: add.out
  type_alias:
    T0: [Float]              # T0 is an alias for float32
  dim_order_alias:
    D0: [[0, 1, 2, 3]]      # D0 is NCHW contiguous
  kernels:
    - arg_meta:
        self: [T0, D0]
        other: [T0, D0]
        out: [T0, D0]
      kernel_name: torch::executor::add_out_float_nchw  # specialized kernel
    - arg_meta: null
      kernel_name: torch::executor::add_out             # fallback
```

**Non-ATen custom ops** — [kernels/portable/custom_ops.yaml](../kernels/portable/custom_ops.yaml):

Custom ops not in PyTorch's `native_functions.yaml` need a **full function signature**:

```yaml
- func: "my_namespace::my_op.out(Tensor self, float scale, *, Tensor(a!) out) -> Tensor(a!)"
  kernels:
    - arg_meta: null
      kernel_name: torch::executor::my_op_out
```

Requirements for custom op signatures:
- Must have an `out` keyword-only parameter (after `*`).
- Return type must be the same tensor as `out`.
- Supported argument types: `Tensor`, `Scalar`, `int`, `float`, `bool`,
  `Tensor[]`, `int[]`, `float[]`, `bool[]`, `str`, `ScalarType`, `MemoryFormat`,
  `Layout`, `Device`.

**Quantized ops** — [kernels/quantized/quantized.yaml](../kernels/quantized/quantized.yaml):

All in the `quantized_decomposed::` namespace. These always use tensor-based
scale/zero_point parameters (never scalars), ensuring they can be lowered by backends.

---

### 6.2 C++ Kernel Pattern

**File:** [kernels/portable/cpu/op_add.cpp](../kernels/portable/cpu/op_add.cpp)

All portable kernels follow a strict pattern:

```cpp
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

// Kernel must match signature from functions.yaml exactly.
// 'out' parameter is the pre-allocated output tensor.
Tensor& add_out(
    KernelRuntimeContext& ctx,  // context for error reporting, logging
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {              // out-param form: caller allocates, kernel fills

  // Validate inputs using ET_KERNEL_CHECK
  // (logs error and returns out on failure — no exceptions)
  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(a, b, out),
      InvalidArgument,
      out);

  // Resize output if needed (broadcast, etc.)
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  // Type dispatch using macros
  ET_SWITCH_REALB_TYPES(compute_type, ctx, "add.out", CTYPE, [&]() {
    // CTYPE is the C++ type for the selected dtype (float, double, int32, etc.)
    apply_binary_elementwise_fn<CTYPE, CTYPE, CTYPE>(
        [val_alpha](const CTYPE val_a, const CTYPE val_b) {
          return val_a + val_alpha * val_b;
        },
        a, b, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
```

**Key macros:**

| Macro | Purpose |
|---|---|
| `ET_KERNEL_CHECK(ctx, cond, error, out)` | Assert condition; on failure, log + return `out` |
| `ET_KERNEL_CHECK_MSG(ctx, cond, error, out, ...)` | Same but with printf-style message |
| `ET_SWITCH_REALB_TYPES(dtype, ctx, name, CTYPE, fn)` | Dispatch to real/bool dtypes |
| `ET_SWITCH_REALH_TYPES(...)` | Dispatch including float16 |
| `ET_SWITCH_COMPLEXH_TYPES(...)` | Dispatch for complex dtypes |
| `ET_SWITCH_ALL_TYPES(...)` | Dispatch for all supported dtypes |

**Rules for portable kernels (strict):**
1. No dynamic memory allocation (`malloc`, `new`, `std::vector`).
2. No C++ exceptions.
3. No C++ stdlib I/O (`std::cout`, file access).
4. No global mutable state.
5. All memory is provided via parameters.
6. Thread-safe (no shared mutable state).
7. Must handle edge cases gracefully via `ET_KERNEL_CHECK` (never crash).

---

### 6.3 Runtime Kernel Registry

**File:** [runtime/kernel/operator_registry.h](../runtime/kernel/operator_registry.h)

```cpp
// A kernel is uniquely identified by its name + KernelKey
struct Kernel {
  const char* name;       // e.g. "aten::add.out"
  KernelKey kernel_key;   // dtype/dim_order spec string, or nullptr for fallback
  OpFunction op_;         // void (*)(KernelRuntimeContext&, Span<EValue*>)
};

// KernelKey encodes which dtypes/dim_orders this kernel handles:
// Format: "v1/<dtype>;<dim_order>,...|<dtype>;<dim_order>,..."
//   - Each "|"-separated group is one tensor argument
//   - Each ","-separated entry is (dtype_int;dim_order_ints)
//   - nullptr = fallback (matches anything)
// Example: "v1/6;0,1,2,3|6;0,1,2,3|6;0,1,2,3"
//   = float32 NCHW input + float32 NCHW input + float32 NCHW output

// Registration (typically via generated static initializers):
Error register_kernels(const Span<const Kernel> kernels);

// Dispatch:
// Collects TensorMeta (dtype + dim_order) for actual input tensors,
// then finds the best matching kernel.
Result<OpFunction> get_op_function_from_registry(
    const char* name,
    Span<const TensorMeta> meta_list);
```

**Limits** (configurable via CMake `MAX_KERNEL_NUM`):
- Default: `kMaxOperators = 250`, `kMaxKernelsPerOp = 8`
- Max total registered kernels: 2000
- Overflow: `Error::RegistrationExceedingMaxKernels` (logged + panic)

**Kernel selection algorithm:**
1. Find all kernels whose `name` matches.
2. Among those, find kernels whose `KernelKey` matches the actual tensor metadata.
3. Return the most specific match; fall back to the `nullptr`-key kernel if no match.

---

### 6.4 Adding a New Custom Operator

Here are the steps to add a completely new op that doesn't exist in ATen:

**Step 1: Declare in YAML**

Add to [kernels/portable/custom_ops.yaml](../kernels/portable/custom_ops.yaml):

```yaml
- func: "my_ns::my_op.out(Tensor self, float scale, *, Tensor(a!) out) -> Tensor(a!)"
  kernels:
    - arg_meta: null
      kernel_name: torch::executor::my_op_out
```

**Step 2: Implement the C++ kernel**

Create `kernels/portable/cpu/op_my_op.cpp`:

```cpp
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& my_op_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    double scale,
    Tensor& out) {

  ET_KERNEL_CHECK(ctx, self.sizes() == out.sizes(), InvalidArgument, out);

  ET_SWITCH_REALB_TYPES(self.scalar_type(), ctx, "my_ns::my_op.out", CTYPE, [&]() {
    // kernel logic here
    const CTYPE* in_data = self.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    for (size_t i = 0; i < self.numel(); ++i) {
      out_data[i] = static_cast<CTYPE>(in_data[i] * scale);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
```

**Step 3: Add a unit test**

Create `kernels/portable/test/op_my_op_test.cpp` following the pattern of existing tests.

**Step 4: Register in CMakeLists**

In `kernels/portable/CMakeLists.txt`, add your file to the sources list.

**Step 5: Register for Python use (export time)**

To use the op in a PyTorch model being exported:

```python
import torch

# Define the op schema for Python
torch.library.define(
    "my_ns::my_op",
    "(Tensor self, float scale) -> Tensor",
)

@torch.library.impl("my_ns::my_op", "cpu")
def my_op_impl(self, scale):
    return self * scale

@torch.library.impl_abstract("my_ns::my_op")
def my_op_abstract(self, scale):
    return torch.empty_like(self)
```

**Step 6: Build**

```bash
cmake --build cmake-out --target portable_kernels
```

This runs codegen that regenerates `NativeFunctions.h` from the YAML files.

---

## 7. Pass Infrastructure

ExecuTorch's compiler uses FX graph passes to transform `ExportedProgram`s.
All passes follow `torch.fx.passes` conventions.

**File:** [exir/pass_manager.py](../exir/pass_manager.py)

```python
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from executorch.exir.pass_manager import PassType, PassManager

# PassType alias:
# PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]

class MyPass(PassBase):
    def call(self, graph_module: torch.fx.GraphModule) -> Optional[PassResult]:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.relu.default:
                # Replace relu with hardtanh
                with graph_module.graph.inserting_after(node):
                    new_node = graph_module.graph.call_function(
                        torch.ops.aten.hardtanh.default,
                        args=(node.args[0],),
                        kwargs={"min_val": 0.0, "max_val": 6.0},
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                modified = True
        graph_module.graph.lint()
        graph_module.recompile()
        return PassResult(graph_module, modified)
```

**Running passes:**

```python
# On EdgeProgramManager (applies to all methods):
edge_manager = edge_manager.transform([MyPass(), AnotherPass()])

# Via PassManager:
pm = PassManager(passes=[MyPass(), AnotherPass()])
result = pm(graph_module)
```

**Key built-in passes and their roles:**

| Pass | Stage | Purpose |
|---|---|---|
| `OpReplacePass` | ATen→Edge | Replace ATen ops with Edge dialect equivalents |
| `MemoryFormatOpsPass` | ATen→Edge | Insert memory layout annotation ops |
| `ReplaceViewOpsWithViewCopyOpsPass` | ATen→Edge | `view` → `view_copy` |
| `SpecPropPass` | Edge→ET | Propagate `TensorSpec` (dtype, layout) through graph |
| `EdgeToBackendOpsPass` | Edge→ET | Convert Edge ops to `out`-variant ExecuTorch ops |
| `ConstraintBasedSymShapeEvalPass` | Edge→ET | Evaluate symbolic shapes using `Dim()` constraints |
| `ToOutVarPass` | Edge→ET | Rewrite functional ops to use pre-allocated `out` params |
| `NormalizeViewCopyBasePass` | Pre-memory | Normalize view ops for memory planning |
| `ReplaceViewCopyWithViewPass` | Pre-memory | Lightweight view ops (if `remove_view_copy=True`) |
| `dead_code_elimination_pass` | Various | Remove unused nodes |
| `MemoryPlanningPass` | ET | Assign tensors to memory arenas |

**Node metadata conventions:**
- `node.meta["val"]` — FakeTensor carrying shape/dtype info (populated by `torch.export`)
- `node.meta["spec"]` — `TensorSpec` (populated by `SpecPropPass`)
- `node.meta["delegation_tag"]` — string tag for partitioning (set by `Partitioner`)
- `node.meta["debug_handle"]` — integer for debug/profiling mapping

**Writing passes:** Subclass `PassBase`, override `call()`. Use `graph_module.graph.lint()`
to validate the graph after modifications. Use `graph_module.recompile()` to regenerate
the Python function. Always return a `PassResult(graph_module, modified_bool)`.

---

## 8. Memory Planning

Memory planning is the process of assigning each intermediate tensor in the graph to a
specific location in a pre-allocated memory arena. This happens in `to_executorch()` and
is critical for efficient on-device execution (no dynamic allocation at inference time).

**File:** [exir/passes/memory_planning_pass.py](../exir/passes/memory_planning_pass.py)

### How It Works

After `ToOutVarPass` rewrites ops to `out`-param form, every intermediate tensor has
an explicit `out` node. `MemoryPlanningPass` analyzes **tensor liveness** across the
graph and assigns non-overlapping offsets within arenas.

```
Graph: a → relu → b → add → c
                       ↑
                       d

Liveness:
  a: [relu_input ... relu_done]
  b: [relu_done ... add_done]
  d: [add_input ... add_done]
  c: [add_done ... output]

Memory layout (greedy):
  Arena 0:
    offset 0 → b (reuse for c since b is dead when c is alive)
    offset 0 → c
    offset X → d
```

Each tensor gets an `AllocationDetails` entry in the schema:
```
AllocationDetails {
  memory_id: 0,          // arena index
  memory_offset_high: 0, // high 32 bits of byte offset
  memory_offset_low: 512 // low 32 bits of byte offset
}
```

### Configuring Memory Planning

In `ExecutorchBackendConfig`:

```python
from executorch.exir.passes import MemoryPlanningPass

ExecutorchBackendConfig(
    memory_planning_pass=MemoryPlanningPass(
        alloc_graph_input=False,   # don't plan memory for graph inputs
        alloc_graph_output=False,  # don't plan memory for graph outputs
    )
)
```

You can also provide a per-method memory planning pass:
```python
ExecutorchBackendConfig(
    memory_planning_pass={
        "prefill": MemoryPlanningPass(alloc_graph_input=False),
        "decode": MemoryPlanningPass(alloc_graph_input=False),
    }
)
```

### Multiple Memory Arenas

ExecuTorch supports multiple named arenas. At runtime, you provide each arena's buffer:

```cpp
// Python side: tensors can be assigned to arena 0 (default) or custom arenas
// via node.meta["spec"].mem_id

// C++ runtime side:
MemoryAllocator arena0(arena0_size, arena0_buffer);
HierarchicalAllocator hierarchical({arena0, arena1, ...});
```

### Dynamic Memory Planning Mode

Controlled by `ExecutorchBackendConfig.dynamic_memory_planning_mode`:

- `DynamicMemoryPlanningMode.UPPER_BOUND` (default): Use the maximum possible size
  for dynamic-shape tensors (based on `Dim()` bounds). Safe but may over-allocate.
- Other modes: custom strategies for tighter memory bounds.

### Mutable Buffers

Buffers that are mutated in-place (like KV caches in LLMs or SSM conv states) are
tracked separately and allocated as **mutable buffers** in the `.pte` file. They are
not reused by the memory planner across op calls.

Set `emit_mutable_buffer_names=True` in `ExecutorchBackendConfig` to serialize buffer
fully qualified names, enabling named buffer access at runtime via `NamedDataMap`.

---

## Key File Quick Reference

| Topic | File |
|---|---|
| Pipeline entry points | [exir/program/_program.py](../exir/program/_program.py) |
| Config dataclasses | [exir/capture/_config.py](../exir/capture/_config.py) |
| Backend interface (Python) | [exir/backend/backend_details.py](../exir/backend/backend_details.py) |
| Partitioner interface | [exir/backend/partitioner.py](../exir/backend/partitioner.py) |
| to_backend() dispatch | [exir/backend/backend_api.py](../exir/backend/backend_api.py) |
| LoweredBackendModule | [exir/lowered_backend_module.py](../exir/lowered_backend_module.py) |
| Backend interface (C++) | [runtime/backend/interface.h](../runtime/backend/interface.h) |
| Kernel registry (C++) | [runtime/kernel/operator_registry.h](../runtime/kernel/operator_registry.h) |
| Pass manager | [exir/pass_manager.py](../exir/pass_manager.py) |
| Memory planning pass | [exir/passes/memory_planning_pass.py](../exir/passes/memory_planning_pass.py) |
| Spec propagation pass | [exir/passes/spec_prop_pass.py](../exir/passes/spec_prop_pass.py) |
| ATen op definitions | [kernels/portable/functions.yaml](../kernels/portable/functions.yaml) |
| Custom op definitions | [kernels/portable/custom_ops.yaml](../kernels/portable/custom_ops.yaml) |
| Quantized op definitions | [kernels/quantized/quantized.yaml](../kernels/quantized/quantized.yaml) |
| Example kernel impl | [kernels/portable/cpu/op_add.cpp](../kernels/portable/cpu/op_add.cpp) |
| XNNPACK quantizer | [backends/xnnpack/quantizer/xnnpack_quantizer.py](../backends/xnnpack/quantizer/xnnpack_quantizer.py) |
| XNNPACK partitioner | [backends/xnnpack/partition/xnnpack_partitioner.py](../backends/xnnpack/partition/xnnpack_partitioner.py) |
| FlatBuffers schema | [schema/program.fbs](../schema/program.fbs) |
| Edge dialect op schemas | [exir/dialects/edge/edge.yaml](../exir/dialects/edge/edge.yaml) |
| IR verifiers | [exir/verification/verifier.py](../exir/verification/verifier.py) |
