# ort-dequant-proof

Lean 4 proof that the new CPU loops for ONNX Runtime's sub-byte `DequantizeLinear`
(int4, uint4, int2, uint2; per-tensor, per-axis and blocked) produce exactly the same
output as the scalar loops they replace.

Check it (Lean core only, no Mathlib):

```sh
lean DequantProof.lean
```

## Why bit-identical output reduces to integers and indices

Both kernels compute, for every output element `e`,

    y[e] = float(q(e) - zp(σ e)) * float(scale(σ e))

with the same IEEE operations in the same order. So the outputs are identical exactly when:

1. the new byte unpacking gives the same integer `q(e)` as `GetElem`, and
2. every output index is written exactly once, reading the scale and zero point index `σ e`
   that the scalar loop reads.

## What is proved

Bytes are `BitVec 8`, indices are `Nat`.

- `decode_int4`, `decode_int2`: for every byte and every sub-element, the new
  `int8(uint8(v << (8 - bits*(j+1)))) >> (8 - bits)` equals `GetElem`'s sign extension through
  `int32`. Checked over all 256 bytes.
- `unpack_bulk`: the byte loop of `UnpackSubByteElements` reads exactly element `first + i`
  for output slot `i`, and stays inside the buffer.
- One `Exact` theorem per loop path. Each says the path's write map is in range, injective and
  surjective onto `[0, M*K*N)`, and reads scale index `σ e`:
  - `axis_last_exact`: per-axis, quantize axis last (`N == 1`), rows split into chunks.
  - `axis_long_rows_exact`, `block_long_rows_exact`: rows of `N >= 512` split into chunks
    (this includes per-tensor).
  - `axis_short_rows_exact`, `block_short_rows_exact`: rows of `N < 512`, several whole rows
    per work unit; `short_rows_buffer` shows they fit the stack buffer.
  - `rowStep_correct`: the short-row loop's `(m, k, block)` counters always equal
    `(r / K, r % K, r % K / B)`.
  - `block_last_exact`: blocked, quantize axis last, blocks grouped per unit;
    `block_loop_bounds` shows the C++ loop bounds visit exactly this domain, and
    `block_last_buffer` shows the group fits the stack buffer.

Because each work unit writes a disjoint set of output indices, the result does not depend on
how the thread pool schedules units.

## Modelling assumptions

- `σ` is the scalar loop's scale index written in closed form:
  per-axis `σ e = e / N % K`, blocked `σ e = (e / N / K * nblk + e / N % K / B) * N + e % N`.
  The scalar loops reach these by incrementing pointers once per element and once per block.
- `size_t` indices do not overflow (all are below the tensor size).
- The float expression itself is not modelled: it is the same source expression on both sides.
  It is an exact small-int to float conversion and one multiply (no add, so no FMA
  contraction), which rounds the same way scalar or vectorized unless built with fast-math.
- The Lean model of each loop path is transcribed by hand from the C++. The tests in the PR
  (zero-tolerance comparison on every path, plus a 3000-shape randomized diff on GCC, Clang
  and MSVC) are what tie this model to the compiled code.

Axioms: `propext`, `Classical.choice`, `Quot.sound` (Lean's standard ones); no `sorry`.
