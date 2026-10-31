/-!
# Sub-byte DequantizeLinear in ONNX Runtime: the new CPU loops are exact

The scalar kernel computes, for every output index `e`,
  `y[e] = float(q(e) - zp(σ e)) * scale(σ e)`
where `q(e)` is the `e`-th packed element and `σ` maps an element to its scale / zero point.
The new kernel evaluates the same IEEE expression per element. So it produces bit-identical output
as long as

1. unpacking a byte yields the same integers as `GetElem` (`decode_int4`, `decode_int2`,
   `unpack_bulk`), and
2. every loop path writes each output index exactly once, with the scale index `σ e` that the
   scalar loop uses (the `Exact` theorems below).

Machine bytes are `BitVec`s. Indices are `Nat`s; the kernel uses `size_t` and all
indices are below the tensor size, so they do not overflow.
-/

/-! ## 1. Decoding packed bytes -/

/-- `Int4x2Base<true>::GetElem` / `Int2x4Base<true>::GetElem`:
`((int32)((v >> bits*j) & mask) << (32 - bits)) >> (32 - bits)`. -/
def getElemSigned (bits j : Nat) (v : BitVec 8) : Int :=
  let x : BitVec 8 := (v >>> (bits * j)) &&& BitVec.ofNat 8 (2 ^ bits - 1)
  ((x.zeroExtend 32 <<< (32 - bits)).sshiftRight (32 - bits)).toInt

/-- New kernel: `int8(uint8(v << (8 - bits*(j+1)))) >> (8 - bits)`. -/
def unpackSigned (bits j : Nat) (v : BitVec 8) : Int :=
  ((v <<< (8 - bits * (j + 1))).sshiftRight (8 - bits)).toInt

/-- Unsigned elements are decoded with the same expression on both sides. -/
def getElemUnsigned (bits j : Nat) (v : BitVec 8) : Nat :=
  ((v >>> (bits * j)) &&& BitVec.ofNat 8 (2 ^ bits - 1)).toNat

theorem decode_int4 : ∀ v : BitVec 8, ∀ j : Fin 2, unpackSigned 4 j v = getElemSigned 4 j v := by
  decide +kernel

theorem decode_int2 : ∀ v : BitVec 8, ∀ j : Fin 4, unpackSigned 2 j v = getElemSigned 2 j v := by
  decide +kernel

/-! ## 2. Arithmetic helpers -/

theorem divmod {C : Nat} (a i : Nat) (hC : 0 < C) (hi : i < C) :
    (a * C + i) / C = a ∧ (a * C + i) % C = i := by
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hC, Nat.div_eq_of_lt hi, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]

theorem recon (e C : Nat) : e / C * C + e % C = e := by
  have := Nat.div_add_mod e C
  rw [Nat.mul_comm] at this
  exact this

theorem lin2 {m M x K : Nat} (hm : m < M) (hx : x < K) : m * K + x < M * K := by
  have : (m + 1) * K ≤ M * K := Nat.mul_le_mul_right _ hm
  rw [Nat.succ_mul] at this
  omega

theorem ceil_gt {x C L : Nat} (hC : 0 < C) (h : x * C < L) : x < (L + C - 1) / C := by
  rw [Nat.lt_div_iff_mul_lt hC]
  omega

/-- Splitting `e < M * K` into `(e / K, e % K)`. -/
theorem split_lt {e M K : Nat} (h : e < M * K) : e / K < M ∧ e % K < K := by
  have hK : 0 < K := by
    rcases Nat.eq_zero_or_pos K with hK | hK
    · subst hK; simp at h
    · exact hK
  exact ⟨(Nat.div_lt_iff_lt_mul hK).2 h, Nat.mod_lt _ hK⟩

/-! ## 3. Exactly-once writes

`D` is the set of loop coordinates a code path visits, `enc d` the output index it writes and
`sc d` the scale / zero-point index it reads. -/

structure Exact {α : Type} (D : α → Prop) (enc sc : α → Nat) (total : Nat) (σ : Nat → Nat) : Prop where
  in_range : ∀ d, D d → enc d < total
  inj : ∀ d₁ d₂, D d₁ → D d₂ → enc d₁ = enc d₂ → d₁ = d₂
  surj : ∀ e, e < total → ∃ d, D d ∧ enc d = e
  scale : ∀ d, D d → sc d = σ (enc d)

/-- Scalar per-axis loop: element `e = (m*K + k)*N + n` uses scale `k`. -/
def σAxis (K N : Nat) (e : Nat) : Nat := e / N % K

/-- Scalar blocked loop: element `e = (m*K + k)*N + n` uses scale `(m*nblk + k/B)*N + n`
(its `scale += N` / `zp_index += N` pointer bumps per block). -/
def σBlock (K N B nblk : Nat) (e : Nat) : Nat := (e / N / K * nblk + e / N % K / B) * N + e % N

/-- Generic two-level split `e = a*C + i` (`i < C`, `e < L`): every element of `[0, L)` is hit once. -/
theorem chunk_exact {C L : Nat} (hC : 0 < C) :
    Exact (fun d : Nat × Nat => d.2 < C ∧ d.1 * C + d.2 < L) (fun d => d.1 * C + d.2) (fun _ => 0) L
      (fun _ => 0) where
  in_range := fun d h => h.2
  inj := by
    rintro ⟨a₁, i₁⟩ ⟨a₂, i₂⟩ h₁ h₂ he
    simp only at he h₁ h₂
    have e1 := divmod a₁ i₁ hC h₁.1
    have e2 := divmod a₂ i₂ hC h₂.1
    rw [he] at e1
    exact Prod.ext (e1.1.symm.trans e2.1) (e1.2.symm.trans e2.2)
  surj := fun e he => ⟨(e / C, e % C), ⟨Nat.mod_lt _ hC, by rw [recon]; exact he⟩, recon e C⟩
  scale := fun _ _ => rfl

theorem pos_of_lt_mul {e M K : Nat} (h : e < M * K) : 0 < K := by
  rcases Nat.eq_zero_or_pos K with hK | hK
  · subst hK; simp at h
  · exact hK

/-! ## 4. `UnpackSubByteElements(src, first, len, dst)`

A leading scalar loop handles `i < a` (it stops at the first `i` with `i = len` or
`(first + i) % E = 0`), then the byte loop writes `dst[a + b*E + j]` from byte
`src[(first + a)/E + b]`, sub-element `j`, then a trailing scalar loop handles the rest.
The byte loop reads exactly element `first + (a + b*E + j)` and stays inside `dst[0, len)`. -/

theorem unpack_bulk {E first a len b j : Nat} (hE : 0 < E) (hj : j < E)
    (hstop : a = len ∨ (first + a) % E = 0) (hb : b < (len - a) / E) :
    (first + (a + b * E + j)) / E = (first + a) / E + b ∧
    (first + (a + b * E + j)) % E = j ∧
    a + b * E + j < len := by
  have hlen : a < len := by
    rcases Nat.eq_zero_or_pos (len - a) with h | h
    · rw [h, Nat.zero_div] at hb; omega
    · omega
  have ha : (first + a) % E = 0 := by
    rcases hstop with h | h
    · omega
    · exact h
  have hq := recon (first + a) E
  rw [ha, Nat.add_zero] at hq
  have heq : first + (a + b * E + j) = ((first + a) / E + b) * E + j := by
    rw [Nat.add_mul]; omega
  rw [heq]
  refine ⟨(divmod _ _ hE hj).1, (divmod _ _ hE hj).2, ?_⟩
  have h1 : b * E + j < (len - a) / E * E := lin2 hb hj
  have h2 := Nat.div_mul_le_self (len - a) E
  omega

/-! ## 5. Rows split into chunks

Element `e` of the scalar loop lies in row `e / N` at column `e % N`, and both scalar loops read
scale index `f (e / N) (e % N)`:
* per-axis: `f r n = r % K`;
* blocked: `f r n = (r / K * nblk + r % K / B) * N + n`.

Unit `u < R * cpr` (`cpr = ceil(N / C)`) is row `r = u / cpr`, chunk `c = u % cpr`; it writes
`r*N + c*C + i` for `i < C`, `c*C + i < N`. -/

theorem σAxis_rows (K N e : Nat) : σAxis K N e = (fun r (_ : Nat) => r % K) (e / N) (e % N) := rfl
theorem σBlock_rows (K N B nblk e : Nat) :
    σBlock K N B nblk e = (fun r n => (r / K * nblk + r % K / B) * N + n) (e / N) (e % N) := rfl

theorem chunked_rows_exact {R N C : Nat} (f : Nat → Nat → Nat) (hC : 0 < C) :
    let cpr := (N + C - 1) / C
    Exact (fun d : Nat × Nat => d.1 < R * cpr ∧ d.2 < C ∧ d.1 % cpr * C + d.2 < N)
      (fun d => d.1 / cpr * N + (d.1 % cpr * C + d.2))
      (fun d => f (d.1 / cpr) (d.1 % cpr * C + d.2)) (R * N) (fun e => f (e / N) (e % N)) := by
  intro cpr
  refine ⟨?_, ?_, ?_, ?_⟩
  · rintro ⟨u, i⟩ ⟨hu, -, hx⟩
    exact lin2 (split_lt hu).1 hx
  · rintro ⟨u₁, i₁⟩ ⟨u₂, i₂⟩ ⟨-, hi₁, hx₁⟩ ⟨-, hi₂, hx₂⟩ he
    simp only at he hx₁ hx₂ hi₁ hi₂
    have hN : 0 < N := Nat.lt_of_le_of_lt (Nat.zero_le _) hx₁
    have a1 := divmod (u₁ / cpr) _ hN hx₁
    have a2 := divmod (u₂ / cpr) _ hN hx₂
    rw [he] at a1
    have b1 := divmod (u₁ % cpr) _ hC hi₁
    have b2 := divmod (u₂ % cpr) _ hC hi₂
    rw [a1.2.symm.trans a2.2] at b1
    refine Prod.ext ?_ (b1.2.symm.trans b2.2)
    rw [← recon u₁ cpr, ← recon u₂ cpr, a1.1.symm.trans a2.1, b1.1.symm.trans b2.1]
  · intro e he
    have ⟨hr, hx⟩ := split_lt he
    have hc : e % N / C < cpr := ceil_gt hC (Nat.lt_of_le_of_lt (Nat.div_mul_le_self _ _) hx)
    have hcpr : 0 < cpr := Nat.lt_of_le_of_lt (Nat.zero_le _) hc
    have hd := divmod (e / N) _ hcpr hc
    refine ⟨(e / N * cpr + e % N / C, e % N % C), ⟨lin2 hr hc, Nat.mod_lt _ hC, ?_⟩, ?_⟩
    · simp only; rw [hd.2, recon]; exact hx
    · simp only; rw [hd.1, hd.2, recon]; exact recon e N
  · rintro ⟨u, i⟩ ⟨-, -, hx⟩
    simp only
    have hN : 0 < N := Nat.lt_of_le_of_lt (Nat.zero_le _) hx
    rw [(divmod _ _ hN hx).1, (divmod _ _ hN hx).2]

/-- Per-axis, quantize axis last (`N == 1`): rows of length `K`, scale index = column. -/
theorem axis_last_exact {M K C : Nat} (hC : 0 < C) :
    let cpr := (K + C - 1) / C
    Exact (fun d : Nat × Nat => d.1 < M * cpr ∧ d.2 < C ∧ d.1 % cpr * C + d.2 < K)
      (fun d => d.1 / cpr * K + (d.1 % cpr * C + d.2))
      (fun d => d.1 % cpr * C + d.2) (M * K) (σAxis K 1) := by
  intro cpr
  have h := chunked_rows_exact (R := M) (N := K) (fun _ x => x) hC
  refine ⟨h.in_range, h.inj, h.surj, fun d hd => ?_⟩
  have hx := hd.2.2
  have hK : 0 < K := Nat.lt_of_le_of_lt (Nat.zero_le _) hx
  simp only [σAxis, Nat.div_one]
  exact ((divmod _ _ hK hx).2).symm

/-- Per-axis, `N >= C`: one scale per row. -/
theorem axis_long_rows_exact {M K N C : Nat} (hC : 0 < C) :
    let cpr := (N + C - 1) / C
    Exact (fun d : Nat × Nat => d.1 < M * K * cpr ∧ d.2 < C ∧ d.1 % cpr * C + d.2 < N)
      (fun d => d.1 / cpr * N + (d.1 % cpr * C + d.2))
      (fun d => d.1 / cpr % K) (M * K * N) (σAxis K N) :=
  chunked_rows_exact (R := M * K) (fun r _ => r % K) hC

/-- Blocked, `N >= C`: `N` consecutive scales per row. -/
theorem block_long_rows_exact {M K N C B nblk : Nat} (hC : 0 < C) :
    let cpr := (N + C - 1) / C
    Exact (fun d : Nat × Nat => d.1 < M * K * cpr ∧ d.2 < C ∧ d.1 % cpr * C + d.2 < N)
      (fun d => d.1 / cpr * N + (d.1 % cpr * C + d.2))
      (fun d => (d.1 / cpr / K * nblk + d.1 / cpr % K / B) * N + (d.1 % cpr * C + d.2))
      (M * K * N) (σBlock K N B nblk) :=
  chunked_rows_exact (R := M * K) (fun r n => (r / K * nblk + r % K / B) * N + n) hC

/-! ## 6. Short rows (`1 < N < C`): several whole rows per unit -/

/-- Short rows (`1 < N < C`): unit `u` covers rows `u*Rpu + t` (`t < Rpu = C / N`) and writes
`(u*Rpu + t)*N + i`. -/
theorem short_rows_exact {R N C : Nat} (f : Nat → Nat → Nat) (hN : 0 < N) (hNC : N ≤ C) :
    let rpu := C / N
    Exact (fun d : Nat × Nat × Nat =>
        d.1 < (R + rpu - 1) / rpu ∧ d.2.1 < rpu ∧ d.1 * rpu + d.2.1 < R ∧ d.2.2 < N)
      (fun d => (d.1 * rpu + d.2.1) * N + d.2.2)
      (fun d => f (d.1 * rpu + d.2.1) d.2.2) (R * N) (fun e => f (e / N) (e % N)) := by
  intro rpu
  have hrpu : 0 < rpu := (Nat.le_div_iff_mul_le hN).2 (by omega)
  refine ⟨?_, ?_, ?_, ?_⟩
  · rintro ⟨u, t, i⟩ ⟨-, -, hr, hi⟩
    exact lin2 hr hi
  · rintro ⟨u₁, t₁, i₁⟩ ⟨u₂, t₂, i₂⟩ ⟨-, ht₁, -, hi₁⟩ ⟨-, ht₂, -, hi₂⟩ he
    simp only at he ht₁ ht₂ hi₁ hi₂
    have a1 := divmod (u₁ * rpu + t₁) _ hN hi₁
    have a2 := divmod (u₂ * rpu + t₂) _ hN hi₂
    rw [he] at a1
    have b1 := divmod u₁ _ hrpu ht₁
    have b2 := divmod u₂ _ hrpu ht₂
    rw [a1.1.symm.trans a2.1] at b1
    rw [b1.1.symm.trans b2.1, b1.2.symm.trans b2.2, a1.2.symm.trans a2.2]
  · intro e he
    have ⟨hr, hi⟩ := split_lt he
    have hu : e / N / rpu * rpu < R := Nat.lt_of_le_of_lt (Nat.div_mul_le_self _ _) hr
    refine ⟨(e / N / rpu, e / N % rpu, e % N),
      ⟨ceil_gt hrpu hu, Nat.mod_lt _ hrpu, by rw [recon]; exact hr, hi⟩, ?_⟩
    simp only; rw [recon, recon]
  · rintro ⟨u, t, i⟩ ⟨-, -, -, hi⟩
    simp only
    rw [(divmod _ _ hN hi).1, (divmod _ _ hN hi).2]

/-- Short rows stay inside the `C`-element stack buffer: index `t*N + i < C`. -/
theorem short_rows_buffer {N C t i : Nat} (hN : 0 < N) (ht : t < C / N) (hi : i < N) : t * N + i < C := by
  have h1 : t * N + i < (C / N) * N := lin2 ht hi
  have h2 := Nat.div_mul_le_self C N
  omega

/-- The short-row loop tracks `(m, k, block, in_block)` with counters instead of dividing:
`++k; if k == K then (k, m, block, in_block) := (0, m+1, 0, 0)
 else if blocked && ++in_block == B then (in_block, block) := (0, block+1)`.
Starting from the division-based values at row `r`, one step gives the values at row `r + 1`. -/
def rowStep (K B : Nat) (st : Nat × Nat × Nat × Nat) : Nat × Nat × Nat × Nat :=
  let (m, k, blk, ib) := st
  if k + 1 = K then (m + 1, 0, 0, 0)
  else if ib + 1 = B then (m, k + 1, blk + 1, 0)
  else (m, k + 1, blk, ib + 1)

def rowPos (K B r : Nat) : Nat × Nat × Nat × Nat := (r / K, r % K, r % K / B, r % K % B)

theorem rowStep_correct {K B : Nat} (hK : 0 < K) (hB : 0 < B) (r : Nat) :
    rowStep K B (rowPos K B r) = rowPos K B (r + 1) := by
  have hk : r % K < K := Nat.mod_lt _ hK
  have hr : r + 1 = r / K * K + (r % K + 1) := by have := recon r K; omega
  have hb : r % K % B < B := Nat.mod_lt _ hB
  have hkb : r % K + 1 = r % K / B * B + (r % K % B + 1) := by have := recon (r % K) B; omega
  simp only [rowStep, rowPos]
  by_cases h : r % K + 1 = K
  · -- k + 1 = K: next row starts a new m
    simp only [h, ↓reduceIte]
    have : r + 1 = (r / K + 1) * K + 0 := by rw [Nat.succ_mul]; omega
    rw [this, (divmod (r / K + 1) 0 hK hK).1, (divmod (r / K + 1) 0 hK hK).2, Nat.zero_div, Nat.zero_mod]
  · simp only [h, ↓reduceIte]
    have hlt : r % K + 1 < K := by omega
    rw [hr, (divmod _ _ hK hlt).1, (divmod _ _ hK hlt).2]
    by_cases h2 : r % K % B + 1 = B
    · simp only [h2, ↓reduceIte]
      have : r % K + 1 = (r % K / B + 1) * B + 0 := by rw [Nat.succ_mul]; omega
      rw [this, (divmod _ 0 hB hB).1, (divmod (r % K / B + 1) 0 hB hB).2]
    · simp only [h2, ↓reduceIte]
      have hlt2 : r % K % B + 1 < B := by omega
      rw [hkb, (divmod _ _ hB hlt2).1, (divmod _ _ hB hlt2).2]

/-! Short-row instances. -/

theorem axis_short_rows_exact {M K N C : Nat} (hN : 0 < N) (hNC : N ≤ C) :
    Exact (fun d : Nat × Nat × Nat =>
        d.1 < (M * K + C / N - 1) / (C / N) ∧ d.2.1 < C / N ∧ d.1 * (C / N) + d.2.1 < M * K ∧ d.2.2 < N)
      (fun d => (d.1 * (C / N) + d.2.1) * N + d.2.2) (fun d => (d.1 * (C / N) + d.2.1) % K)
      (M * K * N) (σAxis K N) :=
  short_rows_exact (fun r _ => r % K) hN hNC

theorem block_short_rows_exact {M K N C B nblk : Nat} (hN : 0 < N) (hNC : N ≤ C) :
    Exact (fun d : Nat × Nat × Nat =>
        d.1 < (M * K + C / N - 1) / (C / N) ∧ d.2.1 < C / N ∧ d.1 * (C / N) + d.2.1 < M * K ∧ d.2.2 < N)
      (fun d => (d.1 * (C / N) + d.2.1) * N + d.2.2)
      (fun d => ((d.1 * (C / N) + d.2.1) / K * nblk + (d.1 * (C / N) + d.2.1) % K / B) * N + d.2.2)
      (M * K * N) (σBlock K N B nblk) :=
  short_rows_exact (fun r n => (r / K * nblk + r % K / B) * N + n) hN hNC

/-! ## 7. Blocked, quantize axis last (`N == 1`)

With `N = 1` the blocked scalar loop's scale index for element `e` is `e / K * nblk + e % K / B`. -/

theorem σBlock_last (K B nblk e : Nat) : σBlock K 1 B nblk e = e / K * nblk + e % K / B := by
  simp [σBlock, Nat.mod_one]

/-- Blocks grouped `bpu` at a time (`bpu = max(1, C / B)`): unit `u < M * upr` is row
`m = u / upr`, group `g = u % upr`; block `g*bpu + j` (`j < bpu`) writes
`m*K + (g*bpu + j)*B + i` (`i < B`) with scale `m*nblk + g*bpu + j`. This also covers
`bpu = 1`, where `DequantizeSubByteRun` walks the block in `C`-element chunks: there the write
order inside one block differs but the set of (index, scale) pairs is the same. -/
theorem block_last_exact {M K B nblk bpu : Nat} (hB : 0 < B) (hbpu : 0 < bpu)
    (hnblk : nblk = (K + B - 1) / B) :
    let upr := (nblk + bpu - 1) / bpu
    Exact (fun d : Nat × Nat × Nat => d.1 < M * upr ∧ d.2.1 < bpu ∧ d.2.2 < B ∧
        (d.1 % upr * bpu + d.2.1) * B + d.2.2 < K)
      (fun d => d.1 / upr * K + ((d.1 % upr * bpu + d.2.1) * B + d.2.2))
      (fun d => d.1 / upr * nblk + (d.1 % upr * bpu + d.2.1)) (M * K) (σBlock K 1 B nblk) := by
  intro upr
  refine ⟨?_, ?_, ?_, ?_⟩
  · rintro ⟨u, j, i⟩ ⟨hu, -, -, hx⟩
    exact lin2 (split_lt hu).1 hx
  · rintro ⟨u₁, j₁, i₁⟩ ⟨u₂, j₂, i₂⟩ ⟨-, hj₁, hi₁, hx₁⟩ ⟨-, hj₂, hi₂, hx₂⟩ he
    simp only at he hj₁ hj₂ hi₁ hi₂ hx₁ hx₂
    have hK : 0 < K := Nat.lt_of_le_of_lt (Nat.zero_le _) hx₁
    have a1 := divmod (u₁ / upr) _ hK hx₁
    have a2 := divmod (u₂ / upr) _ hK hx₂
    rw [he] at a1
    have b1 := divmod (u₁ % upr * bpu + j₁) _ hB hi₁
    have b2 := divmod (u₂ % upr * bpu + j₂) _ hB hi₂
    rw [a1.2.symm.trans a2.2] at b1
    have c1 := divmod (u₁ % upr) _ hbpu hj₁
    have c2 := divmod (u₂ % upr) _ hbpu hj₂
    rw [b1.1.symm.trans b2.1] at c1
    have hu : u₁ = u₂ := by
      rw [← recon u₁ upr, ← recon u₂ upr, a1.1.symm.trans a2.1, c1.1.symm.trans c2.1]
    rw [hu, c1.2.symm.trans c2.2, b1.2.symm.trans b2.2]
  · intro e he
    have ⟨hm, hx⟩ := split_lt he
    -- x = e % K, block = x / B, group = block / bpu
    have hblk : e % K / B < nblk := by
      rw [hnblk]; exact ceil_gt hB (Nat.lt_of_le_of_lt (Nat.div_mul_le_self _ _) hx)
    have hg : e % K / B / bpu < upr :=
      ceil_gt hbpu (Nat.lt_of_le_of_lt (Nat.div_mul_le_self _ _) hblk)
    have hupr : 0 < upr := Nat.lt_of_le_of_lt (Nat.zero_le _) hg
    have hd := divmod (e / K) _ hupr hg
    refine ⟨(e / K * upr + e % K / B / bpu, e % K / B % bpu, e % K % B),
      ⟨lin2 hm hg, Nat.mod_lt _ hbpu, Nat.mod_lt _ hB, ?_⟩, ?_⟩
    · simp only; rw [hd.2, recon, recon]; exact hx
    · simp only; rw [hd.1, hd.2, recon, recon, recon]
  · rintro ⟨u, j, i⟩ ⟨-, -, hi, hx⟩
    simp only at hi hx ⊢
    rw [σBlock_last]
    have hK : 0 < K := Nat.lt_of_le_of_lt (Nat.zero_le _) hx
    rw [(divmod _ _ hK hx).1, (divmod _ _ hK hx).2, (divmod _ _ hB hi).1]

/-- The C++ block loop `for (k = k_begin; k < k_end; k += B)` with
`i < min(B, k_end - k)`, `k_end = min(K, k_begin + bpu*B)`, visits exactly
`j < bpu ∧ i < B ∧ k_begin + j*B + i < K`. -/
theorem block_loop_bounds {K B bpu kb j i : Nat} :
    (kb + j * B < min K (kb + bpu * B) ∧ i < min B (min K (kb + bpu * B) - (kb + j * B))) ↔
      (j < bpu ∧ i < B ∧ kb + j * B + i < K) := by
  constructor
  · rintro ⟨h1, h2⟩
    have hj : j * B < bpu * B := by omega
    exact ⟨Nat.lt_of_mul_lt_mul_right hj, by omega, by omega⟩
  · rintro ⟨hj, hi, hk⟩
    have : (j + 1) * B ≤ bpu * B := Nat.mul_le_mul_right _ hj
    rw [Nat.succ_mul] at this
    omega

/-- The grouped blocks stay inside the `C`-element stack buffer when `bpu * B ≤ C`. -/
theorem block_last_buffer {B C bpu j i : Nat} (hbuf : bpu * B ≤ C) (hj : j < bpu) (hi : i < B) :
    j * B + i < C := Nat.lt_of_lt_of_le (lin2 hj hi) hbuf

#print axioms decode_int4
#print axioms decode_int2
#print axioms unpack_bulk
#print axioms chunked_rows_exact
#print axioms axis_last_exact
#print axioms axis_long_rows_exact
#print axioms block_long_rows_exact
#print axioms short_rows_exact
#print axioms axis_short_rows_exact
#print axioms block_short_rows_exact
#print axioms rowStep_correct
#print axioms block_last_exact
#print axioms block_loop_bounds
