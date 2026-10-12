# Local patches to third-party submodules

The submodule pointers stay pinned; these changes live in the submodule working
trees, so `git submodule update` would drop them. Re-apply with
`bash patches/apply.sh` (idempotent).

- `gpu-ntt.patch` -> `thirdparty/GPU-NTT`
  - `uint128_t::operator>>` / `operator<<` shifted a 64-bit word by 64 when the
    shift was 0 or 64, which is undefined in C++. A 61-bit Barrett modulus
    (the BFV decryption modulus gamma) shifts by exactly bit + 3 = 64, and
    CUDA 13 miscompiles it: every BFV decryption came out wrong. Fixed by
    handling shift 0 and >= 64 explicitly.
  - `Modulus::bit_generator` used floating-point log2, so moduli just below a
    power of two (within ~5760 of 2^61) got one bit too many and Barrett
    reduction broke; now an integer bit length.
  - fast16: radix-8 Shoup NTT for N = 2^16 (from klss_lazy `gpu-ntt-fast16.patch`,
    ported to this GPU-NTT version), enabled with `GPUNTT_FAST16=1`, bit-identical
    to the merge NTT (test/test_ntt_fast16.cu); its Shoup-table cache is
    validated against the live table on every call.
  - `compute_shoup` / `mult_shoup` / `reduce_shoup` helpers (from klss_lazy
    `patches/gpu-ntt-shoup.patch`, hand-ported to this GPU-NTT version).
- `rngongpu-gaussian-sign.patch` -> `thirdparty/RNGonGPU`
  (klss_lazy `patches/rngongpu-gaussian-sign.patch`, paths rewritten): negative
  Gaussian draws were clamped to 0, the ordered variant read past its scratch
  and never scaled by the standard deviation.
