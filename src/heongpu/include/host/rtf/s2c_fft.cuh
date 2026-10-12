// SlotToCoeff_FV as a product of sparse (FFT-like) slot-domain matrices.
//
// BFV slots in HEonGPU: slot (r, k), r in {0, 1}, k in [0, N/2), holds
// p(psi^e) with e = 3^k (r = 0) or -3^k (r = 1) mod 2N; rotate_rows(s) maps
// slot k <- k + s inside each row and rotate_columns swaps the rows.
//
// SlotToCoeff with the dense N x N matrix needs ~sqrt(N) giant-step keys and
// N plaintext diagonals. Here the input slot (r, k) is instead taken to be the
// coefficient pi(r, k) = r*N/2 + bitrev_{log(N/2)}(k). With that layout the
// evaluation map factors like the CKKS special FFT:
//   * one dense 4x4 stage per pair of adjacent slots that mixes both rows
//     (mod 4, 3 = -1, so the two cosets of <3> do not separate there);
//   * log2(N) - 2 butterfly stages with 3 diagonals {0, +-N_l/4}.
// Consecutive butterfly stages are merged into groups (levels_per_group) and
// each group is evaluated with baby-step/giant-step (SlotLinearTransform:
// hoisted baby steps, one key per baby shift, giants by Horner's rule).
// The base stage's five rotations (column, +-1 of x, +-1 of the swapped x)
// are hoisted too (two decompositions), and its six plaintext products are
// accumulated in the NTT domain.
//
// pi is the per-row bit reversal that CKKS CoeffToSlot undoes, so after
// HalfBoot the CKKS slots come out in the BFV slot order.

#ifndef HEONGPU_RTF_S2C_FFT_H
#define HEONGPU_RTF_S2C_FFT_H

#include "rtf/context.cuh"
#include "rtf/encoder.cuh"
#include "rtf/plaintext.cuh"
#include "rtf/ciphertext.cuh"
#include "rtf/evaluationkey.cuh"
#include "rtf/operator.cuh"
#include "rtf/slot_lintrans.cuh"

#include <map>
#include <memory>
#include <vector>

namespace heongpu
{
    class FVSlotToCoeffFFT
    {
      public:
        __host__ FVSlotToCoeffFFT(HEContext<Scheme::RTF>& context,
                                  HEEncoder<Scheme::RTF>& encoder,
                                  HEOperator<Scheme::RTF>& op,
                                  int levels_per_group = 5,
                                  const ExecutionOptions& options =
                                      ExecutionOptions());

        // Row rotations whose Galois keys must exist (the column rotation is
        // needed as well).
        __host__ std::vector<int> rotation_shifts() const;

        // Coefficient that slot i ends up in: r*N/2 + bitrev(k).
        __host__ int coeff_index(int slot) const;

        // Input slots v -> plaintext with coefficient coeff_index(i) = v[i].
        __host__ Ciphertext<Scheme::RTF>
        apply(Ciphertext<Scheme::RTF>& input, Galoiskey<Scheme::RTF>& gk,
              const ExecutionOptions& options = ExecutionOptions());

        __host__ int group_count() const { return (int) groups_.size(); }

      private:
        using DiagMap = SlotLinearTransform::DiagMap;

        __host__ int canon(int rot) const; // into (-H/2, H/2]
        __host__ std::vector<uint64_t> rot_vec(const std::vector<uint64_t>& v,
                                               int a) const;
        __host__ DiagMap compose(const DiagMap& A, const DiagMap& B) const;
        __host__ DiagMap butterfly(int level) const; // N_l = 2^level
        __host__ Plaintext<Scheme::RTF> encode(const std::vector<uint64_t>& v);

        HEContext<Scheme::RTF>& context_;
        HEEncoder<Scheme::RTF>& encoder_;
        HEOperator<Scheme::RTF>& operator_;
        ExecutionOptions options_;

        int n_;
        int H_;
        int log_n_;
        uint64_t t_;
        uint64_t psi_;

        // base stage: the six diagonals lifted to [6][Q][N] (NTT form) and
        // the packed input each one multiplies (x, rot+-1 x, xs, rot+-1 xs)
        std::shared_ptr<DeviceVector<Data64>> base_slab_;
        std::shared_ptr<DeviceVector<int>> base_j_;
        std::vector<SlotLinearTransform> groups_;
    };

} // namespace heongpu

#endif // HEONGPU_RTF_S2C_FFT_H
