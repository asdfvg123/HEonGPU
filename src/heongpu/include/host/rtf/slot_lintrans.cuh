// Slot-domain linear transform  y = sum_r D_r . rot_r(x)  on BFV/RTF
// ciphertexts, evaluated with baby-step/giant-step:
//
//   y = sum_i rot_{i*n1*s}( sum_b D'_{i,b} . rot_{(dmin+b)*s}(x) ),
//   D'_{i,b} = rot_{-i*n1*s}(D_{(dmin + i*n1 + b)*s}),
//
// where s is the common unit of the rotations and dmin*s the lowest one.
// The n1 baby steps rot_{(dmin+b)*s}(x) are all rotations of the same input
// and are computed with hoisted key switching (one decomposition + ModUp of
// x, then one inner product + ModDown per rotation); the lowest rotation is
// folded into the baby steps, so no final offset rotation is needed. Giant
// steps use Horner's rule (acc = rot_{n1*s}(acc) + u_i), one key.
// Keys: every baby shift (dmin+b)*s, b < n1 (exact keys, hoisting cannot
// chain), plus n1*s. If a baby key is missing the baby steps fall back to
// plain rotate_rows of x.
//
// The diagonals are lifted to the full RNS basis and NTT-transformed once at
// construction; apply() transforms all baby steps with one batched NTT and
// accumulates every giant step's products in the NTT domain with one kernel.

#ifndef HEONGPU_RTF_SLOT_LINTRANS_H
#define HEONGPU_RTF_SLOT_LINTRANS_H

#include "rtf/context.cuh"
#include "rtf/encoder.cuh"
#include "rtf/plaintext.cuh"
#include "rtf/ciphertext.cuh"
#include "rtf/evaluationkey.cuh"
#include "rtf/operator.cuh"

#include <climits>
#include <map>
#include <memory>
#include <vector>

namespace heongpu
{
    class SlotLinearTransform
    {
      public:
        // rotation -> N slot values (row 0 then row 1); rotations are taken
        // mod N/2. All rotations must be multiples of a common unit.
        using DiagMap = std::map<int, std::vector<uint64_t>>;

        static constexpr int kAuto = INT_MIN;

        SlotLinearTransform() = default;

        // base_rotation: lowest rotation the baby steps start from (default:
        // the lowest diagonal). Must be <= every diagonal rotation and a
        // multiple of their common unit; transforms sharing a base share
        // their baby-step keys. baby_steps: n1 (default: minimises the
        // cost of hoisted babies vs full-key-switch giants, see the .cu).
        __host__ SlotLinearTransform(HEContext<Scheme::RTF>& context,
                                     HEEncoder<Scheme::RTF>& encoder,
                                     HEOperator<Scheme::RTF>& op,
                                     const DiagMap& diags,
                                     const ExecutionOptions& options =
                                         ExecutionOptions(),
                                     int base_rotation = kAuto,
                                     int baby_steps = 0);

        // Row rotations whose Galois keys apply() needs.
        __host__ std::vector<int> rotation_shifts() const;

        __host__ Ciphertext<Scheme::RTF>
        apply(HEOperator<Scheme::RTF>& op, HEContext<Scheme::RTF>& context,
              Ciphertext<Scheme::RTF>& input, Galoiskey<Scheme::RTF>& gk,
              const ExecutionOptions& options = ExecutionOptions()) const;

        __host__ int diagonal_count() const { return diag_count_; }
        __host__ int key_switch_count() const;

        // helpers on slot vectors (rows of N/2, cyclic)
        static std::vector<uint64_t> rotate(const std::vector<uint64_t>& v,
                                            int a);
        static int canon(int rot, int half_n);

      private:
        int H_ = 0; // N/2
        int s_ = 0; // rotation unit
        int n1_ = 0; // baby steps
        int dmin_ = 0; // lowest rotation / s (baby b rotates by (dmin+b)*s)
        int giants_ = 0;
        int diag_count_ = 0;
        int Q_ = 0; // RNS limbs the diagonals are lifted to
        // Giant step i: slab_[i] = [M_i][Q][N] NTT-form diagonals
        // rot_{-i*n1*s}(D_{(dmin + i*n1 + b)*s}), baby_[i][m] = b.
        std::vector<std::shared_ptr<DeviceVector<Data64>>> slab_;
        std::vector<std::shared_ptr<DeviceVector<int>>> baby_;
        std::vector<int> count_;
    };

} // namespace heongpu

#endif // HEONGPU_RTF_SLOT_LINTRANS_H
