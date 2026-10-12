#include "rtf/s2c_fft.cuh"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace heongpu
{
    namespace
    {
        uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m)
        {
            return static_cast<uint64_t>((unsigned __int128) a * b % m);
        }

        uint64_t powmod(uint64_t b, uint64_t e, uint64_t m)
        {
            uint64_t r = 1;
            b %= m;
            while (e)
            {
                if (e & 1)
                    r = mulmod(r, b, m);
                b = mulmod(b, b, m);
                e >>= 1;
            }
            return r;
        }

        int bitrev(int x, int bits)
        {
            int r = 0;
            for (int i = 0; i < bits; i++)
                r |= ((x >> i) & 1) << (bits - 1 - i);
            return r;
        }
    } // namespace

    __host__ FVSlotToCoeffFFT::FVSlotToCoeffFFT(
        HEContext<Scheme::RTF>& context, HEEncoder<Scheme::RTF>& encoder,
        HEOperator<Scheme::RTF>& op, int levels_per_group,
        const ExecutionOptions& options)
        : context_(context), encoder_(encoder), operator_(op),
          options_(options)
    {
        n_ = static_cast<int>(context.get_poly_modulus_degree());
        H_ = n_ / 2;
        log_n_ = static_cast<int>(std::log2(n_));
        if (n_ < 8 || (1 << log_n_) != n_)
            throw std::invalid_argument("FVSlotToCoeffFFT: N must be 2^k >= 8");
        if (levels_per_group < 1)
            throw std::invalid_argument("FVSlotToCoeffFFT: levels_per_group");
        Modulus64 plain_mod = context.get_plain_modulus();
        t_ = plain_mod.value;
        if ((t_ - 1) % (2 * (uint64_t) n_) != 0)
            throw std::invalid_argument(
                "FVSlotToCoeffFFT: t must be 1 mod 2N (full batching)");
        // Same root as the encoder's plaintext NTT tables.
        psi_ = find_minimal_primitive_root(2 * n_, plain_mod);

        // ---- base stage: size-4 evaluations, both rows mixed --------------
        // block (base, base+1) in each row holds P_m, m = 2 r' + delta.
        const uint64_t z4 = powmod(psi_, n_ / 4, t_); // primitive 8th root
        std::vector<std::vector<uint64_t>> dplain(3, std::vector<uint64_t>(n_, 0));
        std::vector<std::vector<uint64_t>> dswap(3, std::vector<uint64_t>(n_, 0));
        for (int base = 0; base < H_; base += 2)
            for (int r = 0; r < 2; r++)
                for (int kap = 0; kap < 2; kap++)
                {
                    int e = (kap == 0) ? 1 : 3;
                    if (r == 1)
                        e = (8 - e) % 8;
                    for (int rp = 0; rp < 2; rp++)
                        for (int delta = 0; delta < 2; delta++)
                        {
                            const int m = 2 * rp + delta;
                            const int rot = delta - kap; // new[k] = old[k+rot]
                            const uint64_t c = powmod(z4, (uint64_t) (m * e % 8), t_);
                            auto& d = (rp == r) ? dplain : dswap;
                            d[rot + 1][r * H_ + base + kap] = c;
                        }
                }
        // Packed inputs of the base stage: slot 0 = x, 1 = rot_1(x),
        // 2 = rot_-1(x), 3 = xs (rows swapped), 4 = rot_1(xs), 5 = rot_-1(xs).
        {
            cudaStream_t stream = options.stream_;
            const int slot_of_rot[3] = {2, 0, 1}; // rot = -1, 0, 1
            DeviceVector<Data64> blob((size_t) 6 * n_, stream);
            std::vector<int> k_list, j_list;
            for (int m = 0; m < 6; m++)
            {
                const int i = m % 3; // rot + 1
                Plaintext<Scheme::RTF> pt =
                    encode(m < 3 ? dplain[i] : dswap[i]);
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    blob.data() + (size_t) m * n_, pt.data(),
                    (size_t) n_ * sizeof(Data64), cudaMemcpyDeviceToDevice,
                    stream));
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                k_list.push_back(m);
                j_list.push_back(slot_of_rot[i] + (m < 3 ? 0 : 3));
            }
            base_slab_ = std::make_shared<DeviceVector<Data64>>(
                op.batched_plain_to_ntt_slab(blob.data(), k_list, stream));
            base_j_ = std::make_shared<DeviceVector<int>>(j_list, stream);
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // ---- butterfly stages, merged into groups -------------------------
        for (int l0 = 3; l0 <= log_n_; l0 += levels_per_group)
        {
            const int l1 = std::min(log_n_, l0 + levels_per_group - 1);
            DiagMap M = butterfly(l0);
            for (int l = l0 + 1; l <= l1; l++)
                M = compose(butterfly(l), M);

            groups_.emplace_back(context, encoder, op, M, options);
        }
    }

    __host__ int FVSlotToCoeffFFT::canon(int rot) const
    {
        return SlotLinearTransform::canon(rot, H_);
    }

    __host__ std::vector<uint64_t>
    FVSlotToCoeffFFT::rot_vec(const std::vector<uint64_t>& v, int a) const
    {
        return SlotLinearTransform::rotate(v, a);
    }

    // (A o B) x = A(B x),  A(z) = sum_a dA_a . rot_a(z)
    __host__ FVSlotToCoeffFFT::DiagMap
    FVSlotToCoeffFFT::compose(const DiagMap& A, const DiagMap& B) const
    {
        DiagMap res;
        for (auto& [a, da] : A)
            for (auto& [b, db] : B)
            {
                auto rb = rot_vec(db, a);
                auto& out = res[canon(a + b)];
                if (out.empty())
                    out.assign(n_, 0);
                for (int i = 0; i < n_; i++)
                    out[i] = (out[i] + mulmod(da[i], rb[i], t_)) % t_;
            }
        // drop all-zero diagonals
        for (auto it = res.begin(); it != res.end();)
        {
            bool zero = std::all_of(it->second.begin(), it->second.end(),
                                    [](uint64_t x) { return x == 0; });
            it = zero ? res.erase(it) : std::next(it);
        }
        return res;
    }

    // Butterfly of the size-N_l problems: blocks of H_l = N_l/2 slots per row,
    // a = first half, b = second half, w = zeta_l^{+-3^kk}.
    __host__ FVSlotToCoeffFFT::DiagMap FVSlotToCoeffFFT::butterfly(int level) const
    {
        const int Nl = 1 << level, Hl = Nl / 2, half = Nl / 4;
        const uint64_t zl = powmod(psi_, n_ / Nl, t_);
        DiagMap D;
        auto& d0 = D[0];
        d0.assign(n_, 0);
        auto& dp = D[canon(half)];
        if (dp.empty())
            dp.assign(n_, 0);
        auto& dm = D[canon(-half)];
        if (dm.empty())
            dm.assign(n_, 0);
        for (int r = 0; r < 2; r++)
            for (int k = 0; k < H_; k++)
            {
                const int kl = k % Hl, kk = kl % half;
                uint64_t e = powmod(3, kk, 2 * (uint64_t) Nl);
                if (r == 1)
                    e = (2 * (uint64_t) Nl - e) % (2 * (uint64_t) Nl);
                const uint64_t w = powmod(zl, e, t_);
                const int idx = r * H_ + k;
                if (kl < half)
                {
                    d0[idx] = (d0[idx] + 1) % t_;
                    dp[idx] = (dp[idx] + w) % t_;
                }
                else
                {
                    d0[idx] = (d0[idx] + t_ - w) % t_;
                    dm[idx] = (dm[idx] + 1) % t_;
                }
            }
        return D;
    }

    __host__ Plaintext<Scheme::RTF>
    FVSlotToCoeffFFT::encode(const std::vector<uint64_t>& v)
    {
        Plaintext<Scheme::RTF> pt(context_);
        encoder_.encode(pt, v, options_);
        return pt;
    }

    __host__ std::vector<int> FVSlotToCoeffFFT::rotation_shifts() const
    {
        std::set<int> s = {1, -1};
        for (auto& g : groups_)
            for (int v : g.rotation_shifts())
                s.insert(v);
        return std::vector<int>(s.begin(), s.end());
    }

    __host__ int FVSlotToCoeffFFT::coeff_index(int slot) const
    {
        const int r = slot / H_, k = slot % H_;
        return r * H_ + bitrev(k, log_n_ - 1);
    }

    __host__ Ciphertext<Scheme::RTF>
    FVSlotToCoeffFFT::apply(Ciphertext<Scheme::RTF>& input,
                            Galoiskey<Scheme::RTF>& gk,
                            const ExecutionOptions& options)
    {
        auto coeff = [&](Ciphertext<Scheme::RTF>& c)
        {
            if (c.in_ntt_domain())
                operator_.transform_from_ntt_inplace(c, options);
        };
        cudaStream_t stream = options.stream_;
        Ciphertext<Scheme::RTF> x = input;
        coeff(x);

        // base stage: hoisted rotations of x and of xs, one batched NTT,
        // the six products accumulated in the NTT domain, one INTT.
        {
            const size_t cw = (size_t) 2 * context_.get_ciphertext_modulus_count() * n_;
            DeviceVector<Data64> packed(6 * cw, stream);
            Data64* p = packed.data();
            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(p, x.data(),
                                               cw * sizeof(Data64),
                                               cudaMemcpyDeviceToDevice,
                                               stream));
            const int e_p1 = operator_.row_galois_elt(gk, 1);
            const int e_m1 = operator_.row_galois_elt(gk, -1);
            const int e_col = operator_.column_galois_elt(gk);
            operator_.apply_galois_hoisted_raw(
                x.data(), {e_col, e_p1, e_m1}, {p + 3 * cw, p + cw, p + 2 * cw},
                gk, stream);
            operator_.apply_galois_hoisted_raw(
                p + 3 * cw, {e_p1, e_m1}, {p + 4 * cw, p + 5 * cw}, gk, stream);
            operator_.ntt_ciphertexts_inplace(p, 6, true, stream);

            Ciphertext<Scheme::RTF> y = x; // shape only (coefficient form)
            HEONGPU_CUDA_CHECK(
                cudaMemsetAsync(y.data(), 0, cw * sizeof(Data64), stream));
            operator_.accumulate_plain_products_ntt(
                p, base_slab_->data(), base_j_->data(), 6, y.data(), stream);
            operator_.ntt_ciphertexts_inplace(y.data(), 1, false, stream);
            x = std::move(y);
        }

        // butterfly groups (baby-step/giant-step, giants by Horner)
        for (auto& g : groups_)
            x = g.apply(operator_, context_, x, gk, options);
        coeff(x);
        return x;
    }

} // namespace heongpu
