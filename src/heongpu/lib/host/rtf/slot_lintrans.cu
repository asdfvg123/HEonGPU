#include "rtf/slot_lintrans.cuh"

#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

namespace heongpu
{
    int SlotLinearTransform::canon(int rot, int half_n)
    {
        int r = ((rot % half_n) + half_n) % half_n;
        if (r > half_n / 2)
            r -= half_n;
        return r;
    }

    std::vector<uint64_t>
    SlotLinearTransform::rotate(const std::vector<uint64_t>& v, int a)
    {
        const int n = static_cast<int>(v.size()), h = n / 2;
        const int sh = ((a % h) + h) % h;
        std::vector<uint64_t> out(n);
        for (int r = 0; r < 2; r++)
            for (int k = 0; k < h; k++)
                out[r * h + k] = v[r * h + (k + sh) % h];
        return out;
    }

    __host__ SlotLinearTransform::SlotLinearTransform(
        HEContext<Scheme::RTF>& context, HEEncoder<Scheme::RTF>& encoder,
        HEOperator<Scheme::RTF>& op, const DiagMap& diags,
        const ExecutionOptions& options, int base_rotation, int baby_steps)
    {
        cudaStream_t stream = options.stream_;
        const int n = static_cast<int>(context.get_poly_modulus_degree());
        H_ = n / 2;
        if (diags.empty())
            throw std::invalid_argument("SlotLinearTransform: no diagonals");

        // canonical rotations and their common unit
        std::map<int, const std::vector<uint64_t>*> canon_diags;
        int g = 0;
        for (auto& [rot, d] : diags)
        {
            if ((int) d.size() != n)
                throw std::invalid_argument("SlotLinearTransform: size");
            const int c = canon(rot, H_);
            if (canon_diags.count(c))
                throw std::invalid_argument(
                    "SlotLinearTransform: duplicate rotation");
            canon_diags[c] = &d;
            g = std::gcd(g, std::abs(c));
        }
        s_ = (g == 0) ? 1 : g;
        int dmax = 0;
        bool first = true;
        for (auto& kv : canon_diags)
        {
            const int d = kv.first / s_;
            dmin_ = first ? d : std::min(dmin_, d);
            dmax = first ? d : std::max(dmax, d);
            first = false;
        }
        if (base_rotation != kAuto)
        {
            const int b = canon(base_rotation, H_);
            if (b % s_ != 0 || b / s_ > dmin_)
                throw std::invalid_argument(
                    "SlotLinearTransform: base_rotation must be a multiple "
                    "of the rotation unit and <= every diagonal rotation");
            dmin_ = b / s_;
        }
        const int range = dmax - dmin_ + 1;
        if (baby_steps > 0)
        {
            n1_ = std::min(baby_steps, range);
        }
        else
        {
            // Baby steps are hoisted, giant steps are full key switches:
            // pick n1 minimising babies * c_baby + (giants - 1) * c_giant +
            // giants * c_intt, with the relative costs measured at N = 2^16,
            // |Q| = 24, |P| = 4, dnum = 6 (hoisted rotation 1, full key
            // switch 2.5, INTT of a giant step's sum 0.36).
            double best = 0;
            for (int m = 1; m <= range; m++)
            {
                int babies = 0;
                for (int b = 0; b < m; b++)
                    babies += (canon((dmin_ + b) * s_, H_) != 0);
                const int g = (range + m - 1) / m;
                const double c = babies * 1.0 + (g - 1) * 2.5 + g * 0.36;
                if (m == 1 || c < best - 1e-9)
                {
                    best = c;
                    n1_ = m;
                }
            }
        }
        giants_ = (range + n1_ - 1) / n1_;
        diag_count_ = static_cast<int>(canon_diags.size());
        Q_ = static_cast<int>(context.get_ciphertext_modulus_count());

        // Encode every pre-rotated diagonal (coefficient form, N words) into
        // one device blob, then lift + NTT each giant step's set at once.
        DeviceVector<Data64> blob((size_t) diag_count_ * n, stream);
        std::vector<std::vector<int>> k_lists(giants_), b_lists(giants_);
        int k = 0;
        for (int i = 0; i < giants_; i++)
        {
            const int G = i * n1_ * s_;
            for (int b = 0; b < n1_; b++)
            {
                auto it = canon_diags.find((dmin_ + i * n1_ + b) * s_);
                if (it == canon_diags.end())
                    continue;
                // rot_G(D') = D  ->  D' = rot_{-G}(D)
                Plaintext<Scheme::RTF> pt(context);
                encoder.encode(pt, rotate(*it->second, -G), options);
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    blob.data() + (size_t) k * n, pt.data(),
                    (size_t) n * sizeof(Data64), cudaMemcpyDeviceToDevice,
                    stream));
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                k_lists[i].push_back(k++);
                b_lists[i].push_back(b);
            }
        }
        for (int i = 0; i < giants_; i++)
        {
            count_.push_back(static_cast<int>(k_lists[i].size()));
            if (k_lists[i].empty())
            {
                slab_.push_back(nullptr);
                baby_.push_back(nullptr);
                continue;
            }
            slab_.push_back(std::make_shared<DeviceVector<Data64>>(
                op.batched_plain_to_ntt_slab(blob.data(), k_lists[i], stream)));
            baby_.push_back(
                std::make_shared<DeviceVector<int>>(b_lists[i], stream));
        }
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    __host__ std::vector<int> SlotLinearTransform::rotation_shifts() const
    {
        std::set<int> s;
        for (int b = 0; b < n1_; b++)
        {
            const int c = canon((dmin_ + b) * s_, H_);
            if (c != 0)
                s.insert(c);
        }
        if (giants_ > 1)
        {
            const int c = canon(n1_ * s_, H_);
            if (c != 0)
                s.insert(c);
        }
        return std::vector<int>(s.begin(), s.end());
    }

    __host__ int SlotLinearTransform::key_switch_count() const
    {
        int babies = 0;
        for (int b = 0; b < n1_; b++)
            babies += (canon((dmin_ + b) * s_, H_) != 0);
        return babies + (giants_ - 1);
    }

    __host__ Ciphertext<Scheme::RTF> SlotLinearTransform::apply(
        HEOperator<Scheme::RTF>& op, HEContext<Scheme::RTF>& context,
        Ciphertext<Scheme::RTF>& input, Galoiskey<Scheme::RTF>& gk,
        const ExecutionOptions& options) const
    {
        cudaStream_t stream = options.stream_;
        auto coeff = [&](Ciphertext<Scheme::RTF>& c)
        {
            if (c.in_ntt_domain())
                op.transform_from_ntt_inplace(c, options);
        };

        if (input.coeff_modulus_count() != Q_)
            throw std::invalid_argument(
                "SlotLinearTransform: ciphertext level does not match");
        const size_t ct_words = (size_t) 2 * Q_ * (2 * H_);

        Ciphertext<Scheme::RTF> x_copy;
        Ciphertext<Scheme::RTF>* xp = &input;
        if (input.in_ntt_domain())
        {
            x_copy = input;
            coeff(x_copy);
            xp = &x_copy;
        }
        Ciphertext<Scheme::RTF>& x = *xp;

        // ---- baby steps rot_{(dmin+b)*s}(x), hoisted, packed ------------
        DeviceVector<Data64> packed((size_t) n1_ * ct_words, stream);
        std::vector<int> elts, shifts;
        std::vector<Data64*> outs;
        bool exact = true;
        for (int b = 0; b < n1_; b++)
        {
            Data64* dst = packed.data() + (size_t) b * ct_words;
            const int sh = canon((dmin_ + b) * s_, H_);
            if (sh == 0)
            {
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    dst, x.data(), ct_words * sizeof(Data64),
                    cudaMemcpyDeviceToDevice, stream));
                continue;
            }
            const int elt = op.row_galois_elt(gk, sh);
            exact = exact && op.has_galois_key(gk, elt);
            elts.push_back(elt);
            shifts.push_back(sh);
            outs.push_back(dst);
        }
        if (exact)
        {
            op.apply_galois_hoisted_raw(x.data(), elts, outs, gk, stream);
        }
        else
        {
            // no exact key for some baby step: plain rotations of x
            for (size_t k = 0; k < shifts.size(); k++)
            {
                Ciphertext<Scheme::RTF> r(context);
                op.rotate_rows(x, r, gk, shifts[k], options);
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    outs[k], r.data(), ct_words * sizeof(Data64),
                    cudaMemcpyDeviceToDevice, stream));
            }
        }
        op.ntt_ciphertexts_inplace(packed.data(), n1_, true, stream);

        // ---- giant steps by Horner: acc = rot_{n1*s}(acc) + u_i ---------
        Ciphertext<Scheme::RTF> acc;
        bool has = false;
        for (int i = giants_ - 1; i >= 0; i--)
        {
            if (has)
            {
                Ciphertext<Scheme::RTF> r(context);
                op.rotate_rows(acc, r, gk, canon(n1_ * s_, H_), options);
                acc = std::move(r);
            }
            if (count_[i] == 0)
                continue;
            Ciphertext<Scheme::RTF> ui = x; // shape only (coefficient form)
            HEONGPU_CUDA_CHECK(cudaMemsetAsync(
                ui.data(), 0, ct_words * sizeof(Data64), stream));
            op.accumulate_plain_products_ntt(packed.data(), slab_[i]->data(),
                                             baby_[i]->data(), count_[i],
                                             ui.data(), stream);
            op.ntt_ciphertexts_inplace(ui.data(), 1, false, stream);
            if (!has)
            {
                acc = std::move(ui);
                has = true;
            }
            else
            {
                coeff(acc);
                op.add_inplace(acc, ui, options);
            }
        }
        if (!has)
            throw std::logic_error("SlotLinearTransform: all-zero transform");
        coeff(acc);
        return acc;
    }

} // namespace heongpu
