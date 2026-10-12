// Host-side reference of the HERA stream cipher (Cho et al., "Transciphering
// Framework for Approximate Homomorphic Encryption", ASIACRYPT 2021).
//
// The cipher follows plainHera / SampleZqx of the authors' reference
// implementation (KAIST-CryptLab/RtF-Transciphering, examples/ckks_fv). The
// XOF is AES-128-CTR (rubato::AesCtrXof, seed nonce || counter) instead of
// SHAKE256, like our Rubato. State index 4 * row + col; round constants are
// sampled round-major, 16 per ARK.
//
//   HERA = Fin o RF_{r-1} o ... o RF_1 o ARK_0,
//   RF_i = ARK_i o Cube o MixRows o MixColumns,
//   Fin  = ARK_r o MixRows o MixColumns o Cube o MixRows o MixColumns,
//   ARK_i(x) = x + k . rc_i,  MixColumns / MixRows = circ(2, 3, 1, 1).
//
// Checked against dev-tools/hera_ref.py.

#ifndef HEONGPU_RTF_HERA_REF_H
#define HEONGPU_RTF_HERA_REF_H

#include "rtf/rubato.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace heongpu
{
namespace hera
{
    struct Params
    {
        int n; // block size (16, a 4 x 4 state)
        int rounds; // r
        uint64_t q; // plaintext modulus
    };

    // Table 4 of the paper (Par-128a: 25-bit t, r = 5); q as in RtFHeraParams
    // "128af" of the reference.
    inline constexpr Params Par128a = {16, 5, 0x1fc0001ULL};

    // MixColumns / MixRows row: circ(2, 3, 1, 1)
    inline constexpr uint64_t kC4[4] = {2, 3, 1, 1};

    // MixRows o MixColumns = M (x) M as a 16 x 16 matrix (index 4 row + col):
    // L[4r + c][4r' + c'] = M[r][r'] * M[c][c'], M[i][j] = kC4[(j - i) mod 4].
    inline uint64_t linear_entry(int i, int j)
    {
        const int r = i / 4, c = i % 4, r2 = j / 4, c2 = j % 4;
        return kC4[(r2 - r + 4) % 4] * kC4[(c2 - c + 4) % 4];
    }

    // rc[r][i] for r = 0..rounds, i = 0..15, sampled round-major.
    inline std::vector<std::vector<uint64_t>>
    round_constants(const std::vector<uint8_t>& seed, const Params& p)
    {
        rubato::AesCtrXof xof(seed);
        std::vector<std::vector<uint64_t>> rc(p.rounds + 1,
                                              std::vector<uint64_t>(p.n));
        for (int r = 0; r <= p.rounds; r++)
            for (int i = 0; i < p.n; i++)
                rc[r][i] = rubato::sample_zqx(xof, p.q);
        return rc;
    }

    // MixRows o MixColumns on the 4 x 4 state.
    inline void linear_layer(std::vector<uint64_t>& st, uint64_t q)
    {
        std::vector<uint64_t> buf(16);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
            {
                uint64_t s = 0;
                for (int k = 0; k < 4; k++)
                    s += kC4[(k - r + 4) % 4] * st[4 * k + c];
                buf[4 * r + c] = s % q;
            }
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
            {
                uint64_t s = 0;
                for (int k = 0; k < 4; k++)
                    s += kC4[(k - c + 4) % 4] * buf[4 * r + k];
                st[4 * r + c] = s % q;
            }
    }

    inline void cube(std::vector<uint64_t>& st, uint64_t q)
    {
        for (auto& x : st)
            x = static_cast<uint64_t>((unsigned __int128) x * x % q * x % q);
    }

    // Keystream of one block: 16 words in [0, q).
    inline std::vector<uint64_t> keystream(const std::vector<uint64_t>& key,
                                           const std::vector<uint8_t>& seed,
                                           const Params& p)
    {
        if (p.n != 16 || (int) key.size() != p.n)
            throw std::invalid_argument("hera: n = 16 only");
        const uint64_t q = p.q;
        auto rc = round_constants(seed, p);
        auto ark = [&](std::vector<uint64_t>& st, int r)
        {
            for (int i = 0; i < p.n; i++)
                st[i] = static_cast<uint64_t>(
                    (st[i] + (unsigned __int128) rc[r][i] * key[i]) % q);
        };
        std::vector<uint64_t> st(p.n);
        for (int i = 0; i < p.n; i++)
            st[i] = static_cast<uint64_t>(i + 1); // ic
        ark(st, 0);
        for (int r = 1; r < p.rounds; r++)
        {
            linear_layer(st, q);
            cube(st, q);
            ark(st, r);
        }
        linear_layer(st, q);
        cube(st, q);
        linear_layer(st, q);
        ark(st, p.rounds);
        return st;
    }

} // namespace hera
} // namespace heongpu

#endif // HEONGPU_RTF_HERA_REF_H
