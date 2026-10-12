// Host-side reference of the Rubato stream cipher (Ha et al., "Rubato: Noisy
// Ciphers for Approximate Homomorphic Encryption", full version).
//
// The cipher follows plainRubato / SampleZqx of the authors' reference
// implementation (KAIST-CryptLab/RtF-Transciphering). The XOF is AES-128-CTR,
// one of the two instantiations of the paper, instead of the reference's
// SHAKE256. Only the Par-*L parameters (n = 64, v = 8) are supported, which is
// what the RtF server in hera.cu packs.

#ifndef HEONGPU_RTF_RUBATO_H
#define HEONGPU_RTF_RUBATO_H

#include <openssl/evp.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace heongpu
{
namespace rubato
{
    struct Params
    {
        int n; // block size (v * v)
        int l; // output words after truncation
        int rounds; // r
        uint64_t q; // plaintext modulus
        double sigma; // AGN standard deviation, alpha*q / sqrt(2*pi)
    };

    // Table 2 of the paper; q and sigma as in the reference implementation.
    inline constexpr Params Par80L = {64, 60, 2, 0x1fc0001ULL,
                                      0.63830764864229228470391369589501};
    inline constexpr Params Par128L = {64, 60, 2, 0x1fc0001ULL,
                                       1.63566334964587397955377884573096};

    // MDS row for v = 8: y8 = [5, 3, 4, 3, 6, 2, 1, 1], M8 = circ(y8)
    inline constexpr uint64_t kY8[8] = {5, 3, 4, 3, 6, 2, 1, 1};

    // Extendable output function: AES-128 in counter mode. The 16-byte block
    // seed (nonce || counter) is the AES key, the IV is zero, and the output
    // is the CTR keystream, i.e. AES_seed(0), AES_seed(1), ...
    class AesCtrXof
    {
      public:
        explicit AesCtrXof(const std::vector<uint8_t>& seed)
        {
            if (seed.size() != 16)
                throw std::invalid_argument("AES-CTR XOF needs a 16-byte seed");
            const uint8_t iv[16] = {0};
            ctx_ = EVP_CIPHER_CTX_new();
            if (ctx_ == nullptr ||
                EVP_EncryptInit_ex(ctx_, EVP_aes_128_ctr(), nullptr,
                                   seed.data(), iv) != 1)
            {
                throw std::runtime_error("AES-CTR init failed");
            }
        }
        ~AesCtrXof() { EVP_CIPHER_CTX_free(ctx_); }
        AesCtrXof(const AesCtrXof&) = delete;
        AesCtrXof& operator=(const AesCtrXof&) = delete;

        void squeeze(uint8_t* out, size_t len)
        {
            // CTR mode: encrypting zeros yields the keystream itself.
            std::vector<uint8_t> zeros(len, 0);
            int out_len = 0;
            if (EVP_EncryptUpdate(ctx_, out, &out_len, zeros.data(),
                                  static_cast<int>(len)) != 1 ||
                out_len != static_cast<int>(len))
            {
                throw std::runtime_error("AES-CTR squeeze failed");
            }
        }

      private:
        EVP_CIPHER_CTX* ctx_ = nullptr;
    };

    // Uniform sample in [0, q) by rejection, same as SampleZqx of the
    // reference: little-endian bytes, top byte masked to bitlen(q - 2) bits.
    inline uint64_t sample_zqx(AesCtrXof& xof, uint64_t q)
    {
        int bit_len = 64 - __builtin_clzll(q - 2);
        int byte_len = (bit_len + 7) / 8;
        int b = bit_len % 8;
        if (b == 0)
            b = 8;
        uint8_t bytes[8];
        while (true)
        {
            xof.squeeze(bytes, byte_len);
            bytes[byte_len - 1] &= static_cast<uint8_t>((1 << b) - 1);
            uint64_t res = 0;
            for (int i = 0; i < byte_len; i++)
                res += static_cast<uint64_t>(bytes[i]) << (8 * i);
            if (res < q)
                return res;
        }
    }

    // XOF input of one block: nonce || counter, the counter big-endian
    // ("inner-counter mode", z[ctr] = Rubato[k, nc || ctr]).
    inline std::vector<uint8_t> xof_seed(const std::array<uint8_t, 8>& nonce,
                                         uint64_t counter)
    {
        std::vector<uint8_t> seed(nonce.begin(), nonce.end());
        for (int i = 7; i >= 0; i--)
            seed.push_back(static_cast<uint8_t>(counter >> (8 * i)));
        return seed;
    }

    // rc[r][i] for r = 0..rounds, i = 0..n-1, sampled round-major.
    inline std::vector<std::vector<uint64_t>>
    round_constants(const std::vector<uint8_t>& seed, const Params& p)
    {
        AesCtrXof xof(seed);
        std::vector<std::vector<uint64_t>> rc(p.rounds + 1,
                                              std::vector<uint64_t>(p.n));
        for (int r = 0; r <= p.rounds; r++)
            for (int i = 0; i < p.n; i++)
                rc[r][i] = sample_zqx(xof, p.q);
        return rc;
    }

    // MixRows o MixColumns on the 8 x 8 state (row-major, index row*8+col).
    inline void linear_layer(std::vector<uint64_t>& st, uint64_t q)
    {
        std::vector<uint64_t> buf(64);
        for (int row = 0; row < 8; row++)
            for (int col = 0; col < 8; col++)
            {
                unsigned __int128 s = 0;
                for (int k = 0; k < 8; k++)
                    s += (unsigned __int128) kY8[k] *
                         st[((row + k) % 8) * 8 + col];
                buf[row * 8 + col] = static_cast<uint64_t>(s % q);
            }
        for (int row = 0; row < 8; row++)
            for (int col = 0; col < 8; col++)
            {
                unsigned __int128 s = 0;
                for (int k = 0; k < 8; k++)
                    s += (unsigned __int128) kY8[k] *
                         buf[row * 8 + (col + k) % 8];
                st[row * 8 + col] = static_cast<uint64_t>(s % q);
            }
    }

    inline void feistel(std::vector<uint64_t>& st, uint64_t q)
    {
        std::vector<uint64_t> buf = st;
        for (size_t i = 1; i < st.size(); i++)
            st[i] = static_cast<uint64_t>(
                (buf[i] + (unsigned __int128) buf[i - 1] * buf[i - 1]) % q);
    }

    // Rounded Gaussian with standard deviation sigma, cut at 6 sigma like the
    // reference's AGN.
    inline int64_t sample_noise(std::mt19937_64& rng, double sigma)
    {
        std::normal_distribution<double> dist(0.0, sigma);
        const double bound = std::floor(6.0 * sigma);
        while (true)
        {
            double e = std::round(dist(rng));
            if (std::fabs(e) <= bound)
                return static_cast<int64_t>(e);
        }
    }

    // Keystream of one block: l words in [0, q). With rng != nullptr the
    // Gaussian noise of AGN is added (client side); the server evaluates the
    // cipher without it.
    inline std::vector<uint64_t> keystream(const std::vector<uint64_t>& key,
                                           const std::vector<uint8_t>& seed,
                                           const Params& p,
                                           std::mt19937_64* rng = nullptr)
    {
        if (p.n != 64 || (int) key.size() != p.n)
            throw std::invalid_argument("rubato: only n = 64 is supported");
        const uint64_t q = p.q;
        auto rc = round_constants(seed, p);
        std::vector<std::vector<uint64_t>> rk(p.rounds + 1,
                                              std::vector<uint64_t>(p.n));
        for (int r = 0; r <= p.rounds; r++)
            for (int i = 0; i < p.n; i++)
                rk[r][i] = static_cast<uint64_t>(
                    (unsigned __int128) rc[r][i] * key[i] % q);

        std::vector<uint64_t> st(p.n);
        for (int i = 0; i < p.n; i++)
            st[i] = (static_cast<uint64_t>(i + 1) + rk[0][i]) % q; // ic

        for (int r = 1; r < p.rounds; r++)
        {
            linear_layer(st, q);
            feistel(st, q);
            for (int i = 0; i < p.n; i++)
                st[i] = (st[i] + rk[r][i]) % q;
        }
        linear_layer(st, q);
        feistel(st, q);
        linear_layer(st, q);
        if (rng != nullptr)
        {
            for (int i = 0; i < p.n; i++)
            {
                int64_t e = sample_noise(*rng, p.sigma);
                st[i] = static_cast<uint64_t>(
                    ((int64_t) st[i] + e % (int64_t) q + (int64_t) q) %
                    (int64_t) q);
            }
        }
        for (int i = 0; i < p.n; i++)
            st[i] = (st[i] + rk[p.rounds][i]) % q;
        st.resize(p.l);
        return st;
    }

} // namespace rubato
} // namespace heongpu

#endif // HEONGPU_RTF_RUBATO_H
