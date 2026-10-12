// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Regression test: BFV and RTF encrypt -> decrypt with the 61-bit decryption
// modulus gamma.
//
// HEContext::generate() draws gamma as the last of the internal primes, which
// are MAX_MOD_BIT_COUNT = 61 bits wide, so the Barrett reduction modulo gamma
// in decryption_kernel shifts a 128-bit value by bit + 3 = 64. That shift was
// undefined in GPU-NTT's uint128_t and CUDA 13 miscompiled it, which broke
// every BFV / RTF decryption (patches/gpu-ntt.patch). The device arithmetic
// itself is covered by test_modular_arith.cu; this test checks the full path.

#include "heongpu.cuh"
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace
{
    struct Params
    {
        size_t n;
        std::vector<int> log_q;
        std::vector<int> log_p;
        uint64_t t;
        heongpu::keyswitching_type method;
    };

    std::string describe(const Params& p)
    {
        std::string s = "N=" + std::to_string(p.n) + " t=" + std::to_string(p.t) +
                        " logQ={";
        for (int b : p.log_q)
            s += std::to_string(b) + ",";
        s += "} logP={";
        for (int b : p.log_p)
            s += std::to_string(b) + ",";
        return s + "}";
    }

    size_t count_mismatches(const std::vector<uint64_t>& a,
                            const std::vector<uint64_t>& b)
    {
        if (a.size() != b.size())
            return std::max(a.size(), b.size());
        size_t bad = 0;
        for (size_t i = 0; i < a.size(); ++i)
            bad += (a[i] != b[i]);
        return bad;
    }

    template <heongpu::Scheme S> void encrypt_decrypt(const Params& p)
    {
        SCOPED_TRACE(describe(p));

        // Every internal prime (gamma among them) is 61 bits.
        const int internal_count =
            static_cast<int>(p.log_q.size() + p.log_p.size()) + 2;
        for (const Modulus64& g :
             heongpu::generate_internal_primes(p.n, internal_count))
        {
            ASSERT_EQ(g.bit, 61ULL);
        }

        heongpu::HEContext<S> context(p.method, heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(p.n);
        context.set_coeff_modulus_bit_sizes(p.log_q, p.log_p);
        context.set_plain_modulus(p.t);
        context.generate();

        heongpu::HEKeyGenerator<S> keygen(context);
        heongpu::Secretkey<S> secret_key(context);
        keygen.generate_secret_key(secret_key);
        heongpu::Publickey<S> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder<S> encoder(context);
        heongpu::HEEncryptor<S> encryptor(context, public_key);
        heongpu::HEDecryptor<S> decryptor(context, secret_key);
        heongpu::HEArithmeticOperator<S> op(context, encoder);

        std::mt19937_64 rng(p.n ^ p.t);
        std::vector<uint64_t> m1(p.n), m2(p.n), sum(p.n);
        for (size_t i = 0; i < p.n; ++i)
        {
            m1[i] = rng() % p.t;
            m2[i] = rng() % p.t;
            sum[i] = (m1[i] + m2[i]) % p.t;
        }
        // Extremes of the plaintext range.
        m1[0] = 0;
        m1[1] = 1;
        m1[2] = p.t - 1;
        m1[3] = p.t / 2;
        m1[4] = (p.t + 1) / 2;
        for (int i = 0; i < 5; ++i)
            sum[i] = (m1[i] + m2[i]) % p.t;

        heongpu::Plaintext<S> p1(context), p2(context);
        encoder.encode(p1, m1);
        encoder.encode(p2, m2);

        auto decrypt_decode = [&](heongpu::Ciphertext<S>& c)
        {
            heongpu::Plaintext<S> pt(context);
            decryptor.decrypt(pt, c);
            std::vector<uint64_t> out;
            encoder.decode(out, pt);
            cudaDeviceSynchronize();
            return out;
        };

        // 1) Public-key encryption.
        heongpu::Ciphertext<S> c1(context), c2(context);
        encryptor.encrypt(c1, p1);
        encryptor.encrypt(c2, p2);
        // The budget is at most log2(Q / t); it used to be read from an
        // unsynchronized pinned copy and came out larger than that.
        int log_q_total = 0;
        for (int b : p.log_q)
            log_q_total += b;
        const int budget = decryptor.remainder_noise_budget(c1);
        EXPECT_GT(budget, 0);
        EXPECT_LE(budget, log_q_total - heongpu::calculate_bit_count(p.t) + 1);
        EXPECT_EQ(count_mismatches(decrypt_decode(c1), m1), 0u)
            << "Dec(Enc(m)) != m";

        // 2) Noise-free ciphertext (0, 0) + Delta * m: decryption alone.
        heongpu::Ciphertext<S> zero(context), noiseless(context);
        op.sub(c1, c1, zero);
        op.add_plain(zero, p1, noiseless);
        EXPECT_EQ(count_mismatches(decrypt_decode(noiseless), m1), 0u)
            << "Dec((0, 0) + Delta m) != m";

        // 3) Homomorphic addition.
        heongpu::Ciphertext<S> c_sum(context);
        op.add(c1, c2, c_sum);
        EXPECT_EQ(count_mismatches(decrypt_decode(c_sum), sum), 0u)
            << "Dec(Enc(m1) + Enc(m2)) != m1 + m2";
    }

} // namespace

TEST(HEonGPU, BFV_Gamma61_Encrypt_Decrypt)
{
    cudaSetDevice(0);
    const auto I = heongpu::keyswitching_type::KEYSWITCHING_METHOD_I;
    const std::vector<Params> params = {
        {4096, {36, 36}, {37}, 1032193, I},
        {8192, {54, 54, 54}, {55}, 786433, I},
        {8192, {60, 60, 60}, {60}, 65537, I},
        {16384, {54, 54, 54, 54, 55, 55, 55}, {55}, 0x1fc0001, I},
    };
    for (const Params& p : params)
        encrypt_decrypt<heongpu::Scheme::BFV>(p);
}

TEST(HEonGPU, RTF_Gamma61_Encrypt_Decrypt)
{
    cudaSetDevice(0);
    const auto I = heongpu::keyswitching_type::KEYSWITCHING_METHOD_I;
    const auto II = heongpu::keyswitching_type::KEYSWITCHING_METHOD_II;
    // The last two are the RtF / Rubato parameter shape of 19_rtf_rubato:
    // t = Rubato q = 0x1fc0001, a 60-bit q0 followed by 50-bit primes.
    std::vector<int> rubato_q(30, 50);
    rubato_q[0] = 60;
    const std::vector<Params> params = {
        {4096, {36, 36}, {37}, 1032193, I},
        {8192, {54, 54, 54}, {55}, 786433, I},
        {8192, {58, 58, 58}, {59, 59}, 0x1fc0001, II},
        {4096, {60, 50, 50, 50, 50, 50, 50, 50, 50, 50}, {60}, 0x1fc0001, I},
        {4096, rubato_q, {60}, 0x1fc0001, I},
    };
    for (const Params& p : params)
        encrypt_decrypt<heongpu::Scheme::RTF>(p);
}
