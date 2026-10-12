// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// FV mod-down to q0 (HEHERA<RTF>::moddown_FV_inplace), checked exactly.
//
// An FV ciphertext (c0, c1) mod Q = q0 q1 ... qL with c0 + c1 s = Delta p + e
// (Delta = floor(Q / t)) is rescaled by q0 / Q one prime at a time, with
// rounding. The result mod q0 must satisfy
//     c0' + c1' s = round(q0 p / t) + e'   (mod q0)
// with a small e' (rounding noise ~ ||s||_1). This is the step in front of
// HalfBoot in the RtF pipeline; 19_rtf_rubato checks it only end to end (to
// 2^-6 after bootstrapping), this example checks it coefficient by
// coefficient. The secret key is built from a known ternary polynomial so
// the host can decrypt the mod-q0 ciphertext itself.

#include "heongpu.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::RTF;
constexpr auto SchemeCKKS = heongpu::Scheme::CKKS;
using i128 = __int128;
using u128 = unsigned __int128;

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    const int logN = (argc > 1) ? std::atoi(argv[1]) : 12;
    const size_t N = size_t(1) << logN;

    const std::vector<int> logQ = {60, 50, 50, 50, 50};
    const std::vector<int> logP = {60};

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(N);
    context.set_coeff_modulus_bit_sizes(logQ, logP);
    context.set_plain_modulus(heongpu::rubato::Par128L.q);
    context.generate();
    context.print_parameters();

    heongpu::HEContext<SchemeCKKS> contextckks(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        heongpu::sec_level_type::none);
    contextckks.set_poly_modulus_degree(N);
    contextckks.set_coeff_modulus_bit_sizes(logQ, logP);
    contextckks.generate();

    const uint64_t t = context.get_plain_modulus().value;
    const uint64_t q0 = context.get_key_modulus()[0].value;

    // Known ternary secret key.
    std::mt19937_64 rng(18);
    std::vector<int> s(N);
    for (auto& x : s)
        x = static_cast<int>(rng() % 3) - 1;
    heongpu::Secretkey<Scheme> secret_key(s, context);

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);
    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);
    std::vector<int> no_shifts;
    heongpu::Galoiskey<Scheme> galois_key(context, no_shifts); // unused

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);
    heongpu::HEHERA<Scheme> hera(context, contextckks, encoder, encryptor, op,
                                 galois_key, relin_key);

    std::vector<uint64_t> slots(N);
    for (auto& v : slots)
        v = rng() % t;
    heongpu::Plaintext<Scheme> pt(context);
    encoder.encode(pt, slots);
    std::vector<uint64_t> p(N); // plaintext polynomial, coefficients in [0, t)
    cudaDeviceSynchronize();
    cudaMemcpy(p.data(), pt.data(), N * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

    heongpu::Ciphertext<Scheme> ct(context);
    encryptor.encrypt(ct, pt);
    std::cout << "before: " << ct.coeff_modulus_count()
              << " moduli, noise budget " << decryptor.remainder_noise_budget(ct)
              << " bits" << std::endl;

    hera.moddown_FV_inplace(ct);
    cudaDeviceSynchronize();
    std::cout << "after : " << ct.coeff_modulus_count() << " modulus (q0 = "
              << q0 << ", " << std::log2((double) q0) << " bits)" << std::endl;

    bool all_ok = true;
    auto check = [&](const char* what, bool ok)
    {
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << std::endl;
        all_ok &= ok;
    };
    check("ciphertext is left with the single modulus q0",
          ct.coeff_modulus_count() == 1 && !ct.in_ntt_domain());

    // Host decryption mod q0: v = c0 + c1 * s in Z_q0[X] / (X^N + 1).
    std::vector<uint64_t> c(2 * N);
    cudaMemcpy(c.data(), ct.data(), 2 * N * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    std::vector<uint64_t> v(c.begin(), c.begin() + N);
    for (size_t j = 0; j < N; ++j)
    {
        if (s[j] == 0)
            continue;
        for (size_t i = 0; i < N; ++i)
        {
            // c1[i] * s[j] X^(i + j), X^N = -1
            size_t k = i + j;
            bool neg = (s[j] < 0);
            if (k >= N)
            {
                k -= N;
                neg = !neg;
            }
            const uint64_t a = c[N + i];
            v[k] = neg ? (v[k] >= a ? v[k] - a : v[k] + q0 - a)
                       : (v[k] + a >= q0 ? v[k] + a - q0 : v[k] + a);
        }
    }

    // Compare with round(q0 p / t).
    size_t bad = 0;
    double max_err = 0;
    for (size_t i = 0; i < N; ++i)
    {
        const uint64_t expected =
            static_cast<uint64_t>((static_cast<u128>(q0) * p[i] + t / 2) / t);
        i128 e = static_cast<i128>(v[i]) - static_cast<i128>(expected);
        if (e > static_cast<i128>(q0 / 2))
            e -= q0;
        if (e < -static_cast<i128>(q0 / 2))
            e += q0;
        max_err = std::max(max_err, std::fabs((double) e));
        // Decode: round(t v / q0) mod t.
        const uint64_t decoded = static_cast<uint64_t>(
            ((static_cast<u128>(v[i]) * t + q0 / 2) / q0) % t);
        bad += (decoded != p[i]);
    }
    std::cout << "max |e'| = " << max_err << " (2^" << std::log2(max_err + 1)
              << "), q0 / t = 2^" << std::log2((double) q0 / t) << std::endl;
    check("round(t (c0' + c1' s) / q0) == p for every coefficient", bad == 0);
    check("mod-down noise |e'| < 2^20", max_err < std::pow(2.0, 20));

    std::cout << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED")
              << std::endl;
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
