// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SlotToCoeff_FV (HEHERA<RTF>::S2C_FV): an FV ciphertext whose slot i holds
// v[i] becomes one whose plaintext coefficient i is v[i]. Checked on a
// random vector.
//
//   ./15_rtf_fv_slot2coeff [logN] [keyswitching method: 1 or 2]
//
// Defaults: logN = 14, KEYSWITCHING_METHOD_II (19_rtf_rubato covers
// METHOD_I on a keystream). S2C_FV is a BSGS matrix product: the giant-step
// key goes to the HEHERA constructor, the baby-step key (powers of two below
// g2 = ceil(sqrt(N/2)), plus the column swap every Galois key carries) must
// be set with set_galois_key_bs() first.

#include "heongpu.cuh"
#include "../example_util.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::RTF;
constexpr auto SchemeCKKS = heongpu::Scheme::CKKS;

static void print_first(const char* tag, const std::vector<uint64_t>& v,
                        size_t k)
{
    std::cout << tag << " [0:" << k << "): ";
    for (size_t i = 0; i < k; ++i)
        std::cout << v[i] << (i + 1 < k ? ' ' : '\n');
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    const int logN = (argc > 1) ? std::atoi(argv[1]) : 14;
    const int method = (argc > 2) ? std::atoi(argv[2]) : 2;
    const size_t N = size_t(1) << logN;
    const auto ks = (method == 1)
                        ? heongpu::keyswitching_type::KEYSWITCHING_METHOD_I
                        : heongpu::keyswitching_type::KEYSWITCHING_METHOD_II;

    heongpu::HEContext<Scheme> context(ks, heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(N);
    // METHOD_I takes exactly one P prime, METHOD_II at least two.
    const std::vector<int> logQ = {58, 58, 58, 58, 59};
    const std::vector<int> logP = (method == 1) ? std::vector<int>{59}
                                                : std::vector<int>{59, 59};
    context.set_coeff_modulus_bit_sizes(logQ, logP);
    // HEHERA is tied to Rubato: the plain modulus must be its q.
    context.set_plain_modulus(heongpu::rubato::Par128L.q);
    context.generate();
    context.print_parameters();
    std::cout << "keyswitching method " << method << std::endl;

    // HEHERA also needs a CKKS context of the same degree (unused here).
    heongpu::HEContext<SchemeCKKS> contextckks(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        heongpu::sec_level_type::none);
    contextckks.set_poly_modulus_degree(N);
    contextckks.set_coeff_modulus_bit_sizes({60, 50}, {60});
    contextckks.generate();

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);
    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    // Giant steps of the BSGS: multiples of g2 below N/2.
    const int H = static_cast<int>(N / 2);
    const int g2 = static_cast<int>(std::ceil(std::sqrt((double) H)));
    std::set<int> giant;
    for (int r = 0; r < H; ++r)
    {
        const int rot = (r - r % g2) % H;
        if (rot != 0)
            giant.insert(rot);
    }
    std::vector<int> giant_shifts(giant.begin(), giant.end());
    heongpu::Galoiskey<Scheme> galois_key_gs(context, giant_shifts);
    keygen.generate_galois_key(galois_key_gs, secret_key);

    // Baby steps: powers of two below g2.
    std::vector<int> baby_shifts;
    for (int s = 1; s < g2; s <<= 1)
        baby_shifts.push_back(s);
    heongpu::Galoiskey<Scheme> galois_key_bs(context, baby_shifts);
    keygen.generate_galois_key(galois_key_bs, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);
    heongpu::HEHERA<Scheme> hera(context, contextckks, encoder, encryptor, op,
                                 galois_key_gs, relin_key);
    hera.set_galois_key_bs(galois_key_bs);
    hera.gen_FV_S2C_Matrix();

    const uint64_t t = context.get_plain_modulus().value;
    std::mt19937_64 rng(15);
    std::vector<uint64_t> slots(N);
    for (auto& v : slots)
        v = rng() % t;
    print_first("slots (expected coefficients)", slots, 16);

    heongpu::Plaintext<Scheme> pt(context);
    encoder.encode(pt, slots);
    heongpu::Ciphertext<Scheme> ct(context);
    encryptor.encrypt(ct, pt);

    heongpu::Ciphertext<Scheme> ct_s2c = hera.S2C_FV(ct);
    std::cout << "noise budget after S2C_FV: "
              << decryptor.remainder_noise_budget(ct_s2c) << " bits"
              << std::endl;

    heongpu::Plaintext<Scheme> pt_out(context);
    decryptor.decrypt(pt_out, ct_s2c);
    std::vector<uint64_t> coeffs(N);
    cudaDeviceSynchronize();
    cudaMemcpy(coeffs.data(), pt_out.data(), N * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    print_first("decrypted coefficients       ", coeffs, 16);

    size_t bad = 0;
    for (size_t i = 0; i < N; ++i)
        bad += (coeffs[i] != slots[i]);
    const bool ok = (bad == 0);
    std::cout << (ok ? "[PASS] " : "[FAIL] ")
              << "S2C_FV: coefficient i == slot i (" << N - bad << "/" << N
              << ")" << std::endl;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
