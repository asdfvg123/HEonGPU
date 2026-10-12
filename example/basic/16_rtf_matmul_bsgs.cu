// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Baby-step giant-step matrix-vector product on RTF (FV) slots with
// HEArithmeticOperator<RTF>::multiply_matrix_bsgs (the routine behind
// HEHERA::S2C_FV), for a sparse matrix, checked against the plaintext
// product.
//
// Slots form two rows of H = N/2. Diagonal s of the N x N matrix pairs slot x
// with slot col(x, s):
//     s in [0, H):  same row,  column (col(x) + s) mod H        -> group 0
//     s in [H, N):  other row, column (col(x) + s - H) mod H    -> group 1
// and result[x] = sum_s diag_s[x] * v[col(x, s)].
//
// Calling convention: matrix_groups[g][0] holds the encoded diagonals of group
// g back to back (N words each, in the order of shifts[g]); each diagonal is
// pre-rotated right by its giant step G = floor(r / g2) * g2 (r = s mod H,
// g2 = ceil(sqrt(H))), because the routine rotates the inner sum by G after
// multiplying. Group 1 may be omitted. Keys: baby steps = powers of two below
// g2 plus the row swap (every Galois key has it), giant steps = the G used.

#include "heongpu.cuh"
#include "../example_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::RTF;

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    const size_t N = 8192;
    context.set_poly_modulus_degree(N);
    context.set_coeff_modulus_default_values(1);
    const uint64_t t = 65537;
    context.set_plain_modulus(t);
    context.generate();
    context.print_parameters();

    const int H = static_cast<int>(N / 2);
    const int g2 = static_cast<int>(std::ceil(std::sqrt((double) H)));
    std::cout << "BSGS: H = " << H << ", g2 = " << g2 << std::endl;

    // A sparse matrix: diagonals of both groups, inside and across giant
    // steps, including the last one of each row.
    const std::vector<std::vector<int>> shifts = {
        {0, 1, 5, g2 - 1, g2, g2 + 2, 200, 1000, H - 1},
        {H, H + 3, H + g2, H + 130, static_cast<int>(N) - 1}};

    auto giant_of = [&](int s) { return (s % H) / g2 * g2; };

    std::set<int> giant;
    for (const auto& group : shifts)
        for (int s : group)
            if (giant_of(s) != 0)
                giant.insert(giant_of(s));
    std::vector<int> giant_shifts(giant.begin(), giant.end());
    std::vector<int> baby_shifts;
    for (int s = 1; s < g2; s <<= 1)
        baby_shifts.push_back(s);

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);
    heongpu::Galoiskey<Scheme> galois_key_gs(context, giant_shifts);
    keygen.generate_galois_key(galois_key_gs, secret_key);
    heongpu::Galoiskey<Scheme> galois_key_bs(context, baby_shifts);
    keygen.generate_galois_key(galois_key_bs, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    std::mt19937_64 rng(16);
    std::vector<uint64_t> v(N);
    for (auto& x : v)
        x = rng() % t;
    heongpu::Plaintext<Scheme> P1(context);
    encoder.encode(P1, v);
    heongpu::Ciphertext<Scheme> C1(context);
    encryptor.encrypt(C1, P1);

    // Random diagonal values diag[g][k][x].
    std::vector<std::vector<std::vector<uint64_t>>> diag(shifts.size());
    for (size_t g = 0; g < shifts.size(); ++g)
        for (size_t k = 0; k < shifts[g].size(); ++k)
        {
            std::vector<uint64_t> d(N);
            for (auto& x : d)
                x = rng() % t;
            diag[g].push_back(std::move(d));
        }

    // Encode one group: pre-rotate every diagonal right by its giant step
    // within each row, encode, concatenate.
    auto encode_group = [&](size_t g)
    {
        heongpu::DeviceVector<Data64> blob(shifts[g].size() * N);
        heongpu::Plaintext<Scheme> pt(context);
        for (size_t k = 0; k < shifts[g].size(); ++k)
        {
            const int G = giant_of(shifts[g][k]);
            std::vector<uint64_t> d(N);
            for (int x = 0; x < (int) N; ++x)
            {
                const int row = x / H * H;
                d[row + (x % H + G) % H] = diag[g][k][x];
            }
            encoder.encode(pt, d);
            cudaMemcpy(blob.data() + k * N, pt.data(), N * sizeof(Data64),
                       cudaMemcpyDeviceToDevice);
        }
        std::vector<heongpu::DeviceVector<Data64>> out;
        out.push_back(std::move(blob));
        return out;
    };

    // Plaintext reference over the first `groups` groups.
    auto reference = [&](size_t groups)
    {
        std::vector<uint64_t> out(N);
        for (int x = 0; x < (int) N; ++x)
        {
            const int row = x / H * H;
            unsigned __int128 acc = 0;
            for (size_t g = 0; g < groups; ++g)
            {
                const int src_row = (g == 0) ? row : (H - row);
                for (size_t k = 0; k < shifts[g].size(); ++k)
                {
                    const int r = shifts[g][k] % H;
                    const int src = src_row + (x % H + r) % H;
                    acc += static_cast<unsigned __int128>(diag[g][k][x]) * v[src];
                }
            }
            out[x] = static_cast<uint64_t>(acc % t);
        }
        return out;
    };

    auto decrypt = [&](heongpu::Ciphertext<Scheme>& c)
    {
        heongpu::Plaintext<Scheme> p(context);
        decryptor.decrypt(p, c);
        std::vector<uint64_t> out;
        encoder.decode(out, p);
        return out;
    };

    bool all_ok = true;
    auto check = [&](const char* what, const std::vector<uint64_t>& got,
                     const std::vector<uint64_t>& expected)
    {
        size_t bad = 0;
        for (size_t i = 0; i < N; ++i)
            bad += (got[i] != expected[i]);
        std::cout << (bad == 0 ? "[PASS] " : "[FAIL] ") << what << " ("
                  << N - bad << "/" << N << " slots match)" << std::endl;
        all_ok &= (bad == 0);
    };

    std::vector<std::vector<heongpu::DeviceVector<Data64>>> groups;
    groups.push_back(encode_group(0));
    groups.push_back(encode_group(1));

    // 1) Both groups.
    {
        heongpu::Ciphertext<Scheme> C = operators.multiply_matrix_bsgs(
            C1, groups, shifts, galois_key_gs, galois_key_bs);
        std::cout << "noise budget after multiply_matrix_bsgs: "
                  << decryptor.remainder_noise_budget(C) << " bits" << std::endl;
        const auto got = decrypt(C);
        const auto expected = reference(2);
        std::cout << "Expected:";
        display_matrix(expected, H);
        std::cout << "Decrypted:";
        display_matrix(got, H);
        check("multiply_matrix_bsgs, both groups == plaintext product", got,
              expected);
    }

    // 2) Group 0 only (no row swap needed).
    {
        std::vector<std::vector<heongpu::DeviceVector<Data64>>> g0;
        g0.push_back(encode_group(0));
        std::vector<std::vector<int>> s0 = {shifts[0]};
        heongpu::Ciphertext<Scheme> C = operators.multiply_matrix_bsgs(
            C1, g0, s0, galois_key_gs, galois_key_bs);
        check("multiply_matrix_bsgs, group 0 only == plaintext product",
              decrypt(C), reference(1));
    }

    // 3) A malformed call (shift of group 0 outside [0, H)) is rejected
    //    instead of reading out of bounds.
    {
        std::vector<std::vector<heongpu::DeviceVector<Data64>>> g0;
        g0.push_back(encode_group(0));
        std::vector<std::vector<int>> bad = {{0, H + 1}};
        bool threw = false;
        try
        {
            operators.multiply_matrix_bsgs(C1, g0, bad, galois_key_gs,
                                           galois_key_bs);
        }
        catch (const std::invalid_argument& e)
        {
            threw = true;
            std::cout << "rejected as expected: " << e.what() << std::endl;
        }
        std::cout << (threw ? "[PASS] " : "[FAIL] ")
                  << "malformed shifts throw std::invalid_argument" << std::endl;
        all_ok &= threw;
    }

    std::cout << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED")
              << std::endl;
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
