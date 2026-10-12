// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Diagonal (Halevi-Shoup) matrix-vector product on RTF (FV) slots with
// HEArithmeticOperator<RTF>::multiply_matrix, checked against the plaintext
// product.
//
// The 2 x (N/2) slot matrix is treated as two independent rows. For every
// diagonal k with rotation r_k:
//     result[i] = sum_k diag_k[i] * v[row(i) + (col(i) + r_k) mod N/2]
// (rotate_rows by r > 0 moves slot i + r to slot i). multiply_matrix
// decomposes each r_k into signed powers of two, so the Galois key only needs
// the power-of-two rotations.

#include "heongpu.cuh"
#include "../example_util.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::RTF;

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    const size_t N = 4096;
    context.set_poly_modulus_degree(N);
    context.set_coeff_modulus_default_values(1);
    const uint64_t t = 786433;
    context.set_plain_modulus(t);
    context.generate();
    context.print_parameters();

    const int H = static_cast<int>(N / 2);

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    // Signed powers of two below N/2 (the default Galoiskey would also try
    // 2^11 .. 2^15, which do not exist for N = 4096).
    std::vector<int> pow2_shifts;
    for (int s = 1; s < H; s <<= 1)
    {
        pow2_shifts.push_back(s);
        pow2_shifts.push_back(-s);
    }
    heongpu::Galoiskey<Scheme> galois_key(context, pow2_shifts);
    keygen.generate_galois_key(galois_key, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    std::mt19937_64 rng(13);
    std::vector<uint64_t> message(N);
    for (auto& v : message)
        v = rng() % t;

    heongpu::Plaintext<Scheme> P1(context);
    encoder.encode(P1, message);
    heongpu::Ciphertext<Scheme> C1(context);
    encryptor.encrypt(C1, P1);
    std::cout << "Initial noise budget: " << decryptor.remainder_noise_budget(C1)
              << " bits" << std::endl;

    // Diagonals: small, negative, wrapping and non-power-of-two rotations.
    std::vector<std::vector<int>> diags = {{0, 2, -2, 7, -100, 1000, H - 1}};
    const size_t D = diags[0].size();
    std::vector<std::vector<uint64_t>> diag_values(D, std::vector<uint64_t>(N));
    for (auto& d : diag_values)
        for (auto& v : d)
            v = rng() % t;

    // multiply_matrix takes the encoded diagonals of a group concatenated in
    // one device buffer (N plaintext words per diagonal).
    heongpu::DeviceVector<Data64> concatenated(D * N);
    for (size_t k = 0; k < D; ++k)
    {
        heongpu::Plaintext<Scheme> pt(context);
        encoder.encode(pt, diag_values[k]);
        cudaMemcpy(concatenated.data() + k * N, pt.data(), N * sizeof(Data64),
                   cudaMemcpyDeviceToDevice);
    }
    std::vector<heongpu::DeviceVector<Data64>> encoded_matrices;
    encoded_matrices.push_back(std::move(concatenated));

    heongpu::Ciphertext<Scheme> C_result =
        operators.multiply_matrix(C1, encoded_matrices, diags, galois_key);
    std::cout << "Noise budget after multiply_matrix: "
              << decryptor.remainder_noise_budget(C_result) << " bits"
              << std::endl;

    heongpu::Plaintext<Scheme> P_result(context);
    decryptor.decrypt(P_result, C_result);
    std::vector<uint64_t> result;
    encoder.decode(result, P_result);

    // Plaintext reference.
    std::vector<uint64_t> expected(N, 0);
    for (size_t i = 0; i < N; ++i)
    {
        const int row = static_cast<int>(i) / H * H;
        const int col = static_cast<int>(i) % H;
        unsigned __int128 acc = 0;
        for (size_t k = 0; k < D; ++k)
        {
            const int src = row + (((col + diags[0][k]) % H) + H) % H;
            acc += static_cast<unsigned __int128>(diag_values[k][i]) *
                   message[src];
        }
        expected[i] = static_cast<uint64_t>(acc % t);
    }

    size_t bad = 0;
    for (size_t i = 0; i < N; ++i)
        bad += (result[i] != expected[i]);

    std::cout << "\nExpected (plaintext product):";
    display_matrix(expected, H);
    std::cout << "Decrypted multiply_matrix result:";
    display_matrix(result, H);

    const bool ok = (bad == 0);
    std::cout << (ok ? "[PASS] " : "[FAIL] ")
              << "multiply_matrix == plaintext product (" << N - bad << "/"
              << N << " slots match)" << std::endl;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
