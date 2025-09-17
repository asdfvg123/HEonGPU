// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include "../example_util.h"
#include <omp.h>

// These examples have been developed with reference to the Microsoft SEAL
// library.

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::RTF;
int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    size_t poly_modulus_degree = 4096;
    context.set_poly_modulus_degree(poly_modulus_degree);

    context.set_coeff_modulus_default_values(1);

    int plain_modulus = 786433;
    context.set_plain_modulus(plain_modulus);

    context.generate();
    context.print_parameters();

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    heongpu::Galoiskey<Scheme> galois_key(context);
    keygen.generate_galois_key(galois_key, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    std::vector<uint64_t> message(poly_modulus_degree, 0ULL);
    message[0] = 1ULL;
    message[1] = 2ULL;
    message[2] = 3ULL;
    message[3] = 4ULL;

    display_matrix(message, 0);

    heongpu::Plaintext<Scheme> P1(context);
    encoder.encode(P1, message);

    std::cout << "Encrypting plaintext matrix..." << std::endl;
    heongpu::Ciphertext<Scheme> C1(context);
    encryptor.encrypt(C1, P1);

    std::cout << "Initial noise budget in C1: "
              << decryptor.remainder_noise_budget(C1) << " bits"
              << std::endl;

    /********************************************************************************
     * MATRIX MULTIPLICATION EXAMPLE
     ********************************************************************************/
    std::cout << "\n--- Starting Homomorphic Matrix Multiplication ---" << std::endl;

    std::vector<std::vector<int>> diags = {{0, 2, -2}};

    std::cout << "Preparing encoded diagonals of the matrix..." << std::endl;
    std::vector<uint64_t> diag0_vec(poly_modulus_degree, 3ULL);
    std::vector<uint64_t> diag2_vec(poly_modulus_degree, 1ULL);
    std::vector<uint64_t> diag_minus2_vec(poly_modulus_degree, 5ULL);

    heongpu::Plaintext<Scheme> plain_diag0(context), plain_diag2(context), plain_diag_minus2(context);
    encoder.encode(plain_diag0, diag0_vec);
    encoder.encode(plain_diag2, diag2_vec);
    encoder.encode(plain_diag_minus2, diag_minus2_vec);
    
    // 3. 인코딩된 대각선들을 하나의 DeviceVector에 순서대로 이어붙입니다.
    size_t num_diagonals = diags[0].size();
    
    // Plaintext::size()는 요소(element)의 개수를 반환합니다.
    size_t plain_size_elements = plain_diag0.size();
    // 요소의 개수에 요소 하나의 크기(sizeof(uint64_t))를 곱하여 byte 단위 크기를 계산합니다.
    size_t plain_size_bytes = plain_size_elements * sizeof(Data64);
    
    size_t total_elements = num_diagonals * plain_size_elements;
    heongpu::DeviceVector<Data64> concatenated_diagonals(total_elements);

    // 계산된 byte 크기를 사용하여 각 대각선 데이터를 GPU의 연속된 메모리 공간으로 복사합니다.
    cudaMemcpy(concatenated_diagonals.data(), plain_diag0.data(), plain_size_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(concatenated_diagonals.data() + plain_size_elements, plain_diag2.data(), plain_size_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(concatenated_diagonals.data() + (2 * plain_size_elements), plain_diag_minus2.data(), plain_size_bytes, cudaMemcpyDeviceToDevice);

    std::vector<heongpu::DeviceVector<Data64>> encoded_matrices;
    encoded_matrices.push_back(std::move(concatenated_diagonals));

    std::cout << "Calling multiply_matrix function..." << std::endl;
    heongpu::Ciphertext<Scheme> C_result = operators.multiply_matrix(
        C1, encoded_matrices, diags, galois_key);

    std::cout << "Decrypting and decoding the result..." << std::endl;
    heongpu::Plaintext<Scheme> P_result(context);
    decryptor.decrypt(P_result, C_result);

    std::vector<uint64_t> vec_result;
    encoder.decode(vec_result, P_result);

    std::cout << "\nResult matrix after multiplication:" << std::endl;
    display_matrix(vec_result, 0);

    std::cout << "\n--- Matrix Multiplication Example Finished ---" << std::endl;

    return EXIT_SUCCESS;
}