// Copyright 2025
// Demo: run HEOperator<Scheme::RTF>::linear() and print the result only.

#include "heongpu.cuh"
#include "../example_util.h"   // display_matrix(...)

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

typedef unsigned long long Data64;
constexpr auto Scheme = heongpu::Scheme::RTF;

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    // --- Context / params ---
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    size_t poly_modulus_degree = 4096; // divisible by 16
    context.set_poly_modulus_degree(poly_modulus_degree);

    context.set_coeff_modulus_default_values(1);

    int plain_modulus = 786433; // prime t
    // int plain_modulus = 0x1fc0001ULL;

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

    // keygen.generate_galois_key(
        // galois_key, secret_key); // This way will create 16(2x8) different power
        // of 2, if you need more change from define.h
        
        // Alternative way 1 -> calculate dedicated shift value:
    // const int row_len = static_cast<int>(poly_modulus_degree) >> 1; // 2048

    // std::vector<int> shifts = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        //  -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15}; // example
    // example heongpu::Galoiskey galois_key(context, shifts);
    
    heongpu::Galoiskey<Scheme> galois_key(context);
    keygen.generate_galois_key(galois_key, secret_key);


    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    // --- Test message: first 16 slots = 1..16, others 0 ---
    std::vector<uint64_t> message(poly_modulus_degree, 0ULL);
    for (int i = 0; i < 16; ++i) message[i] = static_cast<uint64_t>(i + 1);
    for (int i = 16; i < 32; ++i) message[i] = static_cast<uint64_t>(i + 1 - 16);

    std::vector<uint64_t> keyhera(poly_modulus_degree, 0ULL);
    for (int i = 0; i < 16; ++i) keyhera[i] = static_cast<uint64_t>(i + 1);
    for (int i = 16; i < 32; ++i) keyhera[i] = static_cast<uint64_t>(i + 1 - 16);

    std::cout << "[Input] First 64 entries:\n";
    for(int i = 0; i < 64; ++i) {
        std::cout << message[i] << " ";
        if (i % 16 == 15) std::cout << "\n";
    }

    // Encode / Encrypt
    heongpu::Plaintext<Scheme> P(context);
    encoder.encode(P, message);

    heongpu::Ciphertext<Scheme> C(context);
    encryptor.encrypt(C, P);

    heongpu::Plaintext<Scheme> P_keyhera(context);
    encoder.encode(P_keyhera, keyhera);
    heongpu::Ciphertext<Scheme> C_keyhera(context);
    encryptor.encrypt(C_keyhera, P_keyhera);

    heongpu::HEHERA<Scheme> hera(
        context, encoder, encryptor, operators, galois_key, relin_key);

    auto plain_psi = hera.get_plain_psi();
    std::cout << "plain_psi: " << plain_psi << std::endl;
    hera.print_ntt_table_host(16);

    return EXIT_SUCCESS;
}
