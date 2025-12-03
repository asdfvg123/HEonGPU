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
constexpr auto SchemeCKKS = heongpu::Scheme::CKKS;
int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    // --- Context / params ---
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I, 
        heongpu::sec_level_type::none);

    size_t poly_modulus_degree = 4096; // divisible by 16
    context.set_poly_modulus_degree(poly_modulus_degree);
    std::vector<int> logQ = {
        60, 50, 50, 50, 50, 
        50, 50, 50, 50, 50};
    std::vector<int> logP = {60}; 
    context.set_coeff_modulus_bit_sizes(logQ, logP);
    // context.set_coeff_modulus_default_values(1);
 
    // int plain_modulus = 786433; // prime t
    int plain_modulus = 0x1fc0001ULL;

    context.set_plain_modulus(plain_modulus);

    context.generate();
    context.print_parameters();

    heongpu::HEContext<SchemeCKKS> contextckks(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I, 
        heongpu::sec_level_type::none);
    contextckks.set_poly_modulus_degree(poly_modulus_degree);
    contextckks.set_coeff_modulus_bit_sizes(logQ, logP);
    contextckks.generate(); 

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

    std::vector<int> shifts = {1, 2, 4, 8, 16, 32, -1, -2, -4, -8, -16, -32};
    // example heongpu::Galoiskey galois_key(context, shifts);
    heongpu::ExecutionOptions opt_device;
    opt_device.set_storage_type(heongpu::storage_type::DEVICE);
    heongpu::Galoiskey<Scheme> galois_key(context, shifts);
    keygen.generate_galois_key(galois_key, secret_key, opt_device);

    
    // Alternative way 1 -> calculate dedicated shift value:
    // std::vector<int> shifts = {0, -1, 128, -16, -32, -48, -64, -80, -96,
    // -112}; // example heongpu::Galoiskey galois_key(context, shifts);
    // keygen.generate_galois_key(galois_key, secret_key);


    // Alternative way 2 -> calculate dedicated galois value:
    // std::vector<uint32_t> galois = {0, -1, 128, -16, -32, -48, -64, -80, -96,
    // -112}; // example heongpu::Galoiskey galois_key(context, galois);
    // keygen.generate_galois_key(galois_key, secret_key);
    // use apply_galois instead of rotate_row!


    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    std::vector<uint64_t> message(poly_modulus_degree, 0ULL);
    for (int i = 0; i < 64; ++i) message[i] = static_cast<uint64_t>(i + 1);
    for (int i = 64; i < 128; ++i) message[i] = static_cast<uint64_t>(i + 1 - 64);

    std::vector<uint64_t> keyhera(poly_modulus_degree, 0ULL);
    for (int i = 0; i < 64; ++i) keyhera[i] = static_cast<uint64_t>(i + 1);
    for (int i = 64; i < 128; ++i) keyhera[i] = static_cast<uint64_t>(i + 1 - 64);

    std::cout << "[Input] First 128 entries:\n";
    for(int i = 0; i < 128; ++i) {
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
        context, contextckks, encoder, encryptor, operators, galois_key, relin_key);
    
    hera.set_galois_key_hera(galois_key);
    auto icCt_ = hera.get_icCt();
    heongpu::Plaintext<Scheme> P_ic(context);
    decryptor.decrypt(P_ic, icCt_);
    
    std::vector<uint64_t> vec_ic;
    encoder.decode(vec_ic, P_ic);
    
    // std::cout << "\n[HERA Initial Condition] First 64 entries:\n";
    // for(int i = 0; i < 64; ++i) {
    //     std::cout << vec_ic[i] << " ";
    //     if (i % 16 == 15) std::cout << "\n";
    // }
    // std::cout << std::endl;

    // auto rcVec = hera.get_rcVec();
    // std::cout << "\n[HERA Random Coefficients] First 16 entries of round 0:\n";
    // for(int i = 0; i < 16; ++i) {
    //     std::cout << rcVec[0][i] << " ";
    //     if (i % 16 == 15) std::cout << "\n";
    // }   
    // --- Run GPU linear (MixRows -> MixColumns) ---
    std::cout << "\nRunning hera ...\n";

    // heongpu::DeviceVector<Data64> nonce;
    auto C_lin = hera.gen_stream_key(C_keyhera);
    cudaDeviceSynchronize();

    // Decrypt & Decode
    heongpu::Plaintext<Scheme> P_lin(context);
    decryptor.decrypt(P_lin, C_lin);

    std::vector<uint64_t> vec_lin;
    encoder.decode(vec_lin, P_lin);

    std::cout << "\n[Output after linear()] First 64 entries:\n";
    for(int i = 0; i < 256; ++i) {
        std::cout << vec_lin[i] << " ";
        if (i % 16 == 15) std::cout << "\n";
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
