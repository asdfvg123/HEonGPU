
#include "heongpu.cuh"
#include "../example_util.h"
#include "util.cuh"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cmath> // For std::pow

typedef unsigned long long Data64;
constexpr auto Scheme = heongpu::Scheme::RTF;
constexpr auto SchemeCKKS = heongpu::Scheme::CKKS;

static void get_coeffs(std::vector<uint64_t>& coeffs, heongpu::Plaintext<Scheme>& pt) {
    auto* dptr = pt.data();
    cudaMemcpyAsync(
        coeffs.data(), dptr, pt.size() * sizeof(uint64_t),
        cudaMemcpyDeviceToHost, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);
}

static void get_coeffs(std::vector<double>& coeffs, heongpu::Plaintext<SchemeCKKS>& pt) {
    auto* dptr = pt.data();
    cudaMemcpyAsync(
        coeffs.data(), dptr, pt.size() * sizeof(double),
        cudaMemcpyDeviceToHost, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);
}

static void print_first(const char* tag, const std::vector<uint64_t>& v, size_t k) {
    std::cout << tag << " [0:" << k << "): ";
    for (size_t i = 0; i < k; ++i) std::cout << v[i] << (i+1<k ? ' ' : '\n');
}
static void print_first(const char* tag, const std::vector<double>& v, size_t k) {
    std::cout << tag << " [0:" << k << "): ";
    for (size_t i = 0; i < k; ++i) std::cout << v[i] << (i+1<k ? ' ' : '\n');
}


int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    
    // =========================================================================
    // BFV (RTF) Context and Key Generation
    // =========================================================================
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        heongpu::sec_level_type::none
    );

    size_t poly_modulus_degree = 4096;
    context.set_poly_modulus_degree(poly_modulus_degree);

    std::vector<int> logQ = {
        60, 44, 44, 44, 44, 
        44, 44, 44, 44, 44,
        44, 44, 44, 44, 44,
        44, 44, 44, 44, 44,
        44, 44, 44, 44, 44,
        44, 44, 44, 44, 44};
    std::vector<int> logP = {60, 60, 60};
    context.set_coeff_modulus_bit_sizes(logQ, logP);
    double scale = std::pow(2.0, 44);
    double delta = static_cast<double>(64);
    int plain_modulus = 65537;
    context.set_plain_modulus(plain_modulus);

    context.generate();
    context.print_parameters();

    // =========================================================================
    // CKKS Context Generation
    // =========================================================================
    heongpu::HEContext<SchemeCKKS> contextckks(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        heongpu::sec_level_type::none);
    contextckks.set_poly_modulus_degree(poly_modulus_degree);
    contextckks.set_coeff_modulus_bit_sizes(logQ, logP);
    contextckks.generate();
    std::cout << "\nCKKS Context Generated.\n" << std::endl;

    // =========================================================================
    // Key Generation (Shared for BFV and CKKS)
    // =========================================================================
    heongpu::HEKeyGenerator<Scheme> keygen(context);

    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    std::cout << "Generating Galois keys for S2C_FV function..." << std::endl;
    const int N = static_cast<int>(poly_modulus_degree);
    const int g2 = static_cast<int>(std::ceil(std::sqrt((double)N/2)));
    std::set<int> required_shifts;
    for(int v = 0; v < g2; v++) required_shifts.insert(v);
    required_shifts.insert(-1);
    for (int s = 0; s < N; ++s) {
        int j = (s % g2 + g2) % g2;
        int i = (s - j) / g2;
        long long giant_rot_amount = static_cast<long long>(i) * g2;
        if (giant_rot_amount == 0) continue; 
        const int N_div_2 = N >> 1;
        long long effective_rot = giant_rot_amount % N;
        if (effective_rot < 0) effective_rot += N;
        long long row_rotation_amount = effective_rot % N_div_2;
        if (row_rotation_amount != 0) required_shifts.insert(static_cast<int>(row_rotation_amount));
    }
    std::vector<int> all_required(required_shifts.begin(), required_shifts.end());
    heongpu::Galoiskey<Scheme> galois_key(context, all_required);
    keygen.generate_galois_key(galois_key, secret_key);
    std::cout << "Galois keys generated." << std::endl;

    // =========================================================================
    // Setup HE Operators
    // =========================================================================
    heongpu::HEEncoder<Scheme>    encoder(context);
    heongpu::HEEncryptor<Scheme>  encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme>  decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);
    heongpu::HEHERA<Scheme> hera(context, contextckks, encoder, encryptor, op, galois_key, relin_key);



    // =========================================================================
    // BFV Operations
    // =========================================================================
    std::vector<uint64_t> message(N, 0ULL);
    for (int i = 0; i < N; ++i) message[i] = static_cast<uint64_t>(((i+1) % 16)*delta);
    print_first("(message)", message, 32);

    heongpu::Plaintext<Scheme> pt_in(context);
    encoder.encode(pt_in, message);
    
    heongpu::Ciphertext<Scheme> ct_in(context);
    encryptor.encrypt(ct_in, pt_in); // slot : [0, 1, 2, 3, ...], coeff : [bigInt, ...,]


    hera.gen_FV_S2C_Matrix();
    auto ct_s2c = hera.S2C_FV(ct_in);


    std::vector<uint64_t> message_client(N, 0ULL);
    for (int i = 0; i < N; ++i) message_client[i] = static_cast<uint64_t>((((i+1) % 16)+ 1)*delta);
    print_first("(message_client)", message_client, 32);
    auto pt_ske = hera.vec2poly(message_client);

    auto ct_original_message = hera.eval_decrypt(ct_s2c, pt_ske); // message_client - message
    heongpu::Plaintext<Scheme> pt_out(context);
    decryptor.decrypt(pt_out, ct_original_message);

    std::vector<uint64_t> coeffs(N);
    get_coeffs(coeffs, pt_out);
    print_first("(decrypted FV coeff = message_client - message)", coeffs, 32);


    // auto ct_moddown = hera.moddown_FV_inplace(ct_in);
    
    // =========================================================================
    // Transciphering and CKKS Decryption
    // =========================================================================
    // In this example, the message is integers, so the client-side scaling
    // factor 'delta' is 1.0.
    std::cout << "\nTransciphering from BFV ciphertext to CKKS ciphertext..." << std::endl;
    auto ct_ckks = hera.transcipher_bfv2ckks(ct_original_message); // coeff : [bigInt, ...,] reduced scale
    std::cout << "scale : " << ct_ckks.scale() << std::endl;
    std::cout << "scale I set : " << scale << std::endl;
    
    heongpu::Secretkey<SchemeCKKS> secret_key_ckks(secret_key);
    heongpu::HEDecryptor<SchemeCKKS> decryptor_ckks(contextckks, secret_key_ckks);
    heongpu::HEEncoder<SchemeCKKS>   encoder_ckks(contextckks);
    
    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygenckks(contextckks);
    heongpu::Relinkey<heongpu::Scheme::CKKS> relin_keyckks(contextckks);
    keygenckks.generate_relin_key(relin_keyckks, secret_key_ckks);
    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoderckks(contextckks);
    int StoC_piece = 3;
    heongpu::BootstrappingConfig boot_config(3, StoC_piece, 15, true);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operatorsckks(contextckks, encoderckks);
    

    operatorsckks.generate_bootstrapping_params(scale, boot_config);
    std::vector<int> key_index = operatorsckks.bootstrapping_key_indexs();
    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_keyckks(contextckks, key_index);
    keygenckks.generate_galois_key(galois_keyckks, secret_key_ckks);
    
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> cipher_boot =
        operatorsckks.regular_halfbootstrapping(ct_ckks, galois_keyckks, relin_keyckks); // slot [0, 0, 0]
    std::cout << "cipher_boot scale : " << cipher_boot.scale() << std::endl;

    heongpu::Plaintext<SchemeCKKS> pt_ckks_out(contextckks);
    decryptor_ckks.decrypt(pt_ckks_out, cipher_boot); // coeff should have something.

    // std::vector<double> decrypted_result(pt_ckks_out.size());
    // get_coeffs(decrypted_result, pt_ckks_out);
    // print_first("(decrypted CKKS)", decrypted_result, 32);

    
    std::vector<Complex64> decrypted_result;
    encoder_ckks.decode(decrypted_result, pt_ckks_out); 

    std::cout << "\nFinal result after transciphering and decryption:" << std::endl;
    display_vector(decrypted_result);

    return EXIT_SUCCESS;
}
