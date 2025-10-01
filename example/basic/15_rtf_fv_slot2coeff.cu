
#include "heongpu.cuh"
#include "../example_util.h"
#include "util.cuh"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cassert>

typedef unsigned long long Data64;
constexpr auto Scheme = heongpu::Scheme::RTF;
constexpr auto SchemeCKKS = heongpu::Scheme::CKKS;

static void print_first(const char* tag, const std::vector<uint64_t>& v, size_t k) {
    std::cout << tag << " [0:" << k << "): ";
    for (size_t i = 0; i < k; ++i) std::cout << v[i] << (i+1<k ? ' ' : '\n');
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    cudaStream_t stream = cudaStreamPerThread;

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        heongpu::sec_level_type::none
    );

    size_t poly_modulus_degree = 16384;
    context.set_poly_modulus_degree(poly_modulus_degree);

    heongpu::HEContext<SchemeCKKS> contextckks(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    contextckks.set_poly_modulus_degree(poly_modulus_degree);
    contextckks.set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    contextckks.generate();

    // context.set_coeff_modulus_default_values(1);
    std::vector<int> logQ = {58, 58, 58, 58, 59};
    std::vector<int> logP = {59};
    context.set_coeff_modulus_bit_sizes(logQ, logP);

    // int plain_modulus = 786433;               
    int plain_modulus = 65537;
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

        if (giant_rot_amount == 0) {
            continue; 
        }

        const int N_div_2 = N >> 1;
        long long effective_rot = giant_rot_amount % N;
        if (effective_rot < 0) {
            effective_rot += N;
        }

        long long row_rotation_amount = effective_rot % N_div_2;

        if (row_rotation_amount != 0) {
            required_shifts.insert(static_cast<int>(row_rotation_amount));
        }
    }

    std::vector<int> all_required(required_shifts.begin(), required_shifts.end());
    
    heongpu::Galoiskey<Scheme> galois_key(context, all_required);
    keygen.generate_galois_key(galois_key, secret_key);

    heongpu::HEEncoder<Scheme>    encoder(context);
    heongpu::HEEncryptor<Scheme>  encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme>  decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);
    heongpu::HEHERA<Scheme> hera(context, contextckks, encoder, encryptor, op, galois_key, relin_key);


    auto psi = hera.get_psi();

    {
    // int m = 2 * N;
    // int gen = 3;

    // uint64_t pos = 1;
    // uint64_t t = static_cast<uint64_t>(plain_modulus);

    // for (int i = 0; i < N / 2; ++i) {
    //     uint64_t psi_pow_pos = hera.pow_mod_u64(psi, pos, t);
    //     uint64_t denom_inv = modInverse(psi_pow_pos - 1, t);
    //     message[i] = hera.mul_mod_u128(t-2, denom_inv, t);

    //     pos = hera.mul_mod_u128(pos, gen, m);
    // }

    // for (int i = N / 2; i < N; ++i) {
    //     // conjugate position, m - pos
    //     uint64_t conjugate_pos = m - pos;

    //     uint64_t psi_pow_conjugate_pos = hera.pow_mod_u64(psi, conjugate_pos, t);
    //     uint64_t denom_inv = modInverse(psi_pow_conjugate_pos - 1, t);
    //     message[i] = hera.mul_mod_u128(t - 2, denom_inv, t);

    //     pos = hera.mul_mod_u128(pos, gen, m);
    // }
    }

    // for (int i = 16; i < 32; ++i) message[i]     = static_cast<uint64_t>(i + 1 - 16);
    
    std::vector<uint64_t> message(N, 0ULL);
    for (int i = 0; i < N; ++i) message[i]      = static_cast<uint64_t>((i+1) % 33);
    // message[0] = 1;
    print_first("slots (expected coefficients after S2C)", message, 32);



    heongpu::Plaintext<Scheme> pt(context);
    encoder.encode(pt, message);


    std::vector<uint64_t> coeffs_pt(N);

    pt.store_in_device(stream);
    auto* dptr_pt = pt.data();
    cudaMemcpyAsync(
        coeffs_pt.data(), dptr_pt, pt.size() * sizeof(uint64_t),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    
    // print_first("slots Before (first 64)", message, N);
    print_first("coefficients Before (first 64)", coeffs_pt, 64);
    heongpu::Ciphertext<Scheme> ct(context);
    encryptor.encrypt(ct, pt);

    heongpu::ExecutionOptions opt;
    opt.set_stream(stream);

    auto n_power_hera = hera.get_n_power();
    std::cout << "[INFO] RTF polynomial degree N: " << n_power_hera << "\n";
    std::cout << "[INFO] RTF primitive root of unity (psi): " << psi << "\n";



    // 5) Slot -> Coeff under encryption
    //    ct_s2c should decrypt to coefficients == original slots
    // ----------------------------
    hera.gen_FV_S2C_Matrix();
    heongpu::Ciphertext<Scheme> ct_s2c = hera.S2C_FV(ct, opt);

    std::cout << "Noise budget in ct_s2c: "
              << decryptor.remainder_noise_budget(ct_s2c) << " bits"
              << std::endl;



    heongpu::Plaintext<Scheme> pt_out(context);
    decryptor.decrypt(pt_out, ct_s2c);
    std::vector<uint64_t> vec_result_bsgs;
    encoder.decode(vec_result_bsgs, pt_out);

    std::vector<uint64_t> coeffs(N);
    pt_out.store_in_device(stream);

    auto* dptr = pt_out.data();
    cudaMemcpyAsync(
        coeffs.data(), dptr, pt_out.size() * sizeof(uint64_t),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    print_first("coefficients (first 64)", coeffs, 64);

    return EXIT_SUCCESS;
}
