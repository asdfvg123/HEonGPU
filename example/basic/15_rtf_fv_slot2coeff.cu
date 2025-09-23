// Copyright 2025
// Demo: Verify Slot -> Coeff (factorized forward NTT under encryption)
// Requires HEHERA::gen_FV_S2C_Matrix(...) and HEHERA::slot_to_coeff_forward_ntt(...)

#include "heongpu.cuh"
#include "../example_util.h"   // display_matrix(...), optional helpers
#include "util.cuh"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cassert>

typedef unsigned long long Data64;
constexpr auto Scheme = heongpu::Scheme::RTF;

uint64_t modInverse(int64_t a, uint64_t m) {
    // a가 음수일 경우 양수로 변환
    a = (a % m + m) % m;

    int64_t m0 = m;
    int64_t y = 0, x = 1;

    if (m == 1) return 0;

    while (a > 1) {
        int64_t q = a / m0;
        int64_t t = m0;

        m0 = a % m0;
        a = t;
        t = y;

        y = x - q * y;
        x = t;
    }

    if (x < 0) x += m;

    return x;
}
static void print_first(const char* tag, const std::vector<uint64_t>& v, size_t k) {
    std::cout << tag << " [0:" << k << "): ";
    for (size_t i = 0; i < k; ++i) std::cout << v[i] << (i+1<k ? ' ' : '\n');
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    cudaStream_t stream = cudaStreamPerThread; // or cudaStreamDefault

    // ----------------------------
    // 1) Context / keys
    // ----------------------------
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        heongpu::sec_level_type::none
    );

    size_t poly_modulus_degree = 8192;         // must be power of two
    context.set_poly_modulus_degree(poly_modulus_degree);

    // context.set_coeff_modulus_default_values(1);
    std::vector<int> logQ = {58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
        59, 59};
    std::vector<int> logP = {59, 59};
    context.set_coeff_modulus_bit_sizes(logQ, logP);

    // t must satisfy t ≡ 1 (mod 2N) for full batching
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
    // --- CORRECTED GALOIS KEY GENERATION FOR S2C_FV ---
    std::cout << "Generating Galois keys for S2C_FV function..." << std::endl;

    const int N = static_cast<int>(poly_modulus_degree);
    // S2C_FV's BSGS implementation uses g2 = ceil(sqrt(N))
    const int g2 = static_cast<int>(std::ceil(std::sqrt((double)N/2)));
    
    std::cout << "BSGS parameters for S2C_FV: N=" << N << ", g2=" << g2 << std::endl;

    // Use a set to automatically handle duplicate rotation steps
    std::set<int> required_shifts;

    // 1. Key for baby steps: always need to rotate by +1 repeatedly.
    for(int v = 0; v < g2; v++) required_shifts.insert(v);

    // 2. Key for column rotations: S2C_FV may use rotate_columns.
    // In many libraries, this is requested with a step of -1.
    required_shifts.insert(-1);
    
    // 3. Predict all giant step rotations needed by the S2C function.
    // The S2C function will use all diagonals from s = 0 to N-1.
    for (int s = 0; s < N; ++s) {
        // Decompose the shift 's' using the same logic as in multiply_matrix_bsgs
        int j = (s % g2 + g2) % g2;
        int i = (s - j) / g2;
        long long giant_rot_amount = static_cast<long long>(i) * g2;

        if (giant_rot_amount == 0) {
            continue; // No giant rotation needed for this 's'
        }

        // Now, decompose the giant rotation into elementary row/column rotations
        // to find the actual key we need to generate. This logic must exactly
        // match the decomposition inside multiply_matrix_bsgs.
        const int N_div_2 = N >> 1;
        long long effective_rot = giant_rot_amount % N;
        if (effective_rot < 0) {
            effective_rot += N;
        }

        long long row_rotation_amount = effective_rot % N_div_2;

        if (row_rotation_amount != 0) {
            // We need a key for this specific small row rotation.
            required_shifts.insert(static_cast<int>(row_rotation_amount));
        }
    }

    // Convert the set of required shifts to a vector for the constructor
    std::vector<int> all_required(required_shifts.begin(), required_shifts.end());
    
    std::cout << "Final required elementary rotation keys: ";
    for(int step : all_required) std::cout << step << " ";
    std::cout << std::endl;

    // Build the Galois keys with all the necessary rotation steps
    heongpu::Galoiskey<Scheme> galois_key(context, all_required);
    keygen.generate_galois_key(galois_key, secret_key);

    heongpu::HEEncoder<Scheme>    encoder(context);
    heongpu::HEEncryptor<Scheme>  encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme>  decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);   // if your type is HEArithmeticOperator, just switch the name
    heongpu::HEHERA<Scheme> hera(context, encoder, encryptor, op, galois_key, relin_key);

    // ----------------------------
    // 2) Test message in slots
    // ----------------------------
    auto psi = hera.get_psi();

    {
    // int m = 2 * N;
    // int gen = 3;

    // uint64_t pos = 1;
    // uint64_t t = static_cast<uint64_t>(plain_modulus);

    // // HEEncoder 생성자와 동일한 루프
    // for (int i = 0; i < N / 2; ++i) {
    //     // psi^pos 계산
    //     uint64_t psi_pow_pos = hera.pow_mod_u64(psi, pos, t);
    //     // (psi^pos - 1)의 모듈러 역원 계산
    //     uint64_t denom_inv = modInverse(psi_pow_pos - 1, t);
    //     // message[i] 값 계산
    //     message[i] = hera.mul_mod_u128(t-2, denom_inv, t);

    //     pos = hera.mul_mod_u128(pos, gen, m);
    // }

    // for (int i = N / 2; i < N; ++i) {
    //     // 켤레 위치(conjugate position)인 m - pos를 지수로 사용합니다.
    //     uint64_t conjugate_pos = m - pos;

    //     // psi^(m - pos)를 계산합니다.
    //     uint64_t psi_pow_conjugate_pos = hera.pow_mod_u64(psi, conjugate_pos, t);
    //     uint64_t denom_inv = modInverse(psi_pow_conjugate_pos - 1, t);
    //     message[i] = hera.mul_mod_u128(t - 2, denom_inv, t);

    //     // pos 값은 원래대로 계속 갱신합니다.
    //     pos = hera.mul_mod_u128(pos, gen, m);
    // }
    }


    // Example pattern: first 16 slots = 1..16, next 16 slots = 1..16, rest 0
    // for (int i = 16; i < 32; ++i) message[i]     = static_cast<uint64_t>(i + 1 - 16);
    
    std::vector<uint64_t> message(N, 0ULL);
    for (int i = 0; i < N; ++i) message[i]      = static_cast<uint64_t>(i+1);
    // message[0] = 1;
    print_first("slots (expected coefficients after S2C)", message, 32);

    // ----------------------------
    // 3) Encode & encrypt
    // ----------------------------
    heongpu::Plaintext<Scheme> pt(context);
    encoder.encode(pt, message);     // produces coefficient-domain plaintext (iNTT inside encoder)


    std::vector<uint64_t> coeffs_pt(N);

    // Ensure data is on device, then copy out
    pt.store_in_device(stream);
    auto* dptr_pt = pt.data();               // device pointer
    cudaMemcpyAsync(
        coeffs_pt.data(), dptr_pt, pt.size() * sizeof(uint64_t),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    
    // print_first("slots Before (first 64)", message, N);
    print_first("coefficients Before (first 64)", coeffs_pt, 64);
    heongpu::Ciphertext<Scheme> ct(context);
    encryptor.encrypt(ct, pt);

    // ----------------------------
    // 4) Prepare S2C (factorized forward NTT) matrices
    // ----------------------------
    heongpu::ExecutionOptions opt;
    opt.set_stream(stream);

    auto n_power_hera = hera.get_n_power();
    std::cout << "[INFO] RTF polynomial degree N: " << n_power_hera << "\n";
    std::cout << "[INFO] RTF primitive root of unity (psi): " << psi << "\n";

    hera.gen_FV_S2C_Matrix(opt);
    hera.print_FV_S2C_Matrix();
    // hera.print_s2c_matrix_diagonal(66);

    // hera.print_s2c_matrix_diagonal(65);
    // hera.print_s2c_matrix_diagonal(64);
    // hera.print_s2c_matrix_diagonal(63);


    // hera.print_s2c_matrix_shifts();


    // hera.print_s2c_matrix_diagonals();
    // hera.print_s2c_matrix_shifts();

    // ----------------------------
    // 5) Slot -> Coeff under encryption
    //    ct_s2c should decrypt to coefficients == original slots
    // ----------------------------
    heongpu::Ciphertext<Scheme> ct_s2c = hera.S2C_FV(ct, opt);

    std::cout << "Noise budget in ct_s2c: "
              << decryptor.remainder_noise_budget(ct_s2c) << " bits"
              << std::endl;

    // ----------------------------
    // 6) Decrypt and copy coefficients to host
    // ----------------------------
    heongpu::Plaintext<Scheme> pt_out(context);
    decryptor.decrypt(pt_out, ct_s2c);
    std::vector<uint64_t> vec_result_bsgs;
    encoder.decode(vec_result_bsgs, pt_out);

    std::vector<uint64_t> coeffs(N);

    // Ensure data is on device, then copy out
    pt_out.store_in_device(stream);
    auto* dptr = pt_out.data();               // device pointer
    cudaMemcpyAsync(
        coeffs.data(), dptr, pt_out.size() * sizeof(uint64_t),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    print_first("coefficients (first 64)", coeffs, N);
    // print_first("slots (first 64)", vec_result_bsgs, N);

    // print_first("slots (first 64)", vec_result_bsgs, N);
    // coeffs = decrypted vector from ct_s2c (length n)


    // auto bitrev = [&](int i) {
    //     unsigned r = 0, v = static_cast<unsigned>(i);
    //     for (int b = 0; b < n_power; ++b) { r = (r << 1) | (v & 1); v >>= 1; }
    //     return static_cast<int>(r);
    // };

    // std::vector<uint64_t> coeffs_natural(n);
    // for (int i = 0; i < n; ++i) {
    //     coeffs_natural[i] = coeffs[ bitrev(i) ]; // undo DIF bit-reversal
    // }

    // // Now compare coeffs_natural with your original slot vector 'message'
    // size_t mismatches = 0;
    // for (int i = 0; i < N; ++i) {
    //     if ((coeffs_natural[i] % plain_modulus) != (message[i] % plain_modulus)) {
    //         if (mismatches < 8) std::cerr << "mismatch @"<<i<<": got "
    //             << coeffs_natural[i] << " expected " << message[i] << "\n";
    //         ++mismatches;
    //     }
    // }
    // std::cout << (mismatches ? "[WARN] " : "[OK] ")
    //         << mismatches << " mismatches in first 32 positions.\n";
    

    return EXIT_SUCCESS;
}
