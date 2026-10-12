// Latency, precision and throughput of (v1) slim bootstrapping.
//
//   ./7_ckks_slim_bootstrapping_bench [logN] [hamming_weight] [taylor] [reps]
//   (defaults: 16 16 7 5)
//
// Same chain as 2_ckks_slim_bootstrapping (Q = 60 + 28 x 50, P = 3 x 60,
// KEYSWITCHING_METHOD_II, scale 2^50), random real messages in (-1, 1).

#include "heongpu.cuh"
#include "../example_util.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    const int logN = (argc > 1) ? std::atoi(argv[1]) : 16;
    const int h = (argc > 2) ? std::atoi(argv[2]) : 16;
    const int taylor = (argc > 3) ? std::atoi(argv[3]) : 7;
    const int reps = (argc > 4) ? std::atoi(argv[4]) : 5;
    const size_t N = size_t(1) << logN;

    heongpu::HEContext<heongpu::Scheme::CKKS> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(N);
    std::vector<int> logQ(29, 50);
    logQ[0] = 60;
    context.set_coeff_modulus_bit_sizes(logQ, {60, 60, 60});
    context.generate();
    const double scale = std::pow(2.0, 50);

    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context, h);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
    keygen.generate_public_key(public_key, secret_key);
    heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context, public_key);
    heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(context,
                                                                   encoder);

    const int slot_count = N / 2;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::vector<double> message(slot_count);
    for (auto& m : message)
        m = unif(rng);
    heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
    encoder.encode(P1, message, scale);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
    encryptor.encrypt(C1, P1);

    const int StoC_piece = 3;
    heongpu::BootstrappingConfig boot_config(3, StoC_piece, taylor, true);
    operators.generate_bootstrapping_params(
        scale, boot_config,
        heongpu::arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING);
    std::vector<int> key_index = operators.bootstrapping_key_indexs();
    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(context, key_index);
    keygen.generate_galois_key(galois_key, secret_key);

    for (int i = 0; i < (28 - StoC_piece); i++)
        operators.mod_drop_inplace(C1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> out;
    float total_ms = 0;
    for (int r = 0; r <= reps; ++r) // r = 0 is the warm-up
    {
        heongpu::Ciphertext<heongpu::Scheme::CKKS> in = C1;
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        out = operators.slim_bootstrapping(in, galois_key, relin_key);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        if (r > 0)
            total_ms += ms;
    }
    const double latency_ms = total_ms / reps;

    heongpu::Plaintext<heongpu::Scheme::CKKS> P_res(context);
    decryptor.decrypt(P_res, out);
    std::vector<Complex64> res;
    encoder.decode(res, P_res);
    double sum_err = 0, max_err = 0;
    for (int i = 0; i < slot_count; ++i)
    {
        const double e = std::abs(res[i] - Complex64(message[i], 0.0));
        sum_err += e;
        max_err = std::max(max_err, e);
    }
    const double mean_err = sum_err / slot_count;
    const double bits = -std::log2(mean_err);

    std::cout << "N = 2^" << logN << ", slots = " << slot_count << ", h = " << h
              << ", taylor = " << taylor << ", galois keys = " << key_index.size()
              << "\nlevels left after bootstrapping: "
              << (28 - out.depth()) << "\nlatency: " << latency_ms
              << " ms (avg of " << reps << " after warm-up)"
              << "\nprecision: mean " << bits << " bits, min "
              << -std::log2(max_err) << " bits"
              << "\nthroughput: " << slot_count / (latency_ms / 1000.0)
              << " slots/s, " << slot_count * bits / (latency_ms / 1000.0) / 8e3
              << " KB/s of precision bits" << std::endl;
    return 0;
}
