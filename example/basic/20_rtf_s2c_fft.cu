// SlotToCoeff_FV with the FFT-like factorization (rtf/s2c_fft.cuh).
//
//   ./20_rtf_s2c_fft [logN] [levels_per_group] [reps]
//
// Checks that coefficient coeff_index(i) of the output equals input slot i,
// and times the transform.

#include "heongpu.cuh"
#include "rtf/s2c_fft.cuh"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::RTF;

static double now_ms()
{
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    const int logN = (argc > 1) ? std::atoi(argv[1]) : 12;
    const int lpg = (argc > 2) ? std::atoi(argv[2]) : 5;
    const int reps = (argc > 3) ? std::atoi(argv[3]) : 3;
    const size_t N = size_t(1) << logN;

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(N);
    std::vector<int> logQ(30, 50);
    logQ[0] = 60;
    std::vector<int> logP = {60};
    context.set_coeff_modulus_bit_sizes(logQ, logP);
    context.set_plain_modulus(0x1fc0001ULL);
    context.generate();

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> sk(context);
    keygen.generate_secret_key(sk);
    heongpu::Publickey<Scheme> pk(context);
    keygen.generate_public_key(pk, sk);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, pk);
    heongpu::HEDecryptor<Scheme> decryptor(context, sk);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);

    double t0 = now_ms();
    heongpu::FVSlotToCoeffFFT s2c(context, encoder, op, lpg);
    double t1 = now_ms();
    std::vector<int> shifts = s2c.rotation_shifts();
    std::cout << "N = " << N << ", groups = " << s2c.group_count()
              << ", rotation keys = " << shifts.size() << " (+ column), precompute "
              << (t1 - t0) << " ms" << std::endl;
    std::cout << "shifts:";
    for (int s : shifts)
        std::cout << " " << s;
    std::cout << std::endl;

    heongpu::ExecutionOptions on_device;
    on_device.set_storage_type(heongpu::storage_type::DEVICE);
    heongpu::Galoiskey<Scheme> gk(context, shifts);
    keygen.generate_galois_key(gk, sk, on_device);

    std::mt19937_64 rng(7);
    std::vector<uint64_t> v(N);
    for (auto& x : v)
        x = rng() % 0x1fc0001ULL;
    heongpu::Plaintext<Scheme> pt(context);
    encoder.encode(pt, v);
    heongpu::Ciphertext<Scheme> ct(context);
    encryptor.encrypt(ct, pt);
    std::cout << "noise budget before: " << decryptor.remainder_noise_budget(ct)
              << std::endl;

    auto out = s2c.apply(ct, gk);
    std::cout << "noise budget after:  " << decryptor.remainder_noise_budget(out)
              << std::endl;

    heongpu::Plaintext<Scheme> res(context);
    decryptor.decrypt(res, out);
    std::vector<uint64_t> coeffs(N);
    cudaDeviceSynchronize();
    cudaMemcpy(coeffs.data(), res.data(), N * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    size_t bad = 0;
    for (size_t i = 0; i < N; ++i)
        bad += (coeffs[s2c.coeff_index((int) i)] != v[i]);
    std::cout << (bad == 0 ? "[PASS] " : "[FAIL] ")
              << "coefficient coeff_index(i) == slot i (" << bad
              << " mismatches)" << std::endl;

    double sum = 0;
    for (int r = 0; r < reps; ++r)
    {
        heongpu::Ciphertext<Scheme> c = ct;
        double a = now_ms();
        auto o = s2c.apply(c, gk);
        sum += now_ms() - a;
    }
    std::cout << "S2C_FV (FFT) time: " << sum / reps << " ms (avg of " << reps
              << ")" << std::endl;
    return bad == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
