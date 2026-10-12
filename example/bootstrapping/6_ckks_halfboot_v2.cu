// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// HalfBoot v2 (ModUp + CoeffToSlot + EvalMod, no SlotToCoeff) with the
// parameters of the RtF/Rubato reference implementation
// (RtF-Transciphering/ckks_fv/rtf_params.go):
//   chain "af" = RtFRubatoParams 128af (default): Q = Residual (q0 + 7x45) |
//     DiffScale (22) | SineEval (3 arcsine, 2 double angle, 6 sine, 60-bit) |
//     CtS (4x58), P = 4x61, scale 2^45, message ratio 16, K = 25, sine
//     degree 63, double angle 2, arcsine degree 7, h = 192.
//   chain "f" = RtFHeraParams 128f: Q = Residual (q0 + 11x40) | DiffScale (24)
//     | SineEval (2 double angle, 6 sine) | CtS (4x56), P = 5x61, scale 2^40,
//     message ratio 512, no arcsine.
//
// The input is a level-0 ciphertext whose plaintext coefficients are
// round(Delta_in * m_k), Delta_in = q0 / message_ratio, m_k uniform in (-1, 1)
// over all N coefficients (the output of the RtF BFV->CKKS conversion).
//
// Usage: 6_ckks_halfboot_v2 [arcsine_deg=7] [message_ratio=16] [chain=af]
//                           [reps=10] [cts_bsgs_ratio=2]
//                           [log2_output_scale=45 (af) / 40 (f)]
//
// Keys are cached in /scratch/08467/yjeon/keycache/A-<tag>/ (tag = N, hash of
// the Q/P moduli, key-switching method, secret Hamming weight) and reused.

#include "heongpu.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>

using Scheme = heongpu::Scheme;
constexpr Scheme CKKS = Scheme::CKKS;

namespace
{
    std::vector<Data64> rubato_128af_q()
    {
        return {
            0x10000000006e0001, // 60 Q0
            0x2000000a0001, // 45
            0x2000000e0001, // 45
            0x1fffffc20001, // 45
            0x200000440001, // 45
            0x200000500001, // 45
            0x200000620001, // 45
            0x1fffff980001, // 45
            0x2a0001, // 22 DiffScale
            0xffffffffffc0001, // 60 ArcSine
            0xfffffffff240001, // 60 ArcSine
            0x1000000000f00001, // 60 ArcSine
            0xfffffffff840001, // 60 Double angle
            0x1000000000860001, // 60 Double angle
            0xfffffffff6a0001, // 60 Sine
            0x1000000000980001, // 60 Sine
            0xfffffffff5a0001, // 60 Sine
            0x1000000000b00001, // 60 Sine
            0x1000000000ce0001, // 60 Sine
            0xfffffffff2a0001, // 60 Sine
            0x400000000360001, // 58 CtS
            0x3ffffffffbe0001, // 58 CtS
            0x400000000660001, // 58 CtS
            0x4000000008a0001, // 58 CtS
        };
    }

    std::vector<Data64> rubato_128af_p()
    {
        return {
            0x1fffffffffe00001, // 61
            0x1fffffffffc80001, // 61
            0x1fffffffffb40001, // 61
            0x1fffffffff500001, // 61
        };
    }

    std::vector<Data64> hera_128f_q()
    {
        return {
            0x10000000006e0001, // 60 Q0
            0x10000140001, // 40
            0xffffe80001, // 40
            0xffffc40001, // 40
            0x100003e0001, // 40
            0xffffb20001, // 40
            0x10000500001, // 40
            0xffff940001, // 40
            0xffff8a0001, // 40
            0xffff820001, // 40
            0xffff780001, // 40
            0x10000960001, // 40
            0xfc0001, // 24 DiffScale
            0xfffffffff840001, // 60 Sine (double angle)
            0x1000000000860001, // 60 Sine (double angle)
            0xfffffffff6a0001, // 60 Sine
            0x1000000000980001, // 60 Sine
            0xfffffffff5a0001, // 60 Sine
            0x1000000000b00001, // 60 Sine
            0x1000000000ce0001, // 60 Sine
            0xfffffffff2a0001, // 60 Sine
            0x100000000060001, // 56 CtS
            0xfffffffff00001, // 56 CtS
            0xffffffffd80001, // 56 CtS
            0x1000000002a0001, // 56 CtS
        };
    }

    std::vector<Data64> hera_128f_p()
    {
        return {
            0x1fffffffffe00001, // 61
            0x1fffffffffc80001, // 61
            0x1fffffffffb40001, // 61
            0x1fffffffff500001, // 61
            0x1fffffffff420001, // 61
        };
    }

    int bit_reverse(int x, int bits)
    {
        int r = 0;
        for (int i = 0; i < bits; i++)
        {
            r = (r << 1) | ((x >> i) & 1);
        }
        return r;
    }

    // ---------------------------------------------------------------------
    // Key cache (binary save/load of the key classes)
    // ---------------------------------------------------------------------
    uint64_t fnv1a(const std::vector<Data64>& values)
    {
        uint64_t h = 1469598103934665603ULL;
        for (Data64 v : values)
        {
            for (int b = 0; b < 8; b++)
            {
                h ^= (v >> (8 * b)) & 0xff;
                h *= 1099511628211ULL;
            }
        }
        return h;
    }

    template <typename Key> bool load_key(const std::string& path, Key& key)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            return false;
        }
        key.load(f);
        return static_cast<bool>(f);
    }

    template <typename Key>
    void save_key(const std::string& path, const Key& key)
    {
        const std::string tmp =
            path + ".tmp." + std::to_string(static_cast<long>(getpid()));
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            key.save(f);
            f.flush();
            if (!f)
            {
                throw std::runtime_error("cannot write " + tmp);
            }
        }
        std::filesystem::rename(tmp, path);
    }

    std::string read_text(const std::string& path)
    {
        std::ifstream f(path);
        if (!f)
        {
            return "";
        }
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    void write_text(const std::string& path, const std::string& text)
    {
        const std::string tmp =
            path + ".tmp." + std::to_string(static_cast<long>(getpid()));
        {
            std::ofstream f(tmp, std::ios::trunc);
            f << text;
        }
        std::filesystem::rename(tmp, path);
    }

    std::string index_list_text(const std::vector<int>& idx)
    {
        std::stringstream ss;
        for (int i : idx)
        {
            ss << i << "\n";
        }
        return ss.str();
    }

    // ---------------------------------------------------------------------
    // Level-0 ciphertext whose plaintext coefficients are exactly coeffs[k]
    // (residues mod every q_i, forward NTT with the context's roots, encrypt
    // at the top level, then drop to level 0).
    // ---------------------------------------------------------------------
    heongpu::Ciphertext<CKKS>
    encrypt_coefficients(heongpu::HEContext<CKKS>& context,
                         heongpu::HEEncoder<CKKS>& encoder,
                         heongpu::HEEncryptor<CKKS>& encryptor,
                         heongpu::HEArithmeticOperator<CKKS>& operators,
                         const std::vector<int64_t>& coeffs, double scale,
                         heongpu::Plaintext<CKKS>* plain_out = nullptr)
    {
        const int N = context.get_poly_modulus_degree();
        const int n_power = context.get_log_poly_modulus_degree();
        const int Q_size = context.get_ciphertext_modulus_count();
        std::vector<Modulus64> all_primes = context.get_key_modulus();
        std::vector<Modulus64> q_primes(all_primes.begin(),
                                        all_primes.begin() + Q_size);

        std::vector<Data64> host(static_cast<size_t>(N) * Q_size);
        for (int i = 0; i < Q_size; i++)
        {
            const int64_t q = static_cast<int64_t>(q_primes[i].value);
            for (int k = 0; k < N; k++)
            {
                int64_t r = coeffs[k] % q;
                if (r < 0)
                {
                    r += q;
                }
                host[static_cast<size_t>(i) * N + k] = static_cast<Data64>(r);
            }
        }

        std::vector<Data64> psi =
            heongpu::generate_primitive_root_of_unity(N, q_primes);
        std::vector<Root64> ntt_table =
            heongpu::generate_ntt_table(psi, q_primes, n_power);
        heongpu::DeviceVector<Root64> ntt_table_gpu(ntt_table);
        heongpu::DeviceVector<Modulus64> modulus_gpu(q_primes);

        // Plaintext with the right metadata (depth 0, scale), then overwrite
        // its data.
        heongpu::Plaintext<CKKS> plain(context);
        std::vector<double> zeros(N / 2, 0.0);
        encoder.encode(plain, zeros, scale);
        cudaMemcpy(plain.data(), host.data(), host.size() * sizeof(Data64),
                   cudaMemcpyHostToDevice);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = 0};
        gpuntt::GPU_NTT_Inplace(plain.data(), ntt_table_gpu.data(),
                                modulus_gpu.data(), cfg_ntt, Q_size, Q_size);
        cudaDeviceSynchronize();

        if (plain_out != nullptr)
        {
            *plain_out = plain;
        }

        heongpu::Ciphertext<CKKS> cipher(context);
        encryptor.encrypt(cipher, plain);
        for (int i = 0; i < Q_size - 1; i++)
        {
            operators.mod_drop_inplace(cipher);
        }
        return cipher;
    }

    struct ErrStats
    {
        double mean = 0.0;
        double max = 0.0;
        double imag_max = 0.0;
    };
} // namespace

int main(int argc, char* argv[])
{
    const int arcsine_deg = (argc > 1) ? std::atoi(argv[1]) : 7;
    const double message_ratio = (argc > 2) ? std::atof(argv[2]) : 16.0;
    const std::string chain = (argc > 3) ? argv[3] : "af";
    const int reps = (argc > 4) ? std::atoi(argv[4]) : 10;
    const double cts_bsgs_ratio = (argc > 5) ? std::atof(argv[5]) : 2.0;

    cudaSetDevice(0);

    const int logN = 16;
    const size_t N = size_t(1) << logN;
    const int h = 192;
    const auto ks_method = heongpu::keyswitching_type::KEYSWITCHING_METHOD_II;

    std::vector<Data64> Q, P;
    double output_scale;
    if (chain == "af")
    {
        Q = rubato_128af_q();
        P = rubato_128af_p();
        output_scale = std::pow(2.0, 45);
    }
    else if (chain == "f")
    {
        Q = hera_128f_q();
        P = hera_128f_p();
        output_scale = std::pow(2.0, 40);
    }
    else
    {
        std::cerr << "unknown chain " << chain << " (af | f)" << std::endl;
        return EXIT_FAILURE;
    }
    if (argc > 6)
    {
        output_scale = std::pow(2.0, std::atof(argv[6]));
    }

    heongpu::HEContext<CKKS> context(ks_method,
                                     heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(N);
    context.set_coeff_modulus_values(Q, P);
    context.generate();
    context.print_parameters();

    const int Q_size = static_cast<int>(Q.size());
    const int max_level = Q_size - 1;
    const int cts_piece = 4;
    const int eval_level_start = max_level - cts_piece;
    const int K = 25;
    const int sine_deg = 63;
    const int double_angle = 2;
    const int eval_depth =
        static_cast<int>(std::ceil(std::log2(sine_deg + 1))) + double_angle +
        (arcsine_deg > 0
             ? static_cast<int>(std::ceil(std::log2(arcsine_deg + 1)))
             : 0);
    const int out_level = eval_level_start - eval_depth - 1;

    heongpu::HEEncoder<CKKS> encoder(context);
    heongpu::HEArithmeticOperator<CKKS> operators(context, encoder);

    heongpu::EvalModConfig eval_mod_config(
        0, eval_level_start, message_ratio, K, sine_deg, double_angle,
        arcsine_deg, 0.0);
    // SlotToCoeff matrices are generated by generate_bootstrapping_params_v2
    // but not used by HalfBoot; put them where SlotToCoeff would start.
    heongpu::BootstrappingConfigV2 boot_config(
        heongpu::EncodingMatrixConfig(
            heongpu::LinearTransformType::SLOTS_TO_COEFFS,
            std::max(out_level, 3)),
        eval_mod_config,
        heongpu::EncodingMatrixConfig(
            heongpu::LinearTransformType::COEFFS_TO_SLOTS, max_level,
            cts_bsgs_ratio, cts_piece));
    operators.generate_bootstrapping_params_v2(output_scale, boot_config);
    const std::vector<int> galois_index =
        operators.halfbootstrapping_key_indexs();

    // ---------------------------------------------------------------------
    // Keys (cached on disk, binary save/load of the key classes)
    // ---------------------------------------------------------------------
    std::vector<Data64> qp = Q;
    qp.insert(qp.end(), P.begin(), P.end());
    std::stringstream tag;
    tag << "N" << N << "_QP" << std::hex << fnv1a(qp) << std::dec << "_ks"
        << static_cast<int>(ks_method) << "_h" << h;
    const std::string key_dir =
        "/scratch/08467/yjeon/keycache/A-" + tag.str();
    std::filesystem::create_directories(key_dir);
    const std::string sk_path = key_dir + "/sk.bin";
    const std::string pk_path = key_dir + "/pk.bin";
    const std::string rlk_path = key_dir + "/rlk.bin";
    const std::string gk_path = key_dir + "/gk_halfboot.bin";
    const std::string gk_idx_path = key_dir + "/gk_halfboot_index.txt";

    // Keys are held through unique_ptr so that a failed load is replaced by a
    // freshly constructed object (no key move/assignment involved).
    heongpu::HEKeyGenerator<CKKS> keygen(context);
    std::vector<int> galois_index_mut = galois_index;
    auto new_sk = [&]
    { return std::make_unique<heongpu::Secretkey<CKKS>>(context, h); };
    auto new_pk = [&]
    { return std::make_unique<heongpu::Publickey<CKKS>>(context); };
    auto new_rlk = [&]
    { return std::make_unique<heongpu::Relinkey<CKKS>>(context); };
    auto new_gk = [&]
    {
        return std::make_unique<heongpu::Galoiskey<CKKS>>(context,
                                                          galois_index_mut);
    };

    auto t_key0 = std::chrono::steady_clock::now();
    int generated = 0;
    auto secret_key = new_sk();
    const bool sk_loaded = load_key(sk_path, *secret_key);
    if (!sk_loaded)
    {
        // A new secret key invalidates every other cached key.
        secret_key = new_sk();
        keygen.generate_secret_key_v2(*secret_key);
        save_key(sk_path, *secret_key);
        generated++;
    }
    auto public_key = new_pk();
    if (!(sk_loaded && load_key(pk_path, *public_key)))
    {
        public_key = new_pk();
        keygen.generate_public_key(*public_key, *secret_key);
        save_key(pk_path, *public_key);
        generated++;
    }
    auto relin_key = new_rlk();
    if (!(sk_loaded && load_key(rlk_path, *relin_key)))
    {
        relin_key = new_rlk();
        keygen.generate_relin_key(*relin_key, *secret_key);
        save_key(rlk_path, *relin_key);
        generated++;
    }
    // The rotation set depends on the CtS configuration: the index list is
    // stored next to the keys and the keys are regenerated on mismatch.
    const std::string idx_text = index_list_text(galois_index);
    auto galois_key = new_gk();
    if (!(sk_loaded && read_text(gk_idx_path) == idx_text &&
          load_key(gk_path, *galois_key)))
    {
        galois_key = new_gk();
        keygen.generate_galois_key(*galois_key, *secret_key);
        save_key(gk_path, *galois_key);
        write_text(gk_idx_path, idx_text);
        generated++;
    }
    cudaDeviceSynchronize();
    auto t_key1 = std::chrono::steady_clock::now();
    std::cout << "Keys: " << (4 - generated) << " loaded, " << generated
              << " generated, dir " << key_dir << " ("
              << std::chrono::duration<double>(t_key1 - t_key0).count()
              << " s), HalfBoot rotation keys: " << galois_index.size()
              << std::endl;

    heongpu::HEEncryptor<CKKS> encryptor(context, *public_key);
    heongpu::HEDecryptor<CKKS> decryptor(context, *secret_key);

    // ---------------------------------------------------------------------
    // Input: coefficients round(Delta_in * m_k), Delta_in = q0 / ratio
    // (not a power of two)
    // ---------------------------------------------------------------------
    const double q0 = static_cast<double>(Q[0]);
    const double delta_in = q0 / message_ratio;
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::vector<double> msg(N);
    std::vector<int64_t> coeffs(N);
    for (size_t k = 0; k < N; k++)
    {
        msg[k] = unif(rng);
        coeffs[k] = std::llround(delta_in * msg[k]);
    }

    heongpu::Plaintext<CKKS> plain_in(context);
    heongpu::Ciphertext<CKKS> c_in = encrypt_coefficients(
        context, encoder, encryptor, operators, coeffs, delta_in, &plain_in);
    {
        // Sanity check of the input: decrypt(c_in) == plain_in (slots).
        std::vector<Complex64> z_plain, z_dec;
        encoder.decode(z_plain, plain_in);
        heongpu::Plaintext<CKKS> p_dec(context);
        decryptor.decrypt(p_dec, c_in);
        encoder.decode(z_dec, p_dec);
        double dmax = 0.0;
        for (size_t j = 0; j < z_plain.size(); j++)
        {
            dmax = std::max(dmax,
                            std::hypot(z_plain[j].real() - z_dec[j].real(),
                                       z_plain[j].imag() - z_dec[j].imag()));
        }
        std::cout << "Input level " << c_in.level() << ", scale 2^"
                  << std::setprecision(12) << std::log2(c_in.scale())
                  << std::setprecision(6) << " (= q0/" << message_ratio
                  << "), max |decrypt - plaintext| over slots " << dmax
                  << std::endl;
    }

    std::cout << "chain " << chain << ", message ratio " << message_ratio
              << ", K " << K << ", sine deg " << sine_deg
              << ", double angle " << double_angle << ", arcsine deg "
              << arcsine_deg << ", CtS bsgs ratio " << cts_bsgs_ratio
              << std::endl;

    // ---------------------------------------------------------------------
    // Warm-up + timed runs
    // ---------------------------------------------------------------------
    auto run = [&]()
    {
        return operators.regular_halfbootstrapping_v2(c_in, *galois_key,
                                                      *relin_key);
    };

    auto result = run(); // warm-up
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<float> times;
    for (int r = 0; r < reps; r++)
    {
        cudaEventRecord(start);
        result = run();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
    }

    heongpu::Ciphertext<CKKS>& y0 = result.first;
    heongpu::Ciphertext<CKKS>& y1 = result.second;
    std::cout << "Output level " << y0.level() << " / " << y1.level()
              << ", scale 2^" << std::setprecision(15) << std::log2(y0.scale())
              << " / 2^" << std::log2(y1.scale()) << std::setprecision(6)
              << " (target 2^" << std::log2(output_scale) << ")" << std::endl;
    if (!times.empty())
    {
        std::vector<float> sorted_t = times;
        std::sort(sorted_t.begin(), sorted_t.end());
        double avg = 0;
        for (float t : times)
        {
            avg += t;
        }
        avg /= times.size();
        std::cout << "HalfBoot time: avg " << avg << " ms, median "
                  << sorted_t[sorted_t.size() / 2] << " ms, min "
                  << sorted_t.front() << " ms (" << reps
                  << " runs after warm-up)" << std::endl;
    }

    // ---------------------------------------------------------------------
    // Decode and determine the slot <-> coefficient mapping
    // ---------------------------------------------------------------------
    std::vector<Complex64> d[2];
    {
        heongpu::Plaintext<CKKS> p0(context), p1(context);
        decryptor.decrypt(p0, y0);
        decryptor.decrypt(p1, y1);
        encoder.decode(d[0], p0);
        encoder.decode(d[1], p1);
    }
    const int slots = static_cast<int>(N / 2);
    const int half_bits = logN - 1;

    struct Candidate
    {
        const char* name;
        std::function<int(int, int)> f; // (output, slot) -> coefficient
    };
    std::vector<Candidate> candidates = {
        {"out0 slot s <-> coeff bitrev(s), out1 slot s <-> coeff N/2 + "
         "bitrev(s)",
         [&](int o, int s)
         { return o * slots + bit_reverse(s, half_bits); }},
        {"out0 slot s <-> coeff s, out1 slot s <-> coeff N/2 + s",
         [&](int o, int s) { return o * slots + s; }},
        {"out0 slot s <-> coeff 2 bitrev(s), out1 slot s <-> coeff "
         "2 bitrev(s) + 1",
         [&](int o, int s) { return 2 * bit_reverse(s, half_bits) + o; }},
        {"out0 slot s <-> coeff 2s, out1 slot s <-> coeff 2s + 1",
         [&](int o, int s) { return 2 * s + o; }},
    };

    // A wrong mapping pairs independent uniform values: mean |err| ~ 2/3.
    int best_c = 0;
    double best_err = 1e300;
    for (size_t c = 0; c < candidates.size(); c++)
    {
        double sum = 0.0;
        for (int o = 0; o < 2; o++)
        {
            for (int s = 0; s < slots; s++)
            {
                sum += std::fabs(d[o][s].real() - msg[candidates[c].f(o, s)]);
            }
        }
        const double mean = sum / N;
        std::cout << "  mapping candidate [" << candidates[c].name
                  << "]: mean |err| " << mean << std::endl;
        if (mean < best_err)
        {
            best_err = mean;
            best_c = static_cast<int>(c);
        }
    }
    std::cout << "Slot mapping: " << candidates[best_c].name
              << " (real part; imaginary part ~ 0)" << std::endl;

    ErrStats st[2];
    std::vector<int> covered(N, 0);
    for (int o = 0; o < 2; o++)
    {
        double sum = 0.0;
        for (int s = 0; s < slots; s++)
        {
            const int k = candidates[best_c].f(o, s);
            covered[k]++;
            const double e = std::fabs(d[o][s].real() - msg[k]);
            sum += e;
            st[o].max = std::max(st[o].max, e);
            st[o].imag_max = std::max(st[o].imag_max, std::fabs(d[o][s].imag()));
        }
        st[o].mean = sum / slots;
    }
    const bool bijective =
        std::all_of(covered.begin(), covered.end(),
                    [](int c) { return c == 1; });
    const double mean_all = 0.5 * (st[0].mean + st[1].mean);
    const double max_all = std::max(st[0].max, st[1].max);

    for (int o = 0; o < 2; o++)
    {
        std::cout << std::scientific << std::setprecision(3) << "  out" << o
                  << ": mean |err| " << st[o].mean << std::fixed
                  << std::setprecision(2) << " (" << -std::log2(st[o].mean)
                  << " bits), max |err| " << std::scientific
                  << std::setprecision(3) << st[o].max << std::fixed
                  << std::setprecision(2) << " (" << -std::log2(st[o].max)
                  << " bits), max |imag| " << std::scientific
                  << std::setprecision(3) << st[o].imag_max << std::endl;
    }
    std::cout << std::fixed << std::setprecision(2)
              << "PRECISION over all N coefficients (mapping "
              << (bijective ? "bijective" : "NOT bijective") << "): mean "
              << -std::log2(mean_all) << " bits, min " << -std::log2(max_all)
              << " bits" << std::endl;

    // Split the error into the deterministic EvalMod approximation error
    // (sin, or degree-d Taylor arcsin of sin) and the remaining noise:
    // predicted output = ratio / (2 pi) * g(sin(2 pi m / ratio)), g = id or
    // the arcsine Taylor polynomial.
    {
        auto predicted = [&](double m)
        {
            const double y = std::sin(2.0 * M_PI * m / message_ratio);
            double g = y;
            if (arcsine_deg > 0)
            {
                double c = 1.0, term = y, sum = 0.0;
                for (int i = 1; i <= arcsine_deg; i += 2)
                {
                    sum += c * term;
                    c *= static_cast<double>(i * i) /
                         static_cast<double>((i + 1) * (i + 2));
                    term *= y * y;
                }
                g = sum;
            }
            return message_ratio / (2.0 * M_PI) * g;
        };
        double approx_max = 0.0, noise_sum = 0.0, noise_sq = 0.0,
               noise_max = 0.0;
        for (int o = 0; o < 2; o++)
        {
            for (int s = 0; s < slots; s++)
            {
                const int k = candidates[best_c].f(o, s);
                const double pr = predicted(msg[k]);
                approx_max = std::max(approx_max, std::fabs(pr - msg[k]));
                const double r = d[o][s].real() - pr;
                noise_sum += std::fabs(r);
                noise_sq += r * r;
                noise_max = std::max(noise_max, std::fabs(r));
            }
        }
        std::cout << std::scientific << std::setprecision(3)
                  << "  EvalMod approximation error (analytic): max "
                  << approx_max << "; residual noise: mean |.| "
                  << noise_sum / N << ", rms " << std::sqrt(noise_sq / N)
                  << ", max " << noise_max << std::fixed
                  << std::setprecision(2) << " (" << -std::log2(noise_sum / N)
                  << " / " << -std::log2(noise_max) << " bits)" << std::endl;
    }

    if (const char* dump = std::getenv("HB_DUMP"))
    {
        // coefficient index, message, decoded output (real part)
        std::ofstream f(dump);
        f << std::setprecision(17);
        for (int o = 0; o < 2; o++)
        {
            for (int s = 0; s < slots; s++)
            {
                const int k = candidates[best_c].f(o, s);
                f << k << " " << msg[k] << " " << d[o][s].real() << "\n";
            }
        }
    }

    for (int s = 0; s < 4; s++)
    {
        const int k = candidates[best_c].f(0, s);
        std::cout << std::setprecision(8) << "  out0 slot " << s
                  << " -> coeff " << k << ": expected " << msg[k] << ", got "
                  << d[0][s].real() << std::endl;
    }

    return EXIT_SUCCESS;
}
