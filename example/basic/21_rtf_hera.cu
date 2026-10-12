// RtF transciphering with HERA (Par-128a), checked stage by stage, on the
// HE parameters of the RtF reference implementation ("128af" of
// RtF-Transciphering/ckks_fv/rtf_params.go, RtFHeraParams[2], the same chain
// as RtFRubatoParams[0] used by 19_rtf_rubato):
//
//   Q = q0 (60 bit) | 7 x 45-bit | 22-bit (diff. scale) | 11 x 60-bit (sine)
//       | 4 x 58-bit (CoeffsToSlots),   P = 4 x 61-bit,   h = 192,
//
// hybrid key switching (KEYSWITCHING_METHOD_II, dnum = 6), one sparse ternary
// secret key shared by FV and CKKS, t = 0x1fc0001, q0 / Delta' = 16, arcsine.
//
//   ./21_rtf_hera [logN] [reps] [halfboot]
//        logN      16 for the reference parameters (default 12: same moduli,
//                  no security, for quick tests)
//        reps      timed repetitions after one warm-up (default 3)
//        halfboot  1 | 0: run stage 3 (CKKS HalfBoot) and its timing
//
// Packing: column-wise, one 16-word HERA state per 16-slot block (N / 16
// blocks); MixRows o MixColumns = M4 (x) M4 per block as one slot-domain
// linear transform (rtf/slot_lintrans.cuh), the key k replicated per block.
//
// FV levels (the reference's CipherModDown {5,2,2,2,2,2} for 128af): round i
// (ARK_i, MixRows o MixColumns, Cube) runs on the first k_i primes, the final
// MixRows o MixColumns and ARK_5 on k_5, SlotToCoeff_FV and C - Z on k_s2c.
// Default RTF_LEVELS=19,17,15,13,11,9,5 (k_0..k_5, k_s2c); RTF_LEVELS=off
// runs everything on all primes. Each level is its own RTF context with the
// same secret (rebuilt from its ternary coefficients) and its own keys.
//
// Keys of the full chain (secret, public, relinearization, S2C rotations)
// are cached under /scratch/08467/yjeon/keycache/B-<tag>/ (shared with
// 19_rtf_rubato); RTF_KEYCACHE=off disables, RTF_KEYCACHE=<dir> moves it.
//
// Stage 1: the server's homomorphic HERA keystream equals the host reference
//          (rtf/hera_ref.h) word for word, and the host reference matches the
//          known-answer vectors of dev-tools/hera_ref.py.
// Stage 2: SlotToCoeff_FV moves keystream slot i to coefficient
//          s2c_coeff_index(i) (r*N/2 + bitrev(k) for slot (r, k)).
// Stage 2b: FV down to q0: after C - Z and the mod-down to q0, the level-0
//          ciphertext decrypts (centred mod q0, times t / q0) to exactly
//          round(Delta m) (HERA has no noise).
// Stage 3: end to end, the client's real messages come out of HalfBoot.
// Stage 4: server timing (offline: round constants, HERA, SlotToCoeff_FV;
//          online: VecToPoly, C - Z, mod-down to q0, HalfBoot), memory.

#include "heongpu.cuh"
#include "../example_util.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::RTF;
constexpr auto SchemeCKKS = heongpu::Scheme::CKKS;
namespace hera = heongpu::hera;
namespace fs = std::filesystem;

// RtF "128af" moduli in HEonGPU order (q0 first, CoeffsToSlots last).
static const std::vector<Data64> kQ128af = {
    0x10000000006e0001ULL,                                     // q0, 60
    0x2000000a0001ULL,     0x2000000e0001ULL, 0x1fffffc20001ULL, // 45
    0x200000440001ULL,     0x200000500001ULL, 0x200000620001ULL, // 45
    0x1fffff980001ULL,                                         // 45
    0x2a0001ULL,                                               // 22, diff.
    0xffffffffffc0001ULL,  0xfffffffff240001ULL,               // ArcSine
    0x1000000000f00001ULL,                                     // ArcSine
    0xfffffffff840001ULL,  0x1000000000860001ULL,              // DoubleAngle
    0xfffffffff6a0001ULL,  0x1000000000980001ULL,              // Sine
    0xfffffffff5a0001ULL,  0x1000000000b00001ULL,              // Sine
    0x1000000000ce0001ULL, 0xfffffffff2a0001ULL,               // Sine
    0x400000000360001ULL,  0x3ffffffffbe0001ULL,               // CtS, 58
    0x400000000660001ULL,  0x4000000008a0001ULL};              // CtS, 58
static const std::vector<Data64> kP128af = {
    0x1fffffffffe00001ULL, 0x1fffffffffc80001ULL, 0x1fffffffffb40001ULL,
    0x1fffffffff500001ULL}; // 61
static const int kHammingWeight = 192;

static double now_ms()
{
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static int bit_reverse(int x, int bits)
{
    int r = 0;
    for (int i = 0; i < bits; ++i)
        r |= ((x >> i) & 1) << (bits - 1 - i);
    return r;
}

static bool check(const char* what, bool ok)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << std::endl;
    return ok;
}

static double log2_product(const std::vector<Data64>& v)
{
    double s = 0;
    for (auto x : v)
        s += std::log2((double) x);
    return s;
}

static double gib(size_t bytes) { return bytes / double(1ULL << 30); }

static void report_memory(const char* tag)
{
    auto& mp = MemoryPool::instance();
    double rss = 0, hwm = 0;
    {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line))
        {
            if (line.rfind("VmRSS:", 0) == 0)
                rss = std::stod(line.substr(6)) / (1024.0 * 1024.0);
            if (line.rfind("VmHWM:", 0) == 0)
                hwm = std::stod(line.substr(6)) / (1024.0 * 1024.0);
        }
    }
    const auto flags = std::cout.flags();
    const auto prec = std::cout.precision();
    std::cout << std::fixed << std::setprecision(2) << "[mem] " << tag
              << ": GPU pool in use "
              << gib(mp.get_current_device_pool_memory_usage()) << " GiB (peak "
              << gib(mp.get_peak_device_pool_memory_usage())
              << "), host RSS " << rss << " GiB (peak " << hwm << ")"
              << std::endl;
    std::cout.flags(flags);
    std::cout.precision(prec);
}

// On-disk key cache (write to a temporary name, then rename).
struct KeyCache
{
    bool enabled = false;
    std::string dir;
    std::vector<char> buf = std::vector<char>(size_t(64) << 20);

    std::string path(const std::string& name) const { return dir + "/" + name; }
    bool has(const std::string& name) const
    {
        return enabled && fs::exists(path(name));
    }
    template <typename K> void save(const std::string& name, const K& key)
    {
        if (!enabled)
            return;
        const std::string final_path = path(name);
        const std::string tmp =
            final_path + ".tmp." + std::to_string((long) getpid());
        {
            std::ofstream os;
            os.rdbuf()->pubsetbuf(buf.data(), buf.size());
            os.open(tmp, std::ios::binary | std::ios::trunc);
            if (!os)
                throw std::runtime_error("key cache: cannot write " + tmp);
            key.save(os);
            os.flush();
            if (!os)
                throw std::runtime_error("key cache: write failed " + tmp);
        }
        fs::rename(tmp, final_path);
    }
    template <typename K> void load(const std::string& name, K& key)
    {
        std::ifstream is;
        is.rdbuf()->pubsetbuf(buf.data(), buf.size());
        is.open(path(name), std::ios::binary);
        if (!is)
            throw std::runtime_error("key cache: cannot read " + path(name));
        key.load(is);
        if (!is)
            throw std::runtime_error("key cache: truncated " + path(name));
    }
};

template <typename K, typename G>
static void cached_key(KeyCache& cache, const std::string& name, K& key, G gen)
{
    double t0 = now_ms();
    if (cache.has(name))
    {
        cache.load(name, key);
        std::cout << "  " << name << ": loaded from cache ("
                  << (now_ms() - t0) / 1000.0 << " s)" << std::endl;
    }
    else
    {
        gen();
        cache.save(name, key);
        std::cout << "  " << name << ": generated ("
                  << (now_ms() - t0) / 1000.0 << " s)" << std::endl;
    }
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    setenv("GPUNTT_FAST16", "1", 0);
    const int logN = (argc > 1) ? std::atoi(argv[1]) : 12;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;
    const bool run_halfboot = (argc > 3) ? (std::atoi(argv[3]) != 0) : true;
    const size_t N = size_t(1) << logN;
    const hera::Params& hp = hera::Par128a;
    const int R = hp.rounds; // 5
    bool all_ok = true;

    // ---------------------------------------------------------------------
    // Host reference: known-answer test (key[i] = i + 1, nonce = 0, ctr = j)
    // ---------------------------------------------------------------------
    {
        std::vector<uint64_t> key(hp.n);
        for (int i = 0; i < hp.n; ++i)
            key[i] = i + 1;
        auto z0 = hera::keystream(key, heongpu::rubato::xof_seed({}, 0), hp);
        auto z1 = hera::keystream(key, heongpu::rubato::xof_seed({}, 1), hp);
        all_ok &= check("host HERA matches dev-tools/hera_ref.py",
                        z0[0] == 2460505 && z0[1] == 20696352 &&
                            z0[14] == 17672229 && z0[15] == 25833093 &&
                            z1[0] == 19976987 && z1[3] == 1907281 &&
                            z1[15] == 9520363);
    }

    // ---------------------------------------------------------------------
    // Contexts: one modulus chain for FV and CKKS, hybrid key switching
    // ---------------------------------------------------------------------
    const auto ks_method = heongpu::keyswitching_type::KEYSWITCHING_METHOD_II;
    const std::vector<Data64>& Qmod = kQ128af;
    const std::vector<Data64>& Pmod = kP128af;
    if (std::getenv("HEONGPU_HOST_POOL_GB") == nullptr)
        setenv("HEONGPU_HOST_POOL_GB", "16", 0);

    heongpu::HEContext<Scheme> context(ks_method,
                                       heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(N);
    context.set_coeff_modulus_values(Qmod, Pmod);
    context.set_plain_modulus(hp.q);
    context.generate();

    heongpu::HEContext<SchemeCKKS> contextckks(ks_method,
                                               heongpu::sec_level_type::none);
    contextckks.set_poly_modulus_degree(N);
    contextckks.set_coeff_modulus_values(Qmod, Pmod);
    contextckks.generate();

    const int Qn = context.get_ciphertext_modulus_count();
    const int Pn = context.get_key_modulus_count() - Qn;
    const double logQ = log2_product(Qmod), logP = log2_product(Pmod);
    const int dnum = (Qn + Pn - 1) / Pn;
    std::cout << "N = 2^" << logN << ", |Q| = " << Qn << " primes (logQ = "
              << logQ << "), |P| = " << Pn << " (logP = " << logP
              << "), logQP = " << logQ + logP << ", t = " << hp.q
              << ", h = " << kHammingWeight << ", METHOD_II, dnum = " << dnum
              << std::endl;

    // ---------------------------------------------------------------------
    // Key cache (same directory as 19_rtf_rubato for the same chain)
    // ---------------------------------------------------------------------
    KeyCache cache;
    {
        const char* env = std::getenv("RTF_KEYCACHE");
        std::string root = "/scratch/08467/yjeon/keycache";
        if (env != nullptr && std::string(env) == "off")
            root.clear();
        else if (env != nullptr && *env != '\0')
            root = env;
        if (!root.empty())
        {
            uint64_t h = 1469598103934665603ULL; // FNV-1a over Q, P values
            auto mix = [&](uint64_t x)
            {
                for (int b = 0; b < 8; ++b)
                {
                    h ^= (x >> (8 * b)) & 0xff;
                    h *= 1099511628211ULL;
                }
            };
            for (auto q : Qmod)
                mix(q);
            mix(0xffffffffffffffffULL);
            for (auto p : Pmod)
                mix(p);
            std::ostringstream tag;
            tag << "B-N" << logN << "_QP" << std::hex << std::setw(16)
                << std::setfill('0') << h << std::dec << "_ks"
                << static_cast<int>(ks_method) << "_dnum" << dnum << "_h"
                << kHammingWeight;
            cache.dir = root + "/" + tag.str();
            std::error_code ec;
            fs::create_directories(cache.dir, ec);
            cache.enabled = !ec;
        }
        std::cout << "key cache: "
                  << (cache.enabled ? cache.dir : std::string("disabled"))
                  << std::endl;
    }

    // ---------------------------------------------------------------------
    // Keys of the full chain: one sparse ternary secret (h = 192)
    // ---------------------------------------------------------------------
    std::cout << "keys:" << std::endl;
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context, kHammingWeight);
    cached_key(cache, "sk_rtf.bin", secret_key,
               [&] { keygen.generate_secret_key_v2(secret_key); });
    heongpu::Publickey<Scheme> public_key(context);
    cached_key(cache, "pk_rtf.bin", public_key,
               [&] { keygen.generate_public_key(public_key, secret_key); });
    heongpu::Relinkey<Scheme> relin_key(context);
    cached_key(cache, "rlk_rtf.bin", relin_key,
               [&] { keygen.generate_relin_key(relin_key, secret_key); });

    heongpu::ExecutionOptions on_device;
    on_device.set_storage_type(heongpu::storage_type::DEVICE);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);
    std::vector<int> no_shifts;
    heongpu::Galoiskey<Scheme> gk_none_main(context, no_shifts);
    // Full-chain engine: client-side key encryption and its mod switches.
    heongpu::HEHERA<Scheme> hera_main(context, contextckks, encoder, encryptor,
                                      op, gk_none_main, relin_key,
                                      heongpu::ExecutionOptions(),
                                      heongpu::StreamCipher::Hera);
    const int blocks = hera_main.get_block_count();
    std::cout << "N = " << N << ", HERA blocks per ciphertext = " << blocks
              << " (16 words each)" << std::endl;

    // ---------------------------------------------------------------------
    // FV levels: HERA stage i on the first k[i] primes, S2C on k[R + 1]
    // ---------------------------------------------------------------------
    std::vector<int> kv = {19, 17, 15, 13, 11, 9, 5};
    if (const char* lv = std::getenv("RTF_LEVELS"))
    {
        if (std::string(lv) == "off")
            kv.assign(R + 2, Qn);
        else
        {
            std::vector<int> v;
            std::stringstream ss(lv);
            std::string tok;
            while (std::getline(ss, tok, ','))
                v.push_back(std::stoi(tok));
            if ((int) v.size() != R + 2)
                throw std::invalid_argument("RTF_LEVELS: need r + 2 = 7 values");
            kv = v;
        }
    }
    for (int i = 0; i < R + 2; ++i)
        if (kv[i] < 1 || kv[i] > Qn || (i > 0 && i <= R && kv[i] > kv[i - 1]) ||
            (i == R + 1 && kv[i] > kv[R]))
            throw std::invalid_argument("RTF_LEVELS: non-increasing, <= |Q|");

    struct Level
    {
        int k = 0;
        std::unique_ptr<heongpu::HEContext<Scheme>> ctx;
        std::unique_ptr<heongpu::Secretkey<Scheme>> sk;
        std::unique_ptr<heongpu::HEKeyGenerator<Scheme>> keygen;
        std::unique_ptr<heongpu::Publickey<Scheme>> pk;
        std::unique_ptr<heongpu::Relinkey<Scheme>> rlk;
        std::unique_ptr<heongpu::HEEncoder<Scheme>> encoder;
        std::unique_ptr<heongpu::HEEncryptor<Scheme>> encryptor;
        std::unique_ptr<heongpu::HEDecryptor<Scheme>> decryptor;
        std::unique_ptr<heongpu::HEArithmeticOperator<Scheme>> op;
        std::unique_ptr<heongpu::Galoiskey<Scheme>> gk_none, gk;
        std::unique_ptr<heongpu::HEHERA<Scheme>> hera;
        std::unique_ptr<heongpu::Ciphertext<Scheme>> key_ct;
        int n_keys = 0;
    };
    std::vector<int> sk_ternary = secret_key.ternary(context);
    std::map<int, std::unique_ptr<Level>> levels;
    const double t_lv0 = now_ms();
    auto level_at = [&](int k) -> Level&
    {
        auto& slot = levels[k];
        if (slot)
            return *slot;
        auto L = std::make_unique<Level>();
        L->k = k;
        std::vector<Data64> q(Qmod.begin(), Qmod.begin() + k);
        L->ctx = std::make_unique<heongpu::HEContext<Scheme>>(
            ks_method, heongpu::sec_level_type::none);
        L->ctx->set_poly_modulus_degree(N);
        L->ctx->set_coeff_modulus_values(q, Pmod);
        L->ctx->set_plain_modulus(hp.q);
        L->ctx->generate();
        L->sk = std::make_unique<heongpu::Secretkey<Scheme>>(sk_ternary,
                                                             *L->ctx);
        L->keygen = std::make_unique<heongpu::HEKeyGenerator<Scheme>>(*L->ctx);
        L->pk = std::make_unique<heongpu::Publickey<Scheme>>(*L->ctx);
        L->keygen->generate_public_key(*L->pk, *L->sk);
        L->rlk = std::make_unique<heongpu::Relinkey<Scheme>>(*L->ctx);
        L->keygen->generate_relin_key(*L->rlk, *L->sk);
        L->encoder = std::make_unique<heongpu::HEEncoder<Scheme>>(*L->ctx);
        L->encryptor =
            std::make_unique<heongpu::HEEncryptor<Scheme>>(*L->ctx, *L->pk);
        L->decryptor =
            std::make_unique<heongpu::HEDecryptor<Scheme>>(*L->ctx, *L->sk);
        L->op = std::make_unique<heongpu::HEArithmeticOperator<Scheme>>(
            *L->ctx, *L->encoder);
        L->gk_none =
            std::make_unique<heongpu::Galoiskey<Scheme>>(*L->ctx, no_shifts);
        L->hera = std::make_unique<heongpu::HEHERA<Scheme>>(
            *L->ctx, contextckks, *L->encoder, *L->encryptor, *L->op,
            *L->gk_none, *L->rlk, heongpu::ExecutionOptions(),
            heongpu::StreamCipher::Hera);
        slot = std::move(L);
        return *slot;
    };
    // Rotation keys per level: HERA's linear layer and/or the FFT S2C.
    std::map<int, std::set<int>> shifts;
    for (int i = 0; i <= R; ++i)
        for (int s : level_at(kv[i]).hera->rubato_rotation_shifts())
            shifts[kv[i]].insert(s);
    Level& LS = level_at(kv[R + 1]);
    LS.hera->gen_FV_S2C_FFT(4);
    for (int s : LS.hera->s2c_fft_rotation_shifts())
        shifts[kv[R + 1]].insert(s);
    for (auto& [k, set] : shifts)
    {
        Level& L = *levels[k];
        std::vector<int> v(set.begin(), set.end());
        L.gk = std::make_unique<heongpu::Galoiskey<Scheme>>(*L.ctx, v);
        L.keygen->generate_galois_key(*L.gk, *L.sk, on_device);
        L.hera->set_galois_key_hera(*L.gk);
        L.hera->set_galois_key_s2c(*L.gk);
        L.n_keys = (int) v.size() + 1; // + column rotation
    }
    std::vector<heongpu::HEHERA<Scheme>*> stages;
    for (int i = 0; i <= R; ++i)
        stages.push_back(levels[kv[i]]->hera.get());
    heongpu::HEHERA<Scheme>& H_s2c = *LS.hera;
    {
        std::cout << "FV levels (primes: logQ): HERA rounds";
        for (int i = 0; i <= R; ++i)
            std::cout << " " << kv[i] << ":"
                      << log2_product(std::vector<Data64>(
                             Qmod.begin(), Qmod.begin() + kv[i]));
        std::cout << ", S2C/C-Z " << kv[R + 1] << ":"
                  << log2_product(std::vector<Data64>(
                         Qmod.begin(), Qmod.begin() + kv[R + 1]))
                  << "; contexts + keys in " << (now_ms() - t_lv0) / 1000.0
                  << " s" << std::endl;
        std::cout << "keys per level (relin + rotations incl. column):";
        for (auto& [k, L] : levels)
            std::cout << " " << k << ":1+" << L->n_keys;
        std::cout << std::endl;
        std::cout << "HERA rotation keys (" << shifts[kv[0]].size() << "):";
        for (int s : stages[0]->rubato_rotation_shifts())
            std::cout << " " << s;
        std::cout << std::endl;
    }
    report_memory("after FV keys");

    // ---------------------------------------------------------------------
    // Client: symmetric key k, FV-encrypted with k in every 16-slot block
    // ---------------------------------------------------------------------
    std::mt19937_64 rng(2026);
    std::vector<uint64_t> key(hp.n);
    for (auto& k : key)
        k = rng() % hp.q;
    std::vector<uint64_t> key_slots(N);
    for (size_t i = 0; i < N; ++i)
        key_slots[i] = key[i % hp.n];
    heongpu::Plaintext<Scheme> key_pt(context);
    encoder.encode(key_pt, key_slots);
    heongpu::Ciphertext<Scheme> key_ct(context);
    encryptor.encrypt(key_ct, key_pt);
    std::cout << "noise budget of the encrypted key: "
              << decryptor.remainder_noise_budget(key_ct) << " bits"
              << std::endl;
    // The encrypted key is switched once to every HERA stage's level.
    std::vector<heongpu::Ciphertext<Scheme>*> stage_keys;
    for (int i = 0; i <= R; ++i)
    {
        Level& L = *levels[kv[i]];
        if (!L.key_ct)
        {
            heongpu::Ciphertext<Scheme> tmp = key_ct;
            L.key_ct = std::make_unique<heongpu::Ciphertext<Scheme>>(
                hera_main.mod_switch_to(tmp, *L.ctx));
        }
        stage_keys.push_back(L.key_ct.get());
    }
    std::cout << "noise budget of the key at " << kv[0]
              << " primes: " << levels[kv[0]]->decryptor->remainder_noise_budget(
                                    *stage_keys[0])
              << " bits" << std::endl;

    const std::array<uint8_t, 8> nonce = {0x48, 0x45, 0x52, 0x41,
                                          0x00, 0x00, 0x00, 0x01};

    // ---------------------------------------------------------------------
    // Stage 1: homomorphic keystream vs host keystream
    // ---------------------------------------------------------------------
    using RC = heongpu::HEHERA<Scheme>::RoundConstants;
    heongpu::HEHERA<Scheme>::set_round_constants_staged(
        stages, stages[0]->compute_round_constants(nonce, 0));
    std::vector<int> budgets(R + 1);
    auto ks_ct = heongpu::HEHERA<Scheme>::gen_stream_key_staged(
        stages, stage_keys, heongpu::ExecutionOptions(),
        [&](int i, heongpu::Ciphertext<Scheme>& st)
        {
            const int k = (i < R) ? kv[i + 1] : kv[R];
            budgets[i] = levels[k]->decryptor->remainder_noise_budget(st);
        });
    std::cout << "noise budget after each HERA round (cube + switch):";
    for (int i = 0; i <= R; ++i)
        std::cout << " r" << i + 1 << "@" << ((i < R) ? kv[i + 1] : kv[R])
                  << "p=" << budgets[i];
    std::cout << " (last = after Fin)" << std::endl;
    const int budget_hera = budgets[R];

    Level& LF = *levels[kv[R]];
    heongpu::Plaintext<Scheme> ks_pt(*LF.ctx);
    LF.decryptor->decrypt(ks_pt, ks_ct);
    std::vector<uint64_t> ks_he;
    LF.encoder->decode(ks_he, ks_pt);

    size_t bad_words = 0;
    for (int j = 0; j < blocks; ++j)
    {
        auto z = hera::keystream(key, heongpu::rubato::xof_seed(nonce, j), hp);
        for (int i = 0; i < hp.n; ++i)
            bad_words += (ks_he[j * hp.n + i] != z[i]);
    }
    all_ok &= check("HE keystream == host keystream (all blocks, 16 words)",
                    bad_words == 0);
    if (bad_words != 0)
        std::cout << "  mismatching words: " << bad_words << std::endl;

    // ---------------------------------------------------------------------
    // Stage 2: SlotToCoeff_FV
    // ---------------------------------------------------------------------
    heongpu::Ciphertext<Scheme> ks_s2c =
        (kv[R + 1] != kv[R]) ? stages[R]->mod_switch_to(ks_ct, *LS.ctx) : ks_ct;
    const int budget_s2c_in = LS.decryptor->remainder_noise_budget(ks_s2c);
    std::cout << "noise budget of the keystream at " << kv[R + 1]
              << " primes: " << budget_s2c_in << " bits" << std::endl;
    auto z_coeff_ct = H_s2c.S2C_FV(ks_s2c);
    const int budget_s2c = LS.decryptor->remainder_noise_budget(z_coeff_ct);
    std::cout << "noise budget after SlotToCoeff_FV: " << budget_s2c << " bits"
              << std::endl;
    {
        heongpu::Plaintext<Scheme> pt(*LS.ctx);
        LS.decryptor->decrypt(pt, z_coeff_ct);
        std::vector<uint64_t> coeffs(N);
        cudaDeviceSynchronize();
        cudaMemcpy(coeffs.data(), pt.data(), N * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        size_t bad = 0;
        for (size_t i = 0; i < N; ++i)
            bad += (coeffs[H_s2c.s2c_coeff_index((int) i)] != ks_he[i]);
        all_ok &= check("SlotToCoeff_FV: coefficient s2c_coeff_index(i) == "
                        "keystream slot i",
                        bad == 0);
    }
    report_memory("after stage 2");

    // ---------------------------------------------------------------------
    // Client: real messages m in (-1, 1), c = round(Delta m) + z mod t
    // ---------------------------------------------------------------------
    const double delta = hera_main.get_message_scaling();
    const uint64_t t = hp.q;
    std::vector<double> msg(N);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    for (auto& m : msg)
        m = unif(rng);
    std::vector<uint64_t> client_ct(N, 0); // word (j, w) at 16 j + w
    std::vector<int64_t> client_dm(N, 0);  // round(Delta m)
    for (int j = 0; j < blocks; ++j)
    {
        auto z = hera::keystream(key, heongpu::rubato::xof_seed(nonce, j), hp);
        for (int w = 0; w < hp.n; ++w)
        {
            const int64_t mq = (int64_t) std::llround(delta * msg[j * hp.n + w]);
            const int64_t qq = (int64_t) t;
            client_dm[j * hp.n + w] = mq;
            client_ct[j * hp.n + w] =
                (uint64_t) (((mq % qq + qq) % qq + z[w]) % qq);
        }
    }

    // ---------------------------------------------------------------------
    // Stage 2b: FV correctness down to q0
    // ---------------------------------------------------------------------
    const Data64 q0 = Qmod[0];
    double budget_q0 = 0;
    {
        auto c_pt = H_s2c.vec2poly(client_ct);
        auto x_ct = H_s2c.eval_decrypt(z_coeff_ct, c_pt);
        auto x_ckks = H_s2c.transcipher_bfv2ckks(x_ct); // mod-down to q0
        auto raw = H_s2c.decrypt_raw_q0(x_ckks, secret_key);
        size_t bad = 0;
        long double max_err = 0;
        for (size_t i = 0; i < N; ++i)
        {
            const int64_t v = raw[H_s2c.s2c_coeff_index((int) i)];
            const __int128 num = (__int128) v * (__int128) t;
            const __int128 half = q0 / 2;
            __int128 m = (num >= 0) ? (num + half) / q0 : -((-num + half) / q0);
            const long double e =
                (long double) v - (long double) m * (long double) q0 / t;
            max_err = std::max(max_err, std::fabs(e));
            bad += (m != (__int128) client_dm[i]);
        }
        const double noise_bits_q0 = std::log2((double) max_err);
        budget_q0 = std::log2((double) q0 / (2.0 * t)) - noise_bits_q0;
        std::cout << "level-0 FV noise: max |e| = 2^" << noise_bits_q0
                  << " vs q0/(2t) = 2^" << std::log2((double) q0 / (2.0 * t))
                  << " => " << budget_q0 << " bits left at q0" << std::endl;
        all_ok &= check("FV at q0: round(t/q0 * Dec(ct)) == round(Delta m) "
                        "exactly, all N coefficients",
                        bad == 0);
        if (bad != 0)
            std::cout << "  mismatching coefficients: " << bad << std::endl;
    }

    // ---------------------------------------------------------------------
    // CKKS side: HalfBoot v2 keys (scale of the input q0 * Delta / t)
    // ---------------------------------------------------------------------
    heongpu::Secretkey<SchemeCKKS> secret_key_ckks(secret_key);
    heongpu::HEKeyGenerator<SchemeCKKS> keygen_ckks(contextckks);
    heongpu::HEEncoder<SchemeCKKS> encoder_ckks(contextckks);
    heongpu::HEDecryptor<SchemeCKKS> decryptor_ckks(contextckks,
                                                    secret_key_ckks);
    heongpu::HEArithmeticOperator<SchemeCKKS> op_ckks(contextckks,
                                                      encoder_ckks);
    heongpu::Relinkey<SchemeCKKS> relin_key_ckks(contextckks);
    std::unique_ptr<heongpu::Galoiskey<SchemeCKKS>> galois_key_ckks;
    bool halfboot_ready = false;
    const double ckks_out_scale = std::pow(2.0, 45);
    int n_boot_keys = 0;
    if (run_halfboot)
    {
        try
        {
            const double t_ck = now_ms();
            keygen_ckks.generate_relin_key(relin_key_ckks, secret_key_ckks);
            const int max_level = contextckks.get_ciphertext_modulus_count() - 1;
            const int cts_piece = 4, K = 25, sine_deg = 63, double_angle = 2,
                      arcsine_deg = 7;
            const int eval_level_start = max_level - cts_piece;
            const int eval_depth =
                (int) std::ceil(std::log2(sine_deg + 1)) + double_angle +
                (int) std::ceil(std::log2(arcsine_deg + 1));
            const int out_level = eval_level_start - eval_depth - 1;
            if (out_level < 0)
                throw std::invalid_argument("chain too short for HalfBoot v2");
            heongpu::EvalModConfig eval_mod_config(
                0, eval_level_start, hera_main.get_message_ratio(), K, sine_deg,
                double_angle, arcsine_deg, 0.0);
            heongpu::BootstrappingConfigV2 boot_config(
                heongpu::EncodingMatrixConfig(
                    heongpu::LinearTransformType::SLOTS_TO_COEFFS,
                    std::max(out_level, 3)),
                eval_mod_config,
                heongpu::EncodingMatrixConfig(
                    heongpu::LinearTransformType::COEFFS_TO_SLOTS, max_level,
                    2.0, cts_piece));
            op_ckks.generate_bootstrapping_params_v2(ckks_out_scale,
                                                     boot_config);
            std::vector<int> boot_keys = op_ckks.halfbootstrapping_key_indexs();
            n_boot_keys = (int) boot_keys.size();
            galois_key_ckks =
                std::make_unique<heongpu::Galoiskey<SchemeCKKS>>(contextckks,
                                                                 boot_keys);
            keygen_ckks.generate_galois_key(*galois_key_ckks, secret_key_ckks);
            halfboot_ready = true;
            std::cout << "HalfBoot v2: Delta = q/" << t / delta
                      << ", message scale q0*Delta/t = 2^"
                      << std::log2((double) q0 / t * delta) << ", "
                      << boot_keys.size() << " rotation keys, output level "
                      << out_level << " scale 2^" << std::log2(ckks_out_scale)
                      << "; keys in " << (now_ms() - t_ck) / 1000.0 << " s"
                      << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cout << "HalfBoot setup skipped: " << e.what() << std::endl;
        }
        report_memory("after CKKS keys");
    }

    // ---------------------------------------------------------------------
    // Server pipeline (one run)
    // ---------------------------------------------------------------------
    struct Times
    {
        double rc, cipher, s2c, vec2poly, sub, moddown, halfboot;
    };
    auto run_fv = [&](Times& tm,
                      const RC& rc) -> heongpu::Ciphertext<SchemeCKKS>
    {
        double t0 = now_ms();
        heongpu::HEHERA<Scheme>::set_round_constants_staged(stages, rc);
        double t1 = now_ms();
        auto ks = heongpu::HEHERA<Scheme>::gen_stream_key_staged(stages,
                                                                stage_keys);
        if (kv[R + 1] != kv[R])
            ks = stages[R]->mod_switch_to(ks, *LS.ctx);
        double t2 = now_ms();
        auto z_ct = H_s2c.S2C_FV(ks);
        double t3 = now_ms();
        auto c_pt = H_s2c.vec2poly(client_ct);
        double t4 = now_ms();
        auto x_ct = H_s2c.eval_decrypt(z_ct, c_pt);
        double t5 = now_ms();
        auto x_ckks = H_s2c.transcipher_bfv2ckks(x_ct);
        double t6 = now_ms();
        tm.rc = t1 - t0;
        tm.cipher = t2 - t1;
        tm.s2c = t3 - t2;
        tm.vec2poly = t4 - t3;
        tm.sub = t5 - t4;
        tm.moddown = t6 - t5;
        return x_ckks;
    };
    auto run_halfboot_once = [&](heongpu::Ciphertext<SchemeCKKS>& x_ckks,
                                 Times& tm, std::vector<Complex64>* out0,
                                 std::vector<Complex64>* out1)
    {
        double t0 = now_ms();
        auto [y0, y1] = op_ckks.regular_halfbootstrapping_v2(
            x_ckks, *galois_key_ckks, relin_key_ckks);
        tm.halfboot = now_ms() - t0;
        if (out0 != nullptr)
        {
            heongpu::Plaintext<SchemeCKKS> p0(contextckks), p1(contextckks);
            decryptor_ckks.decrypt(p0, y0);
            decryptor_ckks.decrypt(p1, y1);
            encoder_ckks.decode(*out0, p0);
            encoder_ckks.decode(*out1, p1);
        }
    };

    // ---------------------------------------------------------------------
    // Stage 3: messages after HalfBoot
    // ---------------------------------------------------------------------
    double prec_mean = 0, prec_min = 0;
    if (halfboot_ready)
    {
        try
        {
            Times tm{};
            auto x_ckks =
                run_fv(tm, stages[0]->compute_round_constants(nonce, 0));
            std::vector<Complex64> y0, y1;
            run_halfboot_once(x_ckks, tm, &y0, &y1);
            // Coefficient k of X sits in slot bitrev(k) of y0 (k < N/2) or
            // y1; keystream slot i sits in coefficient s2c_coeff_index(i).
            const int half_bits = logN - 1;
            std::vector<double> coeff_val(N), slot_val(N);
            for (int s2 = 0; s2 < (int) N / 2; ++s2)
            {
                const int k = bit_reverse(s2, half_bits);
                coeff_val[k] = y0[s2].real();
                coeff_val[N / 2 + k] = y1[s2].real();
            }
            for (size_t i = 0; i < N; ++i)
                slot_val[i] = coeff_val[H_s2c.s2c_coeff_index((int) i)];
            bool natural = true;
            for (int s2 = 0; s2 < (int) N / 2 && natural; ++s2)
                natural = slot_val[s2] == y0[s2].real() &&
                          slot_val[N / 2 + s2] == y1[s2].real();
            all_ok &= check("HalfBoot slots are in BFV slot order", natural);
            double max_err = 0, sum_err = 0;
            for (size_t i = 0; i < N; ++i)
            {
                const double e = std::fabs(slot_val[i] - msg[i]);
                max_err = std::max(max_err, e);
                sum_err += e;
            }
            const double mean_err = sum_err / N;
            prec_mean = -std::log2(mean_err);
            prec_min = -std::log2(max_err);
            std::cout << "message error: mean " << mean_err << " ("
                      << prec_mean << " bits), max " << max_err << " ("
                      << prec_min << " bits)" << std::endl;
            all_ok &= check("HalfBoot output == client messages (max error < "
                            "2^-6)",
                            max_err < 1.0 / 64);
        }
        catch (const std::exception& e)
        {
            std::cout << "[FAIL] HalfBoot threw: " << e.what() << std::endl;
            all_ok = false;
            halfboot_ready = false;
        }
    }
    else
    {
        std::cout << "[SKIP] stage 3 (HalfBoot)" << std::endl;
    }

    // ---------------------------------------------------------------------
    // Stage 4: timing (round constants of nonce i+1 sampled on the CPU while
    // the GPU evaluates nonce i), memory, throughput
    // ---------------------------------------------------------------------
    {
        uint64_t counter = 0;
        auto next_rc = stages[0]->prepare_nonce_async(nonce, counter, 16);
        auto pipelined = [&](Times& tm)
        {
            RC rc = next_rc.get();
            counter += blocks;
            next_rc = stages[0]->prepare_nonce_async(nonce, counter, 16);
            return run_fv(tm, rc);
        };
        Times warm{};
        {
            auto x = pipelined(warm);
            if (halfboot_ready)
                run_halfboot_once(x, warm, nullptr, nullptr);
        }
        Times sum{};
        std::vector<Times> all;
        for (int r = 0; r < reps; ++r)
        {
            Times tm{};
            auto x = pipelined(tm);
            if (halfboot_ready)
                run_halfboot_once(x, tm, nullptr, nullptr);
            all.push_back(tm);
            sum.rc += tm.rc;
            sum.cipher += tm.cipher;
            sum.s2c += tm.s2c;
            sum.vec2poly += tm.vec2poly;
            sum.sub += tm.sub;
            sum.moddown += tm.moddown;
            sum.halfboot += tm.halfboot;
        }
        next_rc.wait();
        auto avg = [&](double v) { return v / reps; };
        std::cout << "\n--- server timing (warm-up + " << reps
                  << " reps, ms) ---\n";
        for (int r = 0; r < reps; ++r)
            std::cout << "rep " << r << ": round constants " << all[r].rc
                      << ", HERA " << all[r].cipher << ", S2C_FV "
                      << all[r].s2c << ", mod-down " << all[r].moddown
                      << ", HalfBoot " << all[r].halfboot << "\n";
        const double offline = avg(sum.rc + sum.cipher + sum.s2c);
        const double online =
            avg(sum.vec2poly + sum.sub + sum.moddown + sum.halfboot);
        std::cout << "offline: round constants " << avg(sum.rc) << ", HERA "
                  << avg(sum.cipher) << ", SlotToCoeff_FV " << avg(sum.s2c)
                  << "  => " << offline << "\n"
                  << "online : VecToPoly " << avg(sum.vec2poly) << ", C - Z "
                  << avg(sum.sub) << ", mod-down " << avg(sum.moddown)
                  << ", HalfBoot "
                  << (halfboot_ready ? std::to_string(avg(sum.halfboot))
                                     : std::string("n/a"))
                  << "  => " << online << std::endl;
        std::cout << "noise budget: key at " << kv[0] << " primes -> after HERA "
                  << budget_hera << " bits (" << kv[R] << " primes) -> at "
                  << kv[R + 1] << " primes " << budget_s2c_in
                  << " -> after S2C " << budget_s2c << " bits; at q0 after "
                     "mod-down "
                  << budget_q0 << " bits" << std::endl;
        if (halfboot_ready)
            std::cout << "HalfBoot precision: mean " << prec_mean
                      << " bits, min " << prec_min << " bits; "
                      << n_boot_keys << " CKKS rotation keys" << std::endl;
        report_memory("end");
        // All 16 words of every block carry data: N words x log t bits.
        const double bits_per_word = std::ceil(std::log2((double) hp.q));
        const double bytes_N = (double) N * bits_per_word / 8.0;
        const double sec = (offline + online) / 1000.0;
        std::cout << "throughput (offline+online"
                  << (halfboot_ready ? "" : " without HalfBoot") << ", "
                  << blocks << " blocks x 16 words x " << bits_per_word
                  << " bits = " << bytes_N / 1024.0
                  << " KiB per ciphertext, Presto Table 4 convention): "
                  << bytes_N / sec / 1024.0 << " KiB/s = "
                  << bytes_N / sec / (1024.0 * 1024.0) << " MiB/s"
                  << std::endl;
    }

    std::cout << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED")
              << std::endl;
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
