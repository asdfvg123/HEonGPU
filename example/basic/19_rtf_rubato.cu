// RtF transciphering with Rubato (Par-128L), checked stage by stage, on the
// HE parameters of the RtF reference implementation ("128af" of
// RtF-Transciphering/ckks_fv/rtf_params.go, RtFRubatoParams[0]):
//
//   Q = q0 (60 bit) | 7 x 45-bit | 22-bit (diff. scale) | 11 x 60-bit (sine)
//       | 4 x 58-bit (CoeffsToSlots),   P = 4 x 61-bit,   h = 192,
//
// hybrid key switching (KEYSWITCHING_METHOD_II, dnum = ceil(|Q| / |P|) = 6),
// one sparse ternary secret key shared by FV and CKKS.
//
//   ./19_rtf_rubato [logN] [reps] [s2c_keys] [halfboot] [s2c]
//        logN      16 for the reference parameters (default 12: same moduli,
//                  no security, for quick tests)
//        reps      timed repetitions after one warm-up (default 3)
//        s2c_keys  auto | device | host: where the SlotToCoeff giant-step
//                  Galois keys live (default auto: device if they fit)
//        halfboot  1 | 0: run stage 3 (CKKS HalfBoot) and its timing
//        s2c       fft | dense (default fft): SlotToCoeff_FV by the FFT-like
//                  factorization (rtf/s2c_fft.cuh, ~16 rotation keys) or the
//                  original dense N x N baby-step/giant-step (~sqrt(N/2) keys;
//                  s2c_keys only applies to this one)
//
// FV levels (like the reference's CipherModDown / StCModDown): by default the
// encrypted key is switched down to the first 11 primes for Rubato and the
// keystream to the first 5 for SlotToCoeff_FV and C - Z (RTF_LEVELS=11,5;
// RTF_LEVELS=off keeps all primes). Each level is its own RTF context with the
// same secret (rebuilt from its ternary coefficients) and its own keys.
//
// Keys (secret, public, relinearization, every Galois key) are cached under
// /scratch/08467/yjeon/keycache/B-<tag>/ (tag: N, hash of Q/P, key-switching
// method, dnum, h) and reused by later runs; set RTF_KEYCACHE=off to disable
// or RTF_KEYCACHE=<dir> to move it.
//
// Debug / comparison knobs: RTF_KS=1 (METHOD_I), RTF_CHAIN=old (former chain
// {60, 29 x 50} / {60}) | refP1 | refsub:<i,j,..> | bits:<Q bits>/<P bits>,
// RTF_SK=dense (h ~ N/2 key), RTF_DEBUG=1 (enc/add/mul+relin/rotation
// self-test, then exit).
//
// Stage 1: the server's homomorphic Rubato keystream equals the host
//          reference (rtf/rubato.h) word for word, and the host reference
//          matches the known-answer vectors of dev-tools/rubato_ref.py.
// Stage 2: SlotToCoeff_FV moves keystream slot i to coefficient
//          s2c_coeff_index(i) (i for dense, r*N/2 + bitrev(k) for FFT).
// Stage 2b: FV down to q0: after C - Z and the mod-down to q0, the level-0
//          ciphertext decrypts (centred mod q0, times t / q0) to exactly the
//          client's Rubato ciphertext minus keystream, round(Delta m) + AGN.
// Stage 3: end to end, the client's real messages come out of HalfBoot.
// Stage 4: server timing (offline: round constants, Rubato, SlotToCoeff_FV;
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
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::RTF;
constexpr auto SchemeCKKS = heongpu::Scheme::CKKS;
namespace rubato = heongpu::rubato;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// RtF "128af" moduli (RtFRubatoParams[0], rtf_params.go), in HEonGPU order:
// Q[0] = q0 is the last prime left after rescaling; CKKS rescales from the
// back, so the CoeffsToSlots primes are consumed first, as in Lattigo.
// ---------------------------------------------------------------------------
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

// VmRSS / VmHWM of this process, in GiB.
static std::pair<double, double> host_rss_gib()
{
    std::ifstream f("/proc/self/status");
    std::string line;
    double rss = 0, hwm = 0;
    while (std::getline(f, line))
    {
        if (line.rfind("VmRSS:", 0) == 0)
            rss = std::stod(line.substr(6)) / (1024.0 * 1024.0);
        if (line.rfind("VmHWM:", 0) == 0)
            hwm = std::stod(line.substr(6)) / (1024.0 * 1024.0);
    }
    return {rss, hwm};
}

static void report_memory(const char* tag)
{
    auto& mp = MemoryPool::instance();
    auto [rss, hwm] = host_rss_gib();
    const auto flags = std::cout.flags();
    const auto prec = std::cout.precision();
    std::cout << std::fixed << std::setprecision(2) << "[mem] " << tag
              << ": GPU pool in use " << gib(mp.get_current_device_pool_memory_usage())
              << " GiB (peak " << gib(mp.get_peak_device_pool_memory_usage())
              << "), pinned host pool in use "
              << gib(mp.get_current_host_pool_memory_usage()) << " GiB (peak "
              << gib(mp.get_peak_host_pool_memory_usage())
              << "), host RSS " << rss << " GiB (peak " << hwm << ")"
              << std::endl;
    std::cout.flags(flags);
    std::cout.precision(prec);
}

// ---------------------------------------------------------------------------
// On-disk key cache. Each file is written to a temporary name and renamed, so
// an interrupted run never leaves a half-written key behind.
// ---------------------------------------------------------------------------
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

    template <typename K, typename... A>
    void load(const std::string& name, K& key, A&&... args)
    {
        std::ifstream is;
        is.rdbuf()->pubsetbuf(buf.data(), buf.size());
        is.open(path(name), std::ios::binary);
        if (!is)
            throw std::runtime_error("key cache: cannot read " + path(name));
        key.load(is, std::forward<A>(args)...);
        if (!is)
            throw std::runtime_error("key cache: truncated " + path(name));
    }
};

// Cache-file tag of a rotation-key set (the shifts change with the BSGS
// layout, e.g. hoisted baby steps need one key per baby shift).
static std::string shift_tag(const std::vector<int>& shifts)
{
    uint64_t h = 1469598103934665603ULL;
    for (int v : shifts)
        for (int b = 0; b < 4; ++b)
        {
            h ^= (static_cast<uint32_t>(v) >> (8 * b)) & 0xff;
            h *= 1099511628211ULL;
        }
    std::ostringstream os;
    os << std::hex << std::setw(8) << std::setfill('0') << (h & 0xffffffffULL);
    return os.str();
}

// Loads `name` into key if cached, otherwise runs gen() and saves the key.
// Returns the time spent and prints where the key came from.
template <typename K, typename G, typename... A>
static void cached_key(KeyCache& cache, const std::string& name, K& key,
                       G gen, A&&... load_args)
{
    double t0 = now_ms();
    if (cache.has(name))
    {
        cache.load(name, key, std::forward<A>(load_args)...);
        std::cout << "  " << name << ": loaded from cache ("
                  << (now_ms() - t0) / 1000.0 << " s)" << std::endl;
    }
    else
    {
        gen();
        double t1 = now_ms();
        cache.save(name, key);
        std::cout << "  " << name << ": generated (" << (t1 - t0) / 1000.0
                  << " s" << (cache.enabled ? ", saved to cache in " : "")
                  << (cache.enabled
                          ? std::to_string((now_ms() - t1) / 1000.0) + " s"
                          : std::string())
                  << ")" << std::endl;
    }
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    // fast16 NTT (bit-identical radix-8 Shoup kernels for N = 2^16) on unless
    // the caller sets GPUNTT_FAST16 itself.
    setenv("GPUNTT_FAST16", "1", 0);
    const int logN = (argc > 1) ? std::atoi(argv[1]) : 12;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;
    const std::string s2c_keys_mode = (argc > 3) ? argv[3] : "auto";
    const bool run_halfboot = (argc > 4) ? (std::atoi(argv[4]) != 0) : true;
    const bool dense_s2c = (argc > 5) && std::string(argv[5]) == "dense";
    const size_t N = size_t(1) << logN;
    const rubato::Params& rp = rubato::Par128L;
    bool all_ok = true;

    // ---------------------------------------------------------------------
    // Host reference: known-answer test (key[i] = i + 1, nonce = 0, ctr = j)
    // ---------------------------------------------------------------------
    {
        std::vector<uint64_t> key(rp.n);
        for (int i = 0; i < rp.n; ++i)
            key[i] = i + 1;
        auto z0 = rubato::keystream(key, rubato::xof_seed({}, 0), rp);
        auto z1 = rubato::keystream(key, rubato::xof_seed({}, 1), rp);
        all_ok &= check("host Rubato matches dev-tools/rubato_ref.py",
                        z0[0] == 14798741 && z0[1] == 15302955 &&
                            z0[58] == 21652061 && z0[59] == 26022982 &&
                            z1[0] == 9518515 && z1[3] == 810838 &&
                            z1[59] == 4166068);
    }

    // ---------------------------------------------------------------------
    // Contexts: one modulus chain for FV and CKKS, hybrid key switching
    // ---------------------------------------------------------------------
    // Debug / comparison knobs (defaults = reference parameters):
    //   RTF_KS=1        METHOD_I (needs |P| = 1, so RTF_CHAIN=old)
    //   RTF_CHAIN=old   the former chain logQ = {60, 29 x 50}, logP = {60}
    const char* env_ks = std::getenv("RTF_KS");
    const char* env_chain = std::getenv("RTF_CHAIN");
    const auto ks_method =
        (env_ks != nullptr && std::atoi(env_ks) == 1)
            ? heongpu::keyswitching_type::KEYSWITCHING_METHOD_I
            : heongpu::keyswitching_type::KEYSWITCHING_METHOD_II;
    std::vector<Data64> Qmod = kQ128af, Pmod = kP128af;
    if (env_chain != nullptr && std::string(env_chain).rfind("refsub:", 0) == 0)
    {
        // refsub:1,2,3,8  (subset of the reference Q primes, all of P)
        std::stringstream ss(std::string(env_chain).substr(7));
        std::string tok;
        Qmod.clear();
        while (std::getline(ss, tok, ','))
            Qmod.push_back(kQ128af[std::stoi(tok)]);
    }
    if (env_chain != nullptr && std::string(env_chain) == "refP1")
    {
        Pmod = {kP128af[0]};
    }
    if (env_chain != nullptr && std::string(env_chain).rfind("bits:", 0) == 0)
    {
        // bits:60,45,45/61,61  (Q bit sizes / P bit sizes)
        std::string spec = std::string(env_chain).substr(5);
        auto slash = spec.find('/');
        auto parse = [](const std::string& x)
        {
            std::vector<int> v;
            std::stringstream ss(x);
            std::string tok;
            while (std::getline(ss, tok, ','))
                v.push_back(std::stoi(tok));
            return v;
        };
        auto qb = parse(spec.substr(0, slash));
        auto pb = parse(spec.substr(slash + 1));
        auto all = qb;
        all.insert(all.end(), pb.begin(), pb.end());
        auto primes = heongpu::generate_primes(N, all);
        Qmod.clear();
        Pmod.clear();
        for (size_t i = 0; i < primes.size(); ++i)
            (i < qb.size() ? Qmod : Pmod).push_back(primes[i].value);
    }
    if (env_chain != nullptr && std::string(env_chain) == "old")
    {
        std::vector<int> logQ_old(30, 50);
        logQ_old[0] = 60;
        auto primes = heongpu::generate_primes(N, [&] {
            auto v = logQ_old;
            v.push_back(60);
            return v;
        }());
        Qmod.clear();
        Pmod.clear();
        for (size_t i = 0; i < primes.size(); ++i)
            (i + 1 < primes.size() ? Qmod : Pmod).push_back(primes[i].value);
    }
    // Giant steps of SlotToCoeff_FV (BSGS) and the placement of their keys,
    // decided before the contexts exist so that the pinned host pool can be
    // sized (see HEONGPU_HOST_POOL_GB in MemoryPool::initialize).
    const int H = N / 2;
    const int g2 = static_cast<int>(std::ceil(std::sqrt((double) H)));
    std::set<int> giant;
    for (int s = 0; dense_s2c && s < (int) N; ++s)
    {
        const int r = (s < H) ? s : (s - H);
        const int rot = ((r - r % g2) / g2 * g2) % H;
        if (rot != 0)
            giant.insert(rot);
    }
    std::vector<int> giant_shifts(giant.begin(), giant.end());
    const size_t Qn_est = Qmod.size(), Pn_est = Pmod.size();
    const size_t dnum_est =
        (ks_method == heongpu::keyswitching_type::KEYSWITCHING_METHOD_II)
            ? (Qn_est + Pn_est - 1) / Pn_est
            : Qn_est;
    const size_t gk_bytes_est =
        2 * dnum_est * (Qn_est + Pn_est) * N * sizeof(Data64);
    const size_t s2c_key_bytes =
        dense_s2c ? (giant_shifts.size() + 1) * gk_bytes_est : 0;
    const size_t s2c_diag_bytes = dense_s2c ? N * N * sizeof(Data64) : 0;
    const size_t bsgs_ws_bytes =
        size_t(4) * g2 * 2 * N * Qn_est * sizeof(Data64);
    size_t gpu_free = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    // On the GPU if the giant-step keys, the plaintext diagonals of the S2C
    // matrix (N x N words) and the BSGS working set (~4 x g2 ciphertexts)
    // take at most 60% of the free memory; otherwise in pinned host memory,
    // prefetched one giant step ahead by the BSGS.
    bool s2c_on_device =
        (s2c_key_bytes + s2c_diag_bytes + bsgs_ws_bytes) < 0.6 * gpu_free;
    if (s2c_keys_mode == "device")
        s2c_on_device = true;
    else if (s2c_keys_mode == "host")
        s2c_on_device = false;
    if (std::getenv("HEONGPU_HOST_POOL_GB") == nullptr)
    {
        // Pinned staging only (key save/load) unless the keys live there.
        const double pool_gb =
            16.0 + (s2c_on_device ? 0.0 : 1.25 * gib(s2c_key_bytes));
        setenv("HEONGPU_HOST_POOL_GB", std::to_string(pool_gb).c_str(), 0);
    }

    heongpu::HEContext<Scheme> context(ks_method,
                                       heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(N);
    context.set_coeff_modulus_values(Qmod, Pmod);
    context.set_plain_modulus(rp.q);
    context.generate();

    heongpu::HEContext<SchemeCKKS> contextckks(ks_method,
                                               heongpu::sec_level_type::none);
    contextckks.set_poly_modulus_degree(N);
    contextckks.set_coeff_modulus_values(Qmod, Pmod);
    contextckks.generate();

    const int Qn = context.get_ciphertext_modulus_count();
    const int QPn = context.get_key_modulus_count();
    const int Pn = QPn - Qn;
    const double logQ = log2_product(Qmod), logP = log2_product(Pmod);
    // HEonGPU's hybrid key switching (METHOD_II) splits Q into digits of |P|
    // consecutive primes (RTF: KeySwitchParameterGenerator digit size |P|).
    const int dnum =
        (ks_method == heongpu::keyswitching_type::KEYSWITCHING_METHOD_II)
            ? (Qn + Pn - 1) / Pn
            : Qn;
    std::cout << "N = 2^" << logN << ", |Q| = " << Qn << " primes (logQ = "
              << logQ << "), |P| = " << Pn << " (logP = " << logP
              << "), logQP = " << logQ + logP << ", t = " << rp.q
              << ", h = " << kHammingWeight << ", METHOD_"
              << (ks_method == heongpu::keyswitching_type::KEYSWITCHING_METHOD_II
                      ? "II"
                      : "I")
              << ", dnum = " << dnum
              << std::endl;

    // ---------------------------------------------------------------------
    // Key cache
    // ---------------------------------------------------------------------
    const char* env_sk = std::getenv("RTF_SK"); // "dense": old h ~ N/2 key
    const bool dense_sk = env_sk != nullptr && std::string(env_sk) == "dense";
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
                << (dense_sk ? std::string("dense") : std::to_string(kHammingWeight));
            cache.dir = root + "/" + tag.str();
            std::error_code ec;
            fs::create_directories(cache.dir, ec);
            cache.enabled = !ec;
            std::cout << "key cache: "
                      << (cache.enabled ? cache.dir : std::string("disabled"))
                      << std::endl;
        }
        else
        {
            std::cout << "key cache: disabled" << std::endl;
        }
    }

    // ---------------------------------------------------------------------
    // Keys: one sparse ternary secret (h = 192) for FV and CKKS
    // ---------------------------------------------------------------------
    std::cout << "keys:" << std::endl;
    const double t_keys0 = now_ms();
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key =
        dense_sk ? heongpu::Secretkey<Scheme>(context)
                 : heongpu::Secretkey<Scheme>(context, kHammingWeight);
    cached_key(cache, "sk_rtf.bin", secret_key,
               [&]
               {
                   if (dense_sk)
                       keygen.generate_secret_key(secret_key);
                   else
                       keygen.generate_secret_key_v2(secret_key);
               });
    heongpu::Publickey<Scheme> public_key(context);
    cached_key(cache, "pk_rtf.bin", public_key,
               [&] { keygen.generate_public_key(public_key, secret_key); });
    heongpu::Relinkey<Scheme> relin_key(context);
    cached_key(cache, "rlk_rtf.bin", relin_key,
               [&] { keygen.generate_relin_key(relin_key, secret_key); });

    heongpu::ExecutionOptions on_device;
    on_device.set_storage_type(heongpu::storage_type::DEVICE);
    heongpu::ExecutionOptions on_host;
    on_host.set_storage_type(heongpu::storage_type::HOST);

    heongpu::Galoiskey<Scheme> galois_key_s2c(context, giant_shifts);
    const size_t gk_bytes = galois_key_s2c.get_memory_size_in_bytes();
    if (gk_bytes != gk_bytes_est)
        std::cout << "warning: Galois key size " << gk_bytes
                  << " B, estimated " << gk_bytes_est << " B" << std::endl;
    std::cout << "Galois key: " << gib(gk_bytes) * 1024.0
              << " MiB each (2 x dnum x |QP| x N words), relin key the same; "
                 "S2C giant steps: "
              << giant_shifts.size() << " + 1 keys = " << gib(s2c_key_bytes)
              << " GiB, S2C diagonals " << gib(s2c_diag_bytes)
              << " GiB, BSGS working set ~" << gib(bsgs_ws_bytes)
              << " GiB, free GPU memory " << gib(gpu_free) << " GiB => giant-step keys on "
              << (s2c_on_device ? "GPU" : "host") << " (" << s2c_keys_mode
              << ")" << std::endl;
    std::vector<int> baby_shifts;
    if (dense_s2c)
        baby_shifts = {1, 2, 4, 8, 16, 32, 64, 128};
    heongpu::Galoiskey<Scheme> galois_key_bs(context, baby_shifts);
    if (dense_s2c)
    {
        cached_key(
            cache, "gk_s2c_giant.bin", galois_key_s2c,
            [&]
            {
                keygen.generate_galois_key(galois_key_s2c, secret_key,
                                           s2c_on_device ? on_device : on_host);
            },
            s2c_on_device ? heongpu::storage_type::DEVICE
                          : heongpu::storage_type::HOST);
        cached_key(cache, "gk_s2c_baby.bin", galois_key_bs,
                   [&]
                   {
                       keygen.generate_galois_key(galois_key_bs, secret_key,
                                                  on_device);
                   });
    }
    const double t_keys1 = now_ms();
    std::cout << "FV keys ready in " << (t_keys1 - t_keys0) / 1000.0 << " s"
              << std::endl;
    report_memory("after FV keys");

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);
    heongpu::HEHERA<Scheme> hera(context, contextckks, encoder, encryptor, op,
                                 galois_key_s2c, relin_key);
    const int blocks = hera.get_block_count();
    std::cout << "N = " << N << ", Rubato blocks per ciphertext = " << blocks
              << std::endl;

    // Rubato linear layers by baby-step/giant-step (rtf/slot_lintrans.cuh):
    // one key per hoisted baby shift, the giant step, Feistel's -1.
    std::vector<int> rubato_shifts = hera.rubato_rotation_shifts();
    heongpu::Galoiskey<Scheme> galois_key_rubato(context, rubato_shifts);
    cached_key(cache, "gk_rubato_" + shift_tag(rubato_shifts) + ".bin",
               galois_key_rubato,
               [&]
               {
                   keygen.generate_galois_key(galois_key_rubato, secret_key,
                                              on_device);
               });
    std::cout << "Rubato rotation keys (" << rubato_shifts.size() << ", "
              << gib(rubato_shifts.size() * gk_bytes) << " GiB):";
    for (int x : rubato_shifts)
        std::cout << " " << x;
    std::cout << std::endl;

    // ---------------------------------------------------------------------
    // FV levels: Rubato on the first k_mid primes, SlotToCoeff / C - Z on the
    // first k_low (same P, same secret).
    // ---------------------------------------------------------------------
    int k_mid = 11, k_low = 5;
    if (const char* lv = std::getenv("RTF_LEVELS"))
    {
        if (std::string(lv) == "off")
            k_mid = k_low = 0;
        else if (std::sscanf(lv, "%d,%d", &k_mid, &k_low) != 2)
            k_mid = k_low = 0;
    }
    const bool leveled = !dense_s2c && k_mid > 0 && k_low > 0 &&
                         k_low <= k_mid && k_mid < (int) Qmod.size();
    struct Level
    {
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
    };
    std::vector<int> sk_ternary;
    auto make_level = [&](int k)
    {
        auto L = std::make_unique<Level>();
        std::vector<Data64> q(Qmod.begin(), Qmod.begin() + k);
        L->ctx = std::make_unique<heongpu::HEContext<Scheme>>(
            ks_method, heongpu::sec_level_type::none);
        L->ctx->set_poly_modulus_degree(N);
        L->ctx->set_coeff_modulus_values(q, Pmod);
        L->ctx->set_plain_modulus(rp.q);
        L->ctx->generate();
        if (sk_ternary.empty())
            sk_ternary = secret_key.ternary(context);
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
        std::vector<int> none;
        L->gk_none = std::make_unique<heongpu::Galoiskey<Scheme>>(*L->ctx, none);
        L->hera = std::make_unique<heongpu::HEHERA<Scheme>>(
            *L->ctx, contextckks, *L->encoder, *L->encryptor, *L->op,
            *L->gk_none, *L->rlk);
        return L;
    };
    std::unique_ptr<Level> mid, low;
    if (leveled)
    {
        const double t0 = now_ms();
        mid = make_level(k_mid);
        std::vector<int> sh = mid->hera->rubato_rotation_shifts();
        mid->gk = std::make_unique<heongpu::Galoiskey<Scheme>>(*mid->ctx, sh);
        mid->keygen->generate_galois_key(*mid->gk, *mid->sk, on_device);
        mid->hera->set_galois_key_hera(*mid->gk);
        low = make_level(k_low);
        low->hera->gen_FV_S2C_FFT(4);
        std::vector<int> fs = low->hera->s2c_fft_rotation_shifts();
        low->gk = std::make_unique<heongpu::Galoiskey<Scheme>>(*low->ctx, fs);
        low->keygen->generate_galois_key(*low->gk, *low->sk, on_device);
        low->hera->set_galois_key_s2c(*low->gk);
        std::cout << "FV levels: Rubato on " << k_mid << " primes (logQ "
                  << log2_product(std::vector<Data64>(Qmod.begin(),
                                                      Qmod.begin() + k_mid))
                  << "), SlotToCoeff/C-Z on " << k_low << " primes (logQ "
                  << log2_product(std::vector<Data64>(Qmod.begin(),
                                                      Qmod.begin() + k_low))
                  << "); contexts + keys in " << (now_ms() - t0) / 1000.0
                  << " s" << std::endl;
    }
    // Engines: Rubato and SlotToCoeff / C - Z / mod-down
    heongpu::HEHERA<Scheme>& H_rub = leveled ? *mid->hera : hera;
    heongpu::HEHERA<Scheme>& H_s2c = leveled ? *low->hera : hera;
    heongpu::HEDecryptor<Scheme>& dec_rub = leveled ? *mid->decryptor : decryptor;
    heongpu::HEEncoder<Scheme>& enc_rub = leveled ? *mid->encoder : encoder;
    heongpu::HEDecryptor<Scheme>& dec_s2c = leveled ? *low->decryptor : decryptor;
    heongpu::HEContext<Scheme>& ctx_s2c = leveled ? *low->ctx : context;

    // FFT SlotToCoeff_FV: a handful of rotation keys plus the column rotation.
    std::unique_ptr<heongpu::Galoiskey<Scheme>> galois_key_fft;
    if (!dense_s2c && !leveled)
    {
        const double t0 = now_ms();
        hera.gen_FV_S2C_FFT(4);
        std::cout << "gen_FV_S2C_FFT: " << (now_ms() - t0) / 1000.0 << " s"
                  << std::endl;
        std::vector<int> fft_shifts = hera.s2c_fft_rotation_shifts();
        galois_key_fft =
            std::make_unique<heongpu::Galoiskey<Scheme>>(context, fft_shifts);
        cached_key(cache, "gk_s2c_fft4_" + shift_tag(fft_shifts) + ".bin",
                   *galois_key_fft,
                   [&]
                   {
                       keygen.generate_galois_key(*galois_key_fft, secret_key,
                                                  on_device);
                   });
        hera.set_galois_key_s2c(*galois_key_fft);
        std::cout << "SlotToCoeff_FV: FFT, rotation keys ("
                  << fft_shifts.size() << " + column, "
                  << gib((fft_shifts.size() + 1) * gk_bytes) << " GiB):";
        for (int x : fft_shifts)
            std::cout << " " << x;
        std::cout << " + column" << std::endl;
    }

    // ---------------------------------------------------------------------
    // Client: symmetric key k, FV-encrypted as Concat(k) (k in every block)
    // ---------------------------------------------------------------------
    std::mt19937_64 rng(2026);
    std::vector<uint64_t> key(rp.n);
    for (auto& k : key)
        k = rng() % rp.q;
    std::vector<uint64_t> key_slots(N);
    for (size_t i = 0; i < N; ++i)
        key_slots[i] = key[i % rp.n];
    heongpu::Plaintext<Scheme> key_pt(context);
    encoder.encode(key_pt, key_slots);
    heongpu::Ciphertext<Scheme> key_ct(context);
    encryptor.encrypt(key_ct, key_pt);
    std::cout << "noise budget of the encrypted key: "
              << decryptor.remainder_noise_budget(key_ct) << " bits (logQ = "
              << logQ << ")" << std::endl;

    if (std::getenv("RTF_DEBUG") != nullptr)
    {
        std::vector<uint64_t> a(N), b(N);
        for (size_t i = 0; i < N; ++i)
        {
            a[i] = rng() % rp.q;
            b[i] = rng() % rp.q;
        }
        heongpu::Plaintext<Scheme> pa(context), pb(context);
        encoder.encode(pa, a);
        encoder.encode(pb, b);
        heongpu::Ciphertext<Scheme> ca(context), cb(context), cc(context);
        encryptor.encrypt(ca, pa);
        encryptor.encrypt(cb, pb);
        auto dec = [&](heongpu::Ciphertext<Scheme>& c)
        {
            heongpu::Plaintext<Scheme> p(context);
            decryptor.decrypt(p, c);
            std::vector<uint64_t> v;
            encoder.decode(v, p);
            return v;
        };
        auto cmp = [&](const char* what, const std::vector<uint64_t>& got,
                       auto f)
        {
            size_t bad = 0;
            for (size_t i = 0; i < N; ++i)
                bad += got[i] != f(i);
            std::cout << "  debug " << what << ": " << bad << " bad"
                      << std::endl;
        };
        cmp("enc/dec", dec(ca), [&](size_t i) { return a[i]; });
        op.add(ca, cb, cc);
        cmp("add", dec(cc), [&](size_t i) { return (a[i] + b[i]) % rp.q; });
        op.multiply(ca, cb, cc);
        cmp("mul (no relin)", dec(cc), [&](size_t i)
            { return (uint64_t) ((__uint128_t) a[i] * b[i] % rp.q); });
        op.relinearize_inplace(cc, relin_key);
        std::cout << "  budget after mul+relin "
                  << decryptor.remainder_noise_budget(cc) << std::endl;
        cmp("mul+relin", dec(cc), [&](size_t i)
            { return (uint64_t) ((__uint128_t) a[i] * b[i] % rp.q); });
        heongpu::Ciphertext<Scheme> cr(context);
        op.rotate_rows(ca, cr, galois_key_rubato, 1);
        std::cout << "  budget after rot1 "
                  << decryptor.remainder_noise_budget(cr) << std::endl;
        cmp("rot 1", dec(cr), [&](size_t i)
            {
                size_t row = i / H, c = i % H;
                return a[row * H + (c + 1) % H];
            });
        op.rotate_columns(ca, cr, galois_key_rubato);
        cmp("rot cols", dec(cr), [&](size_t i) { return a[(i + H) % N]; });
        return 0;
    }

    const std::array<uint8_t, 8> nonce = {0x52, 0x75, 0x62, 0x61,
                                          0x74, 0x6f, 0x00, 0x01};

    // ---------------------------------------------------------------------
    // Stage 1: homomorphic keystream vs host keystream
    // ---------------------------------------------------------------------
    hera.set_galois_key_hera(galois_key_rubato);
    // The encrypted symmetric key is switched to the Rubato level once.
    heongpu::Ciphertext<Scheme> key_ct_rub =
        leveled ? hera.mod_switch_to(key_ct, *mid->ctx) : key_ct;
    if (leveled)
        std::cout << "noise budget of the key at " << k_mid << " primes: "
                  << dec_rub.remainder_noise_budget(key_ct_rub) << " bits"
                  << std::endl;
    H_rub.set_nonce(nonce, 0);
    auto ks_ct = H_rub.gen_stream_key(key_ct_rub);
    const int budget_rubato = dec_rub.remainder_noise_budget(ks_ct);
    std::cout << "noise budget after Rubato: " << budget_rubato << " bits"
              << std::endl;

    heongpu::Plaintext<Scheme> ks_pt(leveled ? *mid->ctx : context);
    dec_rub.decrypt(ks_pt, ks_ct);
    std::vector<uint64_t> ks_he;
    enc_rub.decode(ks_he, ks_pt);

    size_t bad_words = 0, bad_trunc = 0;
    for (int j = 0; j < blocks; ++j)
    {
        auto z = rubato::keystream(key, rubato::xof_seed(nonce, j), rp);
        for (int i = 0; i < rp.n; ++i)
        {
            const uint64_t got = ks_he[j * rp.n + i];
            if (i < rp.l)
                bad_words += (got != z[i]);
            else
                bad_trunc += (got != 0);
        }
    }
    all_ok &= check("HE keystream == host keystream (all blocks, 60 words)",
                    bad_words == 0);
    all_ok &= check("truncated words 60..63 are zero", bad_trunc == 0);
    if (bad_words != 0 || bad_trunc != 0)
        std::cout << "  mismatching words: " << bad_words
                  << ", nonzero truncated: " << bad_trunc << std::endl;

    // ---------------------------------------------------------------------
    // Stage 2: SlotToCoeff_FV
    // ---------------------------------------------------------------------
    if (dense_s2c)
    {
        hera.set_galois_key_bs(galois_key_bs);
        double t0 = now_ms();
        hera.gen_FV_S2C_Matrix();
        std::cout << "gen_FV_S2C_Matrix: " << (now_ms() - t0) / 1000.0 << " s"
                  << std::endl;
    }
    heongpu::Ciphertext<Scheme> ks_s2c =
        leveled ? H_rub.mod_switch_to(ks_ct, *low->ctx) : ks_ct;
    if (leveled)
        std::cout << "noise budget of the keystream at " << k_low
                  << " primes: " << dec_s2c.remainder_noise_budget(ks_s2c)
                  << " bits" << std::endl;
    auto z_coeff_ct = H_s2c.S2C_FV(ks_s2c);
    const int budget_s2c = dec_s2c.remainder_noise_budget(z_coeff_ct);
    std::cout << "noise budget after SlotToCoeff_FV: " << budget_s2c
              << " bits" << std::endl;
    {
        heongpu::Plaintext<Scheme> pt(ctx_s2c);
        dec_s2c.decrypt(pt, z_coeff_ct);
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
    // Client: real messages m in (-1, 1), c = round(Delta m) + z (with AGN)
    // ---------------------------------------------------------------------
    const double delta = hera.get_message_scaling();
    const uint64_t t = rp.q;
    const int msg_count = blocks * rp.l;
    std::vector<double> msg(msg_count);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    for (auto& m : msg)
        m = unif(rng);
    std::vector<uint64_t> client_ct(N, 0); // word (j, w) at position 64j + w
    std::vector<int64_t> client_dm(N, 0);  // round(Delta m), 0 if truncated
    for (int j = 0; j < blocks; ++j)
    {
        auto z = rubato::keystream(key, rubato::xof_seed(nonce, j), rp, &rng);
        for (int w = 0; w < rp.l; ++w)
        {
            const int64_t mq = (int64_t) std::llround(delta * msg[j * rp.l + w]);
            const int64_t qq = (int64_t) t;
            client_dm[j * rp.n + w] = mq;
            client_ct[j * rp.n + w] =
                (uint64_t) (((mq % qq + qq) % qq + z[w]) % qq);
        }
    }

    // ---------------------------------------------------------------------
    // Stage 2b: FV correctness down to q0
    // ---------------------------------------------------------------------
    const Data64 q0 = Qmod[0];
    double noise_bits_q0 = 0, budget_q0 = 0;
    {
        auto c_pt = H_s2c.vec2poly(client_ct);
        auto x_ct = H_s2c.eval_decrypt(z_coeff_ct, c_pt);
        auto x_ckks = H_s2c.transcipher_bfv2ckks(x_ct); // mod-down to q0 inside
        auto raw = H_s2c.decrypt_raw_q0(x_ckks, secret_key);

        size_t bad = 0;
        long double max_err = 0;
        int64_t max_agn = 0;
        for (size_t i = 0; i < N; ++i)
        {
            // Slot i's word sits in coefficient s2c_coeff_index(i).
            const int64_t v = raw[H_s2c.s2c_coeff_index((int) i)];
            // v = (q0 / t) * M + e (mod q0, centred) -> M = round(v t / q0)
            const __int128 num = (__int128) v * (__int128) t;
            const __int128 half = q0 / 2;
            __int128 m = (num >= 0) ? (num + half) / q0 : -((-num + half) / q0);
            const long double e =
                (long double) v - (long double) m * (long double) q0 / t;
            max_err = std::max(max_err, std::fabs(e));
            int64_t got = (int64_t) (((m % (__int128) t) + t) % t);
            // Expected: client ciphertext minus noiseless keystream.
            int64_t want = (int64_t) ((client_ct[i] + t - ks_he[i]) % t);
            bad += (got != want);
            int64_t centred = (want > (int64_t) t / 2) ? want - (int64_t) t : want;
            max_agn = std::max<int64_t>(max_agn,
                                        std::llabs(centred - client_dm[i]));
        }
        noise_bits_q0 = std::log2((double) max_err);
        budget_q0 = std::log2((double) q0 / (2.0 * t)) - noise_bits_q0;
        std::cout << "level-0 FV noise: max |e| = 2^" << noise_bits_q0
                  << " vs q0/(2t) = 2^" << std::log2((double) q0 / (2.0 * t))
                  << " => " << budget_q0 << " bits left at q0; |AGN| <= "
                  << max_agn << std::endl;
        all_ok &= check("FV at q0: round(t/q0 * Dec(ct)) == client ct - "
                        "keystream (round(Delta m) + AGN), all N coefficients",
                        bad == 0);
        if (bad != 0)
            std::cout << "  mismatching coefficients: " << bad << std::endl;
    }

    // ---------------------------------------------------------------------
    // CKKS side (stage 3): HalfBoot keys. The transciphered ciphertext has
    // scale q0 * Delta / t.
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
    // HalfBoot v2 (Bossuat et al. EvalMod + arcsine) with the RtF 128af
    // setting: K 25, sine degree 63, double angle 2, arcsine 7, message ratio
    // q0/Delta' = 16, output scale 2^45; CoeffToSlot on the top 4 primes.
    const double ckks_out_scale = std::pow(2.0, 45);
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
                throw std::invalid_argument(
                    "chain too short for HalfBoot v2 (out level " +
                    std::to_string(out_level) + ")");
            heongpu::EvalModConfig eval_mod_config(
                0, eval_level_start, hera.get_message_ratio(), K, sine_deg,
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
        double rc, rubato, s2c, vec2poly, sub, moddown, halfboot;
    };
    // rc: round constants prepared on the CPU (pipelined runs), or nullptr
    // to sample them here (set_nonce).
    using RC = heongpu::HEHERA<Scheme>::RoundConstants;
    auto run_fv = [&](Times& tm,
                      const RC* rc = nullptr) -> heongpu::Ciphertext<SchemeCKKS>
    {
        double t0 = now_ms();
        if (rc != nullptr)
            H_rub.set_round_constants(*rc);
        else
            H_rub.set_nonce(nonce, 0);
        double t1 = now_ms();
        auto ks = H_rub.gen_stream_key(key_ct_rub);
        if (leveled)
            ks = H_rub.mod_switch_to(ks, *low->ctx);
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
        tm.rubato = t2 - t1;
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
    if (halfboot_ready)
    {
        try
        {
            Times tm{};
            auto x_ckks = run_fv(tm);
            std::vector<Complex64> y0, y1;
            run_halfboot_once(x_ckks, tm, &y0, &y1);
            // HalfBoot leaves coefficient k of X in slot bitrev(k) of y0
            // (k < N/2) or y1 (k >= N/2), real part. Keystream slot i sits in
            // coefficient s2c_coeff_index(i); with the FFT S2C that is
            // r*N/2 + bitrev(k) for slot (r, k), so the words come out in
            // slot order.
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
            if (!dense_s2c)
            {
                bool natural = true;
                for (int s2 = 0; s2 < (int) N / 2 && natural; ++s2)
                    natural = slot_val[s2] == y0[s2].real() &&
                              slot_val[N / 2 + s2] == y1[s2].real();
                all_ok &= check("HalfBoot slots are in BFV slot order",
                                natural);
            }
            double max_err = 0, sum_err = 0, max_pad = 0;
            for (int j = 0; j < blocks; ++j)
                for (int w = 0; w < rp.n; ++w)
                {
                    const double v = slot_val[j * rp.n + w];
                    if (w < rp.l)
                    {
                        const double e = std::fabs(v - msg[j * rp.l + w]);
                        max_err = std::max(max_err, e);
                        sum_err += e;
                    }
                    else
                        max_pad = std::max(max_pad, std::fabs(v));
                }
            const double mean_err = sum_err / msg_count;
            std::cout << "message error: mean " << mean_err << " ("
                      << -std::log2(mean_err) << " bits), max " << max_err
                      << " (" << -std::log2(max_err)
                      << " bits); truncated positions max |v| " << max_pad
                      << std::endl;
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
    // Stage 4: timing, memory, throughput
    // ---------------------------------------------------------------------
    {
        // Pipelined: while the GPU evaluates nonce i, a CPU thread samples the
        // round constants of nonce i+1 (counter advanced by one ciphertext's
        // blocks), so "round constants" below is only the wait + encoding.
        uint64_t counter = 0;
        auto next_rc = H_rub.prepare_nonce_async(nonce, counter, 16);
        auto pipelined = [&](Times& tm)
        {
            RC rc = next_rc.get();
            counter += blocks;
            next_rc = H_rub.prepare_nonce_async(nonce, counter, 16);
            return run_fv(tm, &rc);
        };
        Times warm{};
        {
            auto x = pipelined(warm); // warm-up
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
            sum.rubato += tm.rubato;
            sum.s2c += tm.s2c;
            sum.vec2poly += tm.vec2poly;
            sum.sub += tm.sub;
            sum.moddown += tm.moddown;
            sum.halfboot += tm.halfboot;
        }
        auto avg = [&](double v) { return v / reps; };
        std::cout << "\n--- server timing (warm-up + " << reps
                  << " reps, ms) ---\n";
        for (int r = 0; r < reps; ++r)
            std::cout << "rep " << r << ": set_nonce " << all[r].rc
                      << ", gen_stream_key " << all[r].rubato << ", S2C_FV "
                      << all[r].s2c << ", mod-down " << all[r].moddown
                      << ", HalfBoot " << all[r].halfboot << "\n";
        const double offline = avg(sum.rc + sum.rubato + sum.s2c);
        const double online =
            avg(sum.vec2poly + sum.sub + sum.moddown + sum.halfboot);
        std::cout << "offline: round constants " << avg(sum.rc)
                  << ", Rubato " << avg(sum.rubato) << ", SlotToCoeff_FV "
                  << avg(sum.s2c) << "  => " << offline << "\n"
                  << "online : VecToPoly " << avg(sum.vec2poly) << ", C - Z "
                  << avg(sum.sub) << ", mod-down " << avg(sum.moddown)
                  << ", HalfBoot "
                  << (halfboot_ready ? std::to_string(avg(sum.halfboot))
                                     : std::string("n/a"))
                  << "  => " << online << std::endl;
        std::cout << "noise budget: fresh key ct -> after Rubato "
                  << budget_rubato << " bits -> after S2C " << budget_s2c
                  << " bits (logQ = " << logQ << "); at q0 after mod-down "
                  << budget_q0 << " bits" << std::endl;
        report_memory("end");
        // Throughput: l = 60 words of log q = 25 bits per 64-word block.
        const double bits_per_word = std::ceil(std::log2((double) rp.q));
        const double bytes_l = (double) blocks * rp.l * bits_per_word / 8.0;
        const double sec = (offline + online) / 1000.0;
        const double bytes_N = (double) N * bits_per_word / 8.0;
        std::cout << "throughput (offline+online"
                  << (halfboot_ready ? "" : " without HalfBoot") << ", "
                  << blocks << " blocks x " << rp.l << " words x "
                  << bits_per_word << " bits = " << bytes_l
                  << " B): " << bytes_l / sec / 1000.0 << " KB/s\n"
                  << "throughput, Presto Table 4 convention (N words x "
                  << bits_per_word << " bits = " << bytes_N / 1024.0
                  << " KiB per ciphertext): " << bytes_N / sec / 1024.0
                  << " KiB/s" << std::endl;
    }

    std::cout << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED")
              << std::endl;
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
