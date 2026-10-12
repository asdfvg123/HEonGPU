// Bit-identity of the GPUNTT_FAST16 path (N = 2^16 radix-8 Shoup kernels)
// against the generic merge kernels, through every public GPU-NTT entry point
// HEonGPU uses: RNS GPU_NTT / GPU_NTT_Inplace, GPU_NTT_Modulus_Ordered(_Inplace),
// GPU_NTT_Poly_Ordered(_Inplace) and the single-prime GPU_NTT(_Inplace).
//
// Each case runs the same call twice on identical device buffers, once with
// gpuntt::fast16::set_enabled(0) (generic kernels) and once with
// set_enabled(1), and compares the whole output buffer word for word.

#include "heongpu.cuh"
#include "ntt.cuh"
#include "ntt_fast16.cuh"
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace
{
    using u64 = Data64;
    constexpr int LOGN = 16;
    constexpr size_t N = size_t(1) << LOGN;

    size_t g_cases = 0;
    size_t g_words = 0;

#define CK(x) ASSERT_EQ((x), cudaSuccess)

    u64 mulmod(u64 a, u64 b, u64 m)
    {
        return (u64) ((unsigned __int128) a * b % m);
    }
    u64 powmod(u64 a, u64 e, u64 m)
    {
        u64 r = 1;
        a %= m;
        while (e)
        {
            if (e & 1)
                r = mulmod(r, a, m);
            a = mulmod(a, a, m);
            e >>= 1;
        }
        return r;
    }
    bool is_prime(u64 n)
    {
        if (n < 2)
            return false;
        for (u64 p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
                      23ull, 29ull, 31ull, 37ull})
        {
            if (n % p == 0)
                return n == p;
        }
        u64 d = n - 1;
        int s = 0;
        while ((d & 1) == 0)
        {
            d >>= 1;
            s++;
        }
        for (u64 a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
                      23ull, 29ull, 31ull, 37ull})
        {
            u64 x = powmod(a, d, n);
            if (x == 1 || x == n - 1)
                continue;
            bool comp = true;
            for (int r = 1; r < s; r++)
            {
                x = mulmod(x, x, n);
                if (x == n - 1)
                {
                    comp = false;
                    break;
                }
            }
            if (comp)
                return false;
        }
        return true;
    }

    // `count` primes q = 1 (mod 2^17) just below 2^bits, distinct from `taken`
    std::vector<u64> ntt_primes(int bits, int count, std::vector<u64>& taken)
    {
        std::vector<u64> out;
        const u64 step = u64(1) << (LOGN + 1);
        u64 c = ((u64(1) << bits) / step) * step + 1;
        while ((int) out.size() < count && c > step)
        {
            c -= step;
            if (is_prime(c) &&
                std::find(taken.begin(), taken.end(), c) == taken.end())
            {
                out.push_back(c);
                taken.push_back(c);
            }
        }
        return out;
    }

    // The RtF chain primes named in the task plus a 61-bit P prime.
    const std::vector<u64> kNamed = {0x10000000006e0001ull, 0x400000000360001ull,
                                     0x2000000a0001ull, 0x2a0001ull,
                                     0x1fffffffffe00001ull};

    std::vector<u64> chain28()
    {
        std::vector<u64> taken = kNamed;
        std::vector<u64> q = kNamed;
        const int bits[] = {60, 58, 45, 22, 61, 50, 36, 55, 40, 30, 59, 52};
        int i = 0;
        while (q.size() < 28)
        {
            auto p = ntt_primes(bits[i % 12], 1, taken);
            if (!p.empty())
                q.push_back(p[0]);
            i++;
        }
        return q;
    }

    template <typename T> struct DevBuf
    {
        T* p = nullptr;
        size_t n = 0;
        explicit DevBuf(size_t count) : n(count)
        {
            if (cudaMalloc(&p, count * sizeof(T)) != cudaSuccess)
                p = nullptr;
        }
        DevBuf(const std::vector<T>& h) : DevBuf(h.size()) { upload(h); }
        ~DevBuf() { cudaFree(p); }
        void upload(const std::vector<T>& h)
        {
            cudaMemcpy(p, h.data(), h.size() * sizeof(T),
                       cudaMemcpyHostToDevice);
        }
        std::vector<T> download() const
        {
            std::vector<T> h(n);
            cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
            return h;
        }
        DevBuf(const DevBuf&) = delete;
        DevBuf& operator=(const DevBuf&) = delete;
    };

    struct Chain
    {
        std::vector<u64> q;
        std::vector<Modulus64> mods;
        DevBuf<Modulus64> modulus;
        DevBuf<Root64> ntt_table, intt_table;
        DevBuf<Ninverse64> n_inverse;

        static std::vector<Modulus64> to_mod(const std::vector<u64>& q)
        {
            std::vector<Modulus64> m;
            for (u64 v : q)
                m.push_back(Modulus64(v));
            return m;
        }
        explicit Chain(const std::vector<u64>& primes)
            : q(primes), mods(to_mod(primes)), modulus(mods),
              ntt_table(build(true)), intt_table(build(false)),
              n_inverse(heongpu::generate_n_inverse(N, mods))
        {
        }
        std::vector<Root64> build(bool fwd) const
        {
            auto psi = heongpu::generate_primitive_root_of_unity(N, mods);
            return fwd ? heongpu::generate_ntt_table(psi, mods, LOGN)
                       : heongpu::generate_intt_table(psi, mods, LOGN);
        }
    };

    gpuntt::ntt_rns_configuration<u64> rns_cfg(const Chain& c, bool fwd,
                                               int offset = 0)
    {
        gpuntt::ntt_rns_configuration<u64> cfg = {
            .n_power = LOGN,
            .ntt_type = fwd ? gpuntt::FORWARD : gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = c.n_inverse.p + offset,
            .stream = 0};
        return cfg;
    }

    // random limbs, limb z reduced modulo prime_of(z); `pattern` 0 random,
    // 1 all q-1, 2 all 0, 3 alternating 0 / q-1
    std::vector<u64> make_input(size_t limbs,
                                const std::function<u64(size_t)>& prime_of,
                                std::mt19937_64& rng, int pattern = 0)
    {
        std::vector<u64> h(limbs * N);
        for (size_t z = 0; z < limbs; z++)
        {
            const u64 q = prime_of(z);
            for (size_t i = 0; i < N; i++)
            {
                u64 v;
                switch (pattern)
                {
                    case 1:
                        v = q - 1;
                        break;
                    case 2:
                        v = 0;
                        break;
                    case 3:
                        v = (i & 1) ? q - 1 : 0;
                        break;
                    default:
                        v = rng() % q;
                }
                h[z * N + i] = v;
            }
        }
        return h;
    }

    // Runs `call(in, out)` with the generic and the fast kernels on copies of
    // the same buffers and compares the whole `out` buffer (and `in`).
    void compare(const std::string& name, const std::vector<u64>& in_h,
                 size_t out_limbs, bool inplace,
                 const std::function<void(u64*, u64*)>& call)
    {
        std::vector<u64> fill(out_limbs * N);
        for (size_t i = 0; i < fill.size(); i++)
            fill[i] = 0x5a5a5a5a00000000ull + i; // untouched words must match

        std::vector<u64> res[2], res_in[2];
        for (int fast = 0; fast < 2; fast++)
        {
            gpuntt::fast16::set_enabled(fast);
            DevBuf<u64> in(in_h);
            DevBuf<u64> out(inplace ? 1 : out_limbs * N);
            if (!inplace)
                out.upload(fill);
            ASSERT_NE(in.p, nullptr);
            ASSERT_NE(out.p, nullptr);
            call(in.p, inplace ? in.p : out.p);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            res[fast] = inplace ? in.download() : out.download();
            res_in[fast] = in.download();
        }
        gpuntt::fast16::set_enabled(-1);

        size_t bad = 0, first = 0;
        for (size_t i = 0; i < res[0].size(); i++)
        {
            if (res[0][i] != res[1][i])
            {
                if (bad == 0)
                    first = i;
                bad++;
            }
        }
        EXPECT_EQ(bad, 0u)
            << name << ": " << bad << " / " << res[0].size()
            << " words differ, first at limb " << first / N << " index "
            << first % N << " (generic " << res[0][first] << ", fast "
            << res[1][first] << ")";
        EXPECT_TRUE(res_in[0] == res_in[1]) << name << ": input buffer differs";
        // the fast path must actually have changed the output
        if (!inplace)
        {
            size_t changed = 0;
            for (size_t i = 0; i < fill.size(); i++)
                changed += res[1][i] != fill[i];
            EXPECT_GT(changed, 0u) << name;
        }
        g_cases++;
        g_words += res[0].size();
    }

    struct Summary : public ::testing::Environment
    {
        void TearDown() override
        {
            std::printf("[fast16] %zu bit-identity cases, %zu words compared "
                        "(%.1f M)\n",
                        g_cases, g_words, g_words / 1e6);
        }
    };
    ::testing::Environment* const g_env =
        ::testing::AddGlobalTestEnvironment(new Summary);

} // namespace

TEST(NTTFast16, RNS_BatchedAndInplace)
{
    cudaSetDevice(0);
    std::mt19937_64 rng(1);
    Chain named(kNamed);
    Chain big(chain28());

    struct Cfg
    {
        Chain* c;
        int batch, mod_count, offset;
    };
    std::vector<Cfg> cfgs = {
        {&named, 1, 1, 0},  {&named, 5, 5, 0},  {&named, 10, 5, 0},
        {&named, 3, 5, 0},  {&named, 7, 3, 0},  {&named, 2, 1, 3},
        {&named, 4, 2, 3},  {&big, 24, 24, 0},  {&big, 48, 24, 0},
        {&big, 28, 28, 0},  {&big, 56, 28, 0},  {&big, 12, 12, 16},
        {&big, 1, 1, 27},   {&big, 2, 1, 27},   {&big, 17, 28, 0}};
    for (const Cfg& k : cfgs)
    {
        Chain& c = *k.c;
        auto prime_of = [&](size_t z)
        { return c.q[k.offset + z % k.mod_count]; };
        for (int fwd = 1; fwd >= 0; fwd--)
        {
            for (int inplace = 0; inplace < 2; inplace++)
            {
                auto in_h = make_input(k.batch, prime_of, rng);
                char name[160];
                std::snprintf(name, sizeof(name),
                              "RNS %s %s q%zu batch=%d mod_count=%d off=%d",
                              fwd ? "NTT" : "INTT",
                              inplace ? "Inplace" : "out-of-place", c.q.size(),
                              k.batch, k.mod_count, k.offset);
                auto cfg = rns_cfg(c, fwd, k.offset);
                Root64* tab = (fwd ? c.ntt_table.p : c.intt_table.p) +
                              ((size_t) k.offset << LOGN);
                Modulus64* mod = c.modulus.p + k.offset;
                compare(name, in_h, k.batch, inplace,
                        [&](u64* in, u64* out)
                        {
                            if (inplace)
                                gpuntt::GPU_NTT_Inplace(in, tab, mod, cfg,
                                                        k.batch, k.mod_count);
                            else
                                gpuntt::GPU_NTT(in, out, tab, mod, cfg,
                                                k.batch, k.mod_count);
                        });
            }
        }
    }
}

TEST(NTTFast16, RNS_EdgeInputs)
{
    cudaSetDevice(0);
    std::mt19937_64 rng(2);
    Chain named(kNamed);
    for (int pattern = 1; pattern <= 3; pattern++)
    {
        for (int fwd = 1; fwd >= 0; fwd--)
        {
            auto prime_of = [&](size_t z) { return named.q[z % 5]; };
            auto in_h = make_input(10, prime_of, rng, pattern);
            auto cfg = rns_cfg(named, fwd);
            Root64* tab = fwd ? named.ntt_table.p : named.intt_table.p;
            compare(std::string("edge pattern ") + std::to_string(pattern) +
                        (fwd ? " NTT" : " INTT"),
                    in_h, 10, false,
                    [&](u64* in, u64* out)
                    {
                        gpuntt::GPU_NTT(in, out, tab, named.modulus.p, cfg, 10,
                                        5);
                    });
        }
    }
}

TEST(NTTFast16, ModulusOrdered)
{
    cudaSetDevice(0);
    std::mt19937_64 rng(3);
    Chain named(kNamed);
    Chain big(chain28());

    struct Cfg
    {
        Chain* c;
        int batch;
        std::vector<int> order;
    };
    std::vector<int> sub12(28);
    std::iota(sub12.begin(), sub12.end(), 0);
    std::shuffle(sub12.begin(), sub12.end(), rng);
    sub12.resize(12);
    std::vector<int> all28(28);
    std::iota(all28.begin(), all28.end(), 0);
    std::reverse(all28.begin(), all28.end());
    std::vector<Cfg> cfgs = {{&named, 6, {4, 0, 2}},
                             {&named, 2, {3, 1, 4}}, // batch < mod_count
                             {&named, 5, {0, 1, 2, 3, 4}},
                             {&named, 1, {4}},
                             {&big, 24, sub12},
                             {&big, 56, all28},
                             {&big, 3, {27}}};
    for (const Cfg& k : cfgs)
    {
        Chain& c = *k.c;
        const int mod_count = (int) k.order.size();
        DevBuf<int> order(k.order);
        auto prime_of = [&](size_t z) { return c.q[k.order[z % mod_count]]; };
        for (int fwd = 1; fwd >= 0; fwd--)
        {
            for (int inplace = 0; inplace < 2; inplace++)
            {
                auto in_h = make_input(k.batch, prime_of, rng);
                char name[160];
                std::snprintf(
                    name, sizeof(name),
                    "ModulusOrdered %s %s q%zu batch=%d mod_count=%d",
                    fwd ? "NTT" : "INTT", inplace ? "Inplace" : "out-of-place",
                    c.q.size(), k.batch, mod_count);
                auto cfg = rns_cfg(c, fwd);
                Root64* tab = fwd ? c.ntt_table.p : c.intt_table.p;
                compare(name, in_h, k.batch, inplace,
                        [&](u64* in, u64* out)
                        {
                            if (inplace)
                                gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                                    in, tab, c.modulus.p, cfg, k.batch,
                                    mod_count, order.p);
                            else
                                gpuntt::GPU_NTT_Modulus_Ordered(
                                    in, out, tab, c.modulus.p, cfg, k.batch,
                                    mod_count, order.p);
                        });
            }
        }
    }
}

TEST(NTTFast16, PolyOrdered)
{
    cudaSetDevice(0);
    std::mt19937_64 rng(4);
    Chain named(kNamed);
    Chain big(chain28());

    struct Cfg
    {
        Chain* c;
        int polys; // limbs in the buffer
        int mod_count;
        int offset; // prime / table offset, as HEonGPU's rescale passes
        std::vector<int> order; // batch = order.size()
    };
    std::vector<int> perm24(24);
    std::iota(perm24.begin(), perm24.end(), 0);
    std::shuffle(perm24.begin(), perm24.end(), rng);
    std::vector<Cfg> cfgs = {
        {&named, 10, 4, 0, {9, 4, 8, 3, 6, 1, 2, 0}},
        {&named, 8, 2, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        {&named, 6, 1, 4, {2, 5}}, // HEonGPU: batch 2, mod_count 1
        {&named, 6, 1, 2, {1, 4}},
        {&big, 24, 12, 0, perm24},
        {&big, 30, 1, 27, {28, 29}}};
    for (const Cfg& k : cfgs)
    {
        Chain& c = *k.c;
        const int batch = (int) k.order.size();
        DevBuf<int> order(k.order);
        // limb order[z] uses prime offset + z % mod_count; other limbs any
        std::vector<int> prime_idx(k.polys, k.offset);
        for (int z = 0; z < batch; z++)
            prime_idx[k.order[z]] = k.offset + z % k.mod_count;
        auto prime_of = [&](size_t p) { return c.q[prime_idx[p]]; };
        for (int fwd = 1; fwd >= 0; fwd--)
        {
            for (int inplace = 0; inplace < 2; inplace++)
            {
                auto in_h = make_input(k.polys, prime_of, rng);
                char name[160];
                std::snprintf(
                    name, sizeof(name),
                    "PolyOrdered %s %s q%zu polys=%d batch=%d mod_count=%d "
                    "off=%d",
                    fwd ? "NTT" : "INTT", inplace ? "Inplace" : "out-of-place",
                    c.q.size(), k.polys, batch, k.mod_count, k.offset);
                auto cfg = rns_cfg(c, fwd, k.offset);
                Root64* tab = (fwd ? c.ntt_table.p : c.intt_table.p) +
                              ((size_t) k.offset << LOGN);
                Modulus64* mod = c.modulus.p + k.offset;
                compare(name, in_h, k.polys, inplace,
                        [&](u64* in, u64* out)
                        {
                            if (inplace)
                                gpuntt::GPU_NTT_Poly_Ordered_Inplace(
                                    in, tab, mod, cfg, batch, k.mod_count,
                                    order.p);
                            else
                                gpuntt::GPU_NTT_Poly_Ordered(
                                    in, out, tab, mod, cfg, batch,
                                    k.mod_count, order.p);
                        });
            }
        }
    }
}

TEST(NTTFast16, SinglePrime)
{
    cudaSetDevice(0);
    std::mt19937_64 rng(5);
    Chain named(kNamed);
    for (int i = 0; i < (int) named.q.size(); i++)
    {
        std::vector<Ninverse64> ninv = heongpu::generate_n_inverse(
            N, std::vector<Modulus64>{named.mods[i]});
        for (int batch : {1, 3})
        {
            for (int fwd = 1; fwd >= 0; fwd--)
            {
                for (int inplace = 0; inplace < 2; inplace++)
                {
                    auto in_h = make_input(
                        batch, [&](size_t) { return named.q[i]; }, rng);
                    gpuntt::ntt_configuration<u64> cfg = {
                        .n_power = LOGN,
                        .ntt_type = fwd ? gpuntt::FORWARD : gpuntt::INVERSE,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .mod_inverse = ninv[0],
                        .stream = 0};
                    Root64* tab = (fwd ? named.ntt_table.p
                                       : named.intt_table.p) +
                                  ((size_t) i << LOGN);
                    char name[160];
                    std::snprintf(name, sizeof(name),
                                  "single prime %#llx %s %s batch=%d",
                                  (unsigned long long) named.q[i],
                                  fwd ? "NTT" : "INTT",
                                  inplace ? "Inplace" : "out-of-place", batch);
                    compare(name, in_h, batch, inplace,
                            [&](u64* in, u64* out)
                            {
                                if (inplace)
                                    gpuntt::GPU_NTT_Inplace(
                                        in, tab, named.mods[i], cfg, batch);
                                else
                                    gpuntt::GPU_NTT(in, out, tab,
                                                    named.mods[i], cfg, batch);
                            });
                }
            }
        }
    }
}

// fast INTT(fast NTT(x)) == x, and generic INTT(fast NTT(x)) == x
TEST(NTTFast16, RoundTrip)
{
    cudaSetDevice(0);
    std::mt19937_64 rng(6);
    Chain big(chain28());
    const int batch = 56, mod_count = 28;
    auto in_h = make_input(
        batch, [&](size_t z) { return big.q[z % mod_count]; }, rng);
    for (int inv_fast = 0; inv_fast < 2; inv_fast++)
    {
        DevBuf<u64> d(in_h);
        gpuntt::fast16::set_enabled(1);
        gpuntt::GPU_NTT_Inplace(d.p, big.ntt_table.p, big.modulus.p,
                                rns_cfg(big, true), batch, mod_count);
        gpuntt::fast16::set_enabled(inv_fast);
        gpuntt::GPU_NTT_Inplace(d.p, big.intt_table.p, big.modulus.p,
                                rns_cfg(big, false), batch, mod_count);
        gpuntt::fast16::set_enabled(-1);
        CK(cudaDeviceSynchronize());
        EXPECT_TRUE(d.download() == in_h)
            << "round trip, inverse " << (inv_fast ? "fast" : "generic");
    }
}

// A table freed and replaced at the same address by another chain's table
// must not reuse the stale Shoup cache.
TEST(NTTFast16, StaleCacheIsDetected)
{
    cudaSetDevice(0);
    std::mt19937_64 rng(7);
    Chain a(kNamed);
    const std::vector<u64> all = chain28();
    Chain b(std::vector<u64>(all.begin() + 5, all.begin() + 10)); // other primes

    DevBuf<Modulus64> mod(a.mods);
    DevBuf<Root64> tab(a.build(true));
    DevBuf<Root64> itab(a.build(false));
    DevBuf<Ninverse64> ninv(heongpu::generate_n_inverse(N, a.mods));

    // the first transform of a case runs on a stale entry (its kernels fall
    // back to on-the-fly Shoup values and flag the host); the second one
    // runs after the host has rebuilt the cache
    auto run_case = [&](Chain& c, const char* tag, bool fwd_first)
    {
        auto prime_of = [&](size_t z) { return c.q[z % 5]; };
        for (int step = 0; step < 2; step++)
        {
            const int fwd = (step == 0) == fwd_first;
            auto in_h = make_input(10, prime_of, rng);
            gpuntt::ntt_rns_configuration<u64> cfg = {
                .n_power = LOGN,
                .ntt_type = fwd ? gpuntt::FORWARD : gpuntt::INVERSE,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .mod_inverse = ninv.p,
                .stream = 0};
            compare(std::string("stale cache ") + tag + (fwd ? " NTT" : " INTT"),
                    in_h, 10, false,
                    [&](u64* in, u64* out)
                    {
                        gpuntt::GPU_NTT(in, out, fwd ? tab.p : itab.p, mod.p,
                                        cfg, 10, 5);
                    });
        }
    };

    run_case(a, "chain A", true);
    // overwrite tables, primes and n^-1 in place with chain B
    mod.upload(b.mods);
    tab.upload(b.build(true));
    itab.upload(b.build(false));
    ninv.upload(heongpu::generate_n_inverse(N, b.mods));
    run_case(b, "chain B, stale INTT first", false);
    run_case(b, "chain B, rebuilt", true);
    // forward / inverse tables swapped at the same addresses (same primes)
    tab.upload(b.build(false));
    itab.upload(b.build(true));
    run_case(b, "chain B swapped, stale NTT first", true);
    // back to chain A
    mod.upload(a.mods);
    tab.upload(a.build(true));
    itab.upload(a.build(false));
    ninv.upload(heongpu::generate_n_inverse(N, a.mods));
    run_case(a, "chain A again, stale INTT", true);
}

// Throughput of batched RNS NTT / INTT (in place), generic vs fast. Disabled
// by default; run with
//   ntt_fast16_testcases --gtest_also_run_disabled_tests
//   --gtest_filter='*Throughput*'
TEST(NTTFast16, DISABLED_Throughput)
{
    cudaSetDevice(0);
    std::mt19937_64 rng(8);
    const int iters = 200;
    std::vector<u64> primes = chain28();
    std::printf("%-9s %-6s %-8s %12s %12s %12s %12s\n", "mod_count", "batch",
                "path", "NTT us", "INTT us", "NTT+INTT us", "us/limb(F+I)");
    for (int mod_count : {24, 28})
    {
        Chain c(std::vector<u64>(primes.begin(), primes.begin() + mod_count));
        for (int mult : {1, 2, 4, 8, 16})
        {
            const int batch = mod_count * mult;
            auto in_h = make_input(
                batch, [&](size_t z) { return c.q[z % mod_count]; }, rng);
            DevBuf<u64> d(in_h);
            double t[2][3];
            std::vector<u64> out[2];
            for (int fast = 0; fast < 2; fast++)
            {
                gpuntt::fast16::set_enabled(fast);
                auto fwd = [&]
                {
                    gpuntt::GPU_NTT_Inplace(d.p, c.ntt_table.p, c.modulus.p,
                                            rns_cfg(c, true), batch,
                                            mod_count);
                };
                auto inv = [&]
                {
                    gpuntt::GPU_NTT_Inplace(d.p, c.intt_table.p, c.modulus.p,
                                            rns_cfg(c, false), batch,
                                            mod_count);
                };
                for (int w = 0; w < 20; w++)
                {
                    fwd();
                    inv();
                }
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                for (int what = 0; what < 3; what++)
                {
                    cudaEventRecord(e0);
                    for (int i = 0; i < iters; i++)
                    {
                        if (what != 1)
                            fwd();
                        if (what != 0)
                            inv();
                    }
                    cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    t[fast][what] = 1000.0 * ms / iters;
                    // keep the data reduced: an NTT-only loop leaves it in the
                    // NTT domain, which is still a valid input
                }
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                fwd();
                out[fast] = d.download();
                inv();
                CK(cudaDeviceSynchronize());
                std::printf("%-9d %-6d %-8s %12.2f %12.2f %12.2f %12.3f\n",
                            mod_count, batch, fast ? "fast16" : "generic",
                            t[fast][0], t[fast][1], t[fast][2],
                            t[fast][2] / batch);
            }
            gpuntt::fast16::set_enabled(-1);
            std::printf("%-9d %-6d %-8s %11.2fx %11.2fx %11.2fx\n", mod_count,
                        batch, "speedup", t[0][0] / t[1][0], t[0][1] / t[1][1],
                        t[0][2] / t[1][2]);
            // both paths started from the same state and applied the same
            // sequence of transforms
            EXPECT_TRUE(out[0] == out[1]);
        }
    }
}
