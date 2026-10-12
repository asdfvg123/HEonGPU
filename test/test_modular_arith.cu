// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Regression test for the device Barrett arithmetic of GPU-NTT
// (thirdparty/GPU-NTT/src/include/common/modular_arith.cuh), checked against
// exact host 128-bit arithmetic.
//
// uint128_t::operator>> / operator<< used to shift a 64-bit word by 64 (UB)
// when the shift amount was 0 or 64. Barrett multiplication shifts by
// modulus.bit + 3, which is exactly 64 for a 61-bit modulus such as the BFV
// decryption modulus gamma; CUDA 13 miscompiled that shift and every BFV
// decryption came out wrong. See patches/gpu-ntt.patch.

#include "heongpu.cuh"
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    using u128 = unsigned __int128;

    // Minimal owning device buffer (plain cudaMalloc: the HEonGPU memory pool
    // is not needed here).
    template <typename T> class DeviceBuffer
    {
      public:
        explicit DeviceBuffer(const std::vector<T>& host) : size_(host.size())
        {
            EXPECT_EQ(cudaMalloc(&ptr_, size_ * sizeof(T)), cudaSuccess);
            EXPECT_EQ(cudaMemcpy(ptr_, host.data(), size_ * sizeof(T),
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
        }
        explicit DeviceBuffer(size_t size) : size_(size)
        {
            EXPECT_EQ(cudaMalloc(&ptr_, size_ * sizeof(T)), cudaSuccess);
        }
        ~DeviceBuffer() { cudaFree(ptr_); }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        T* data() { return ptr_; }
        std::vector<T> to_host() const
        {
            std::vector<T> host(size_);
            EXPECT_EQ(cudaMemcpy(host.data(), ptr_, size_ * sizeof(T),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            return host;
        }

      private:
        T* ptr_ = nullptr;
        size_t size_;
    };

    // out[2i] = v >> s, out[2i+1] = v << s, each as (lo, hi). The shift is
    // read from memory so the compiler cannot fold it.
    __global__ void uint128_shift_kernel(const Data64* lo, const Data64* hi,
                                         const unsigned* shift, Data64* out,
                                         int count)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count)
            return;
        // OPERATOR_GPU_64's uint128_t is a private nested type; name it
        // through the public mult128 instead.
        using U128 = decltype(OPERATOR_GPU_64::mult128(Data64(0), Data64(0)));
        U128 v;
        v.value.x = lo[i];
        v.value.y = hi[i];
        U128 r = v >> shift[i];
        U128 l = v << shift[i];
        out[4 * i + 0] = r.value.x;
        out[4 * i + 1] = r.value.y;
        out[4 * i + 2] = l.value.x;
        out[4 * i + 3] = l.value.y;
    }

    // out[4i + 0] = mult(a, b)            (a, b < q)
    // out[4i + 1] = reduce(x)             (x < min(2^64, q^2))
    // out[4i + 2] = reduce_forced(y)      (any 64-bit y)
    // out[4i + 3] = reduce({z_lo, z_hi})  (128-bit z < q^2)
    __global__ void barrett_kernel(const Data64* a, const Data64* b,
                                   const Data64* x, const Data64* y,
                                   const Data64* z_lo, const Data64* z_hi,
                                   Modulus64 modulus, Data64* out, int count)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count)
            return;
        out[4 * i + 0] = OPERATOR_GPU_64::mult(a[i], b[i], modulus);
        out[4 * i + 1] = OPERATOR_GPU_64::reduce(x[i], modulus);
        out[4 * i + 2] = OPERATOR_GPU_64::reduce_forced(y[i], modulus);
        Data64 wide[2] = {z_lo[i], z_hi[i]};
        out[4 * i + 3] = OPERATOR_GPU_64::reduce(wide, modulus);
    }

    constexpr int kThreads = 256;

    int blocks_for(int count) { return (count + kThreads - 1) / kThreads; }

    struct BarrettResult
    {
        size_t bad_mult = 0, bad_reduce = 0, bad_forced = 0, bad_wide = 0;
        std::string first_failure;
    };

    BarrettResult check_barrett(const Modulus64& m, size_t random_count,
                                uint64_t seed)
    {
        const uint64_t q = m.value;
        // Largest x for which a single Barrett reduce is specified: q^2,
        // capped at 2^64.
        const u128 q_sq = static_cast<u128>(q) * q;
        const uint64_t x_max =
            (q_sq > static_cast<u128>(UINT64_MAX)) ? UINT64_MAX
                                                   : static_cast<uint64_t>(q_sq - 1);

        std::vector<Data64> a, b, x, y;
        // Edge cases first.
        const std::vector<Data64> edge_ab = {0, 1, 2, q / 2, q / 2 + 1,
                                               q - 2, q - 1};
        for (uint64_t ea : edge_ab)
            for (uint64_t eb : edge_ab)
            {
                a.push_back(ea);
                b.push_back(eb);
            }
        const std::vector<Data64> edge_x = {0,         1,         q - 1,
                                              q,         q + 1,     2 * q - 1,
                                              2 * q,     x_max - q, x_max - 1,
                                              x_max};
        const std::vector<Data64> edge_y = {0,
                                              1,
                                              q - 1,
                                              q,
                                              2 * q,
                                              uint64_t(1) << 63,
                                              UINT64_MAX - q,
                                              UINT64_MAX - 1,
                                              UINT64_MAX};
        std::mt19937_64 rng(seed);
        const size_t count = a.size() + random_count;
        for (size_t i = a.size(); i < count; ++i)
        {
            a.push_back(rng() % q);
            b.push_back(rng() % q);
        }
        for (size_t i = 0; i < count; ++i)
        {
            x.push_back(i < edge_x.size()
                            ? edge_x[i]
                            : (x_max == UINT64_MAX ? rng() : rng() % (x_max + 1)));
            y.push_back(i < edge_y.size() ? edge_y[i] : rng());
        }
        std::vector<Data64> z_lo(count), z_hi(count);
        for (size_t i = 0; i < count; ++i)
        {
            const u128 z = static_cast<u128>(a[i]) * b[i];
            z_lo[i] = static_cast<uint64_t>(z);
            z_hi[i] = static_cast<uint64_t>(z >> 64);
        }

        DeviceBuffer<Data64> d_a(a), d_b(b), d_x(x), d_y(y), d_zlo(z_lo),
            d_zhi(z_hi), d_out(4 * count);
        barrett_kernel<<<blocks_for(count), kThreads>>>(
            d_a.data(), d_b.data(), d_x.data(), d_y.data(), d_zlo.data(),
            d_zhi.data(), m, d_out.data(), static_cast<int>(count));
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<Data64> out = d_out.to_host();

        BarrettResult res;
        auto note = [&](const char* op, size_t i, uint64_t got, uint64_t exp)
        {
            if (!res.first_failure.empty())
                return;
            std::ostringstream os;
            os << op << " i=" << i << " a=" << a[i] << " b=" << b[i]
               << " x=" << x[i] << " y=" << y[i] << ": got " << got
               << ", expected " << exp;
            res.first_failure = os.str();
        };
        for (size_t i = 0; i < count; ++i)
        {
            const uint64_t e_mult =
                static_cast<uint64_t>(static_cast<u128>(a[i]) * b[i] % q);
            const uint64_t e_x = x[i] % q;
            const uint64_t e_y = y[i] % q;
            if (out[4 * i + 0] != e_mult)
            {
                ++res.bad_mult;
                note("mult", i, out[4 * i + 0], e_mult);
            }
            if (out[4 * i + 1] != e_x)
            {
                ++res.bad_reduce;
                note("reduce", i, out[4 * i + 1], e_x);
            }
            if (out[4 * i + 2] != e_y)
            {
                ++res.bad_forced;
                note("reduce_forced", i, out[4 * i + 2], e_y);
            }
            if (out[4 * i + 3] != e_mult)
            {
                ++res.bad_wide;
                note("reduce(128-bit)", i, out[4 * i + 3], e_mult);
            }
        }
        return res;
    }

    // NTT-friendly primes (q = 1 mod 2N) of the given bit size, as the
    // context would draw them.
    std::vector<Modulus64> primes_of_bits(int bits, size_t count)
    {
        const Data64 factor = 2 * 4096;
        std::vector<Modulus64> out;
        for (Data64 p : heongpu::generate_proper_primes(factor, bits, count))
            out.emplace_back(p);
        return out;
    }

} // namespace

TEST(HEonGPU, Uint128_Shifts_Match_Host)
{
    cudaSetDevice(0);

    std::vector<Data64> lo, hi;
    std::vector<unsigned> shift;
    std::mt19937_64 rng(0x5eed);
    const std::vector<std::pair<uint64_t, uint64_t>> edge = {
        {0, 0},
        {1, 0},
        {0, 1},
        {UINT64_MAX, UINT64_MAX},
        {uint64_t(1) << 63, uint64_t(1) << 63},
        {0x0123456789abcdefULL, 0xfedcba9876543210ULL}};
    // Every shift in [0, 128) for every value; 0 and 64 are the cases that
    // were undefined.
    const int random_values = 64;
    for (int v = 0; v < (int) edge.size() + random_values; ++v)
    {
        const uint64_t l = v < (int) edge.size() ? edge[v].first : rng();
        const uint64_t h = v < (int) edge.size() ? edge[v].second : rng();
        for (unsigned s = 0; s < 128; ++s)
        {
            lo.push_back(l);
            hi.push_back(h);
            shift.push_back(s);
        }
    }
    const int count = static_cast<int>(lo.size());

    DeviceBuffer<Data64> d_lo(lo), d_hi(hi), d_out(4 * count);
    DeviceBuffer<unsigned> d_shift(shift);
    uint128_shift_kernel<<<blocks_for(count), kThreads>>>(
        d_lo.data(), d_hi.data(), d_shift.data(), d_out.data(), count);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<Data64> out = d_out.to_host();

    size_t bad_right = 0, bad_left = 0;
    std::vector<unsigned> bad_shifts;
    for (int i = 0; i < count; ++i)
    {
        const u128 v = (static_cast<u128>(hi[i]) << 64) | lo[i];
        const u128 r = v >> shift[i];
        const u128 l = v << shift[i];
        const bool ok_r = out[4 * i + 0] == static_cast<uint64_t>(r) &&
                          out[4 * i + 1] == static_cast<uint64_t>(r >> 64);
        const bool ok_l = out[4 * i + 2] == static_cast<uint64_t>(l) &&
                          out[4 * i + 3] == static_cast<uint64_t>(l >> 64);
        bad_right += !ok_r;
        bad_left += !ok_l;
        if ((!ok_r || !ok_l) && bad_shifts.size() < 8)
            bad_shifts.push_back(shift[i]);
    }
    std::ostringstream os;
    for (unsigned s : bad_shifts)
        os << s << " ";
    EXPECT_EQ(bad_right, 0u) << "operator>> mismatches, e.g. shifts " << os.str();
    EXPECT_EQ(bad_left, 0u) << "operator<< mismatches, e.g. shifts " << os.str();
}

TEST(HEonGPU, Barrett_Mult_Reduce_Match_Host)
{
    cudaSetDevice(0);

    struct Case
    {
        int bits;
        Modulus64 modulus;
    };
    std::vector<Case> cases;
    for (int bits : {20, 36, 50, 59, 60, 61})
        for (const Modulus64& m : primes_of_bits(bits, 2))
            cases.push_back({bits, m});
    // The decryption modulus gamma of a BFV / RTF context is the last of the
    // internal primes, which are MAX_MOD_BIT_COUNT = 61 bits wide.
    for (size_t n : {size_t(4096), size_t(65536)})
        for (const Modulus64& m : heongpu::generate_internal_primes(n, 3))
            cases.push_back({MAX_MOD_BIT_COUNT, m});

    uint64_t seed = 1;
    for (const Case& c : cases)
    {
        SCOPED_TRACE("modulus " + std::to_string(c.modulus.value) + " (" +
                     std::to_string(c.bits) + " bits)");
        ASSERT_EQ(static_cast<int>(c.modulus.bit), c.bits);
        if (c.bits == 61)
        {
            // The case that broke: Barrett shifts by bit + 3 == 64.
            ASSERT_EQ(c.modulus.bit + 3, 64ULL);
        }
        const BarrettResult r = check_barrett(c.modulus, 1 << 16, seed++);
        EXPECT_EQ(r.bad_mult, 0u) << r.first_failure;
        EXPECT_EQ(r.bad_reduce, 0u) << r.first_failure;
        EXPECT_EQ(r.bad_forced, 0u) << r.first_failure;
        EXPECT_EQ(r.bad_wide, 0u) << r.first_failure;
    }
}
