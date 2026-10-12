// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef HEONGPU_KERNEL_REDUCTION_H
#define HEONGPU_KERNEL_REDUCTION_H

#include "modular_arith.cuh"

namespace heongpu
{
    // Exact x mod q for a two-word x = x[0] + 2^64 * x[1].
    //
    // OPERATOR_GPU_64::reduce(Data64*, ...) is a single Barrett pass, exact
    // only for x < 2^(2 * bit(q)) (roughly x < q^2). That holds for the usual
    // 40-60-bit primes, but not for small primes such as the 22-bit
    // DiffScaleModulus of the RtF parameter sets, where a CKKS coefficient of
    // ~2^60 gets an unreduced/incorrect residue. The fast path (and hence the
    // result) is unchanged whenever the single pass is exact.
    __device__ __forceinline__ Data64 reduce_exact_two_word(Data64* x,
                                                           const Modulus64& m)
    {
        const int x_bits = (x[1] != 0) ? (128 - __clzll(x[1]))
                                       : (64 - __clzll(x[0]));
        if (x_bits < 2 * static_cast<int>(m.bit) - 1)
        {
            return OPERATOR_GPU_64::reduce(x, m);
        }

        Data64 lo = OPERATOR_GPU_64::reduce_forced(x[0], m);
        if (x[1] == 0)
        {
            return lo;
        }
        Data64 hi = OPERATOR_GPU_64::reduce_forced(x[1], m);
        // 2^64 mod q
        Data64 two_pow_64 = OPERATOR_GPU_64::reduce_forced(~0ULL, m) + 1;
        if (two_pow_64 >= m.value)
        {
            two_pow_64 -= m.value;
        }
        return OPERATOR_GPU_64::add(OPERATOR_GPU_64::mult(hi, two_pow_64, m),
                                    lo, m);
    }

} // namespace heongpu

#endif // HEONGPU_KERNEL_REDUCTION_H
