// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_MULTIPLICATION_H
#define HEONGPU_MULTIPLICATION_H

#include "cuda_runtime.h"
#include "modular_arith.cuh"
#include "defines.h"

namespace heongpu
{
    // Homomorphic Multiplication Kernels

    __global__ void cross_multiplication(Data64* in1, Data64* in2, Data64* out,
                                         Modulus64* modulus, int n_power,
                                         int decomp_size);

    __global__ void
    fast_convertion(Data64* in1, Data64* in2, Data64* out1, Modulus64* ibase,
                    Modulus64* obase, Modulus64 m_tilde,
                    Data64 inv_prod_q_mod_m_tilde, Data64* inv_m_tilde_mod_Bsk,
                    Data64* prod_q_mod_Bsk, Data64* base_change_matrix_Bsk,
                    Data64* base_change_matrix_m_tilde,
                    Data64* inv_punctured_prod_mod_base_array, int n_power,
                    int ibase_size, int obase_size);

    __global__ void fast_floor(
        Data64* in_baseq_Bsk, Data64* out1, Modulus64* ibase, Modulus64* obase,
        Modulus64 plain_modulus, Data64* inv_punctured_prod_mod_base_array,
        Data64* base_change_matrix_Bsk, Data64* inv_prod_q_mod_Bsk,
        Data64* inv_punctured_prod_mod_B_array, Data64* base_change_matrix_q,
        Data64* base_change_matrix_msk, Data64 inv_prod_B_mod_m_sk,
        Data64* prod_B_mod_q, int n_power, int ibase_size, int obase_size);

    __global__ void threshold_kernel(Data64* plain_in, Data64* output,
                                     Modulus64* modulus,
                                     Data64* plain_upper_half_increment,
                                     Data64 plain_upper_half_threshold,
                                     int n_power, int decomp_size);

    __global__ void cipherplain_kernel(Data64* cipher, Data64* plain_in,
                                       Data64* output, Modulus64* modulus,
                                       int n_power, int decomp_size);

    __global__ void cipherplain_multiplication_kernel(Data64* in1, Data64* in2,
                                                      Data64* out,
                                                      Modulus64* modulus,
                                                      int n_power);

    __global__ void cipherplain_multiply_accumulate_kernel(
        Data64* in1, Data64* in2, Data64* out, Modulus64* modulus,
        int iteration_count, int current_decomp_count, int first_decomp_count,
        int n_power);

    __global__ void cipherplain_multiply_accumulate_ctpacked_ptindexed_kernel(
    const Data64* __restrict__ in1,             // packed baby-steps (per-iter)
    const Data64* __restrict__ base_diagonals,  // ALL diagonals (blob)
    const int*    __restrict__ k_of_iter,       // [iteration_count]
    Data64*             out,                    // NTT domain ciphertext for this bucket
    const Modulus64*    modulus,
    int iteration_count,
    int current_decomp_count,   // = Q_size_
    int first_decomp_count,     // = Q_size_ (diag stride)
    int n_power);

    __global__ void cipherplain_multiply_accumulate_idx_kernel(
        const Data64* __restrict__ packed_baby_steps,  // [g2 * (2*N*Q)], 그룹에서 1회만 pack
        const Data64* __restrict__ base_diagonals,     // gen_FV_S2C_Matrix()의 blob
        const int*    __restrict__ j_of_iter,          // [iteration_count], 각 it의 baby-step j
        const int*    __restrict__ k_of_iter,          // [iteration_count], 각 it의 diag k
        Data64*             out,                       // NTT 결과
        const Modulus64*    modulus,
        int iteration_count, int current_decomp_count, int first_decomp_count, int n_power);

} // namespace heongpu
#endif // HEONGPU_MULTIPLICATION_H