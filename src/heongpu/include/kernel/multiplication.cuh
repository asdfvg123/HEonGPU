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

    __global__ void cipherplain_multiply_add_one_idx_kernel(
        const Data64* __restrict__ packed_baby_steps, // [g2 * (2*N*Q)]
        const Data64* __restrict__ one_diag_ntt,      // [N*Q] for this single diagonal
        int jslot,                                    // baby-step index for this diagonal
        Data64* __restrict__ out_ntt,                 // accumulator in NTT domain
        const Modulus64* __restrict__ modulus,        // [Q]
        int current_decomp_count, int n_power);

    __global__ void cipherplain_multiply_accumulate_block_kernel(
        const Data64* __restrict__ packed_baby_steps, // [g2 * (2*N*Q)], pack_baby_steps_all 결과
        const Data64* __restrict__ diags_ntt_slab,    // [M * (N*Q)] 이번 배치의 NTT된 대각선들
        const int*    __restrict__ j_of_m,            // [M] 각 대각선에 대응되는 baby-step j
        Data64*       __restrict__ out_ntt,           // 누적 대상 (NTT domain, 2*N*Q)
        const Modulus64* __restrict__ modulus,        // [Q]
        int M,                                        // 이번 배치 크기
        int current_decomp_count,                     // = Q_size_
        int n_power);
    __global__ void threshold_kernel_batched(
        const Data64* __restrict__ plains,            // [B * N]
        Data64*       __restrict__ out_rns,           // [Q * B * N] limb-major
        const Modulus64* __restrict__ modulus,        // [Q]
        const Data64* __restrict__ upper_inc,         // [Q]
        const Data64  upper_thresh,
        int n_power, int Q, int B);
    __global__ void repack_limbMajor_to_batchMajor(
        const Data64* __restrict__ in_rns,   // [Q * B * N]
        Data64*       __restrict__ out_slab, // [B * (Q*N)]
        int n_power, int Q, int B);
    __global__ void threshold_to_polymajor_kernel(
        const Data64* __restrict__ plains_BN,   // [B*N], b단위로 N씩
        Data64*       __restrict__ out_polyBQ_N,// [(B*Q)*N]
        const Modulus64* __restrict__ modulus,  // [Q]
        const Data64* __restrict__ upper_inc,   // [Q]
        Data64 upper_thresh, int n_power, int Q, int B);
    __global__ void repack_poly_to_slab_kernel(
        const Data64* __restrict__ in_polyBQ_N, // [(B*Q)*N]
        Data64*       __restrict__ out_slab_B_QN,// [B*(Q*N)]
        int n_power, int Q, int B);
    __global__ void gather_threshold_to_polymajor_kernel(
        const Data64* __restrict__ base_plain,   // [num_diags * N]
        const int*   __restrict__ k_list,        // [B]
        Data64*      __restrict__ in_polyBQ_N,   // [(B*Q) * N] (poly-major)
        const Modulus64* __restrict__ modulus,   // [Q]
        const Data64* __restrict__ upper_inc,    // [Q]
        const Data64  upper_thresh,
        int n_power, int Q, int B);
} // namespace heongpu
#endif // HEONGPU_MULTIPLICATION_H