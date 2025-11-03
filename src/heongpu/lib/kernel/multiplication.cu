// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "multiplication.cuh"

namespace heongpu
{
    __global__ void
    fast_convertion(Data64* in1, Data64* in2, Data64* out1, Modulus64* ibase,
                    Modulus64* obase, Modulus64 m_tilde,
                    Data64 inv_prod_q_mod_m_tilde, Data64* inv_m_tilde_mod_Bsk,
                    Data64* prod_q_mod_Bsk, Data64* base_change_matrix_Bsk,
                    Data64* base_change_matrix_m_tilde,
                    Data64* inv_punctured_prod_mod_base_array, int n_power,
                    int ibase_size, int obase_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // cipher count * 2 // for input

        int location = idx + (((idy % 2) * ibase_size)
                              << n_power); // ibase_size = decomp_modulus_count
        Data64* input = ((idy >> 1) == 0) ? in1 : in2;

        Data64 temp[MAX_BSK_SIZE];
        Data64 temp_[MAX_BSK_SIZE];

        // taking input from global and mult with m_tilde
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            temp_[i] = input[location + (i << n_power)];
            temp[i] = OPERATOR_GPU_64::mult(temp_[i], m_tilde.value, ibase[i]);
            temp[i] = OPERATOR_GPU_64::mult(
                temp[i], inv_punctured_prod_mod_base_array[i], ibase[i]);
        }

        // for Bsk
        Data64 temp2[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            temp2[i] = 0;
#pragma unroll
            for (int j = 0; j < ibase_size; j++)
            {
                Data64 mult = OPERATOR_GPU_64::mult(
                    temp[j], base_change_matrix_Bsk[j + (i * ibase_size)],
                    obase[i]);
                temp2[i] = OPERATOR_GPU_64::add(temp2[i], mult, obase[i]);
            }
        }

        // for m_tilde
        temp2[obase_size] = 0;
#pragma unroll
        for (int j = 0; j < ibase_size; j++)
        {
            Data64 temp_in = OPERATOR_GPU_64::reduce_forced(temp[j], m_tilde);
            Data64 mult = OPERATOR_GPU_64::mult(
                temp_in, base_change_matrix_m_tilde[j], m_tilde);
            temp2[obase_size] =
                OPERATOR_GPU_64::add(temp2[obase_size], mult, m_tilde);
        }

        // sm_mrq
        Data64 m_tilde_div_2 = m_tilde.value >> 1;
        Data64 r_m_tilde = OPERATOR_GPU_64::mult(
            temp2[obase_size], inv_prod_q_mod_m_tilde, m_tilde);
        r_m_tilde = m_tilde.value - r_m_tilde;

#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            Data64 temp3 = r_m_tilde;
            if (temp3 >= m_tilde_div_2)
            {
                temp3 = obase[i].value - m_tilde.value;
                temp3 = OPERATOR_GPU_64::add(temp3, r_m_tilde, obase[i]);
            }

            temp3 = OPERATOR_GPU_64::mult(temp3, prod_q_mod_Bsk[i], obase[i]);
            temp3 = OPERATOR_GPU_64::add(temp2[i], temp3, obase[i]);
            temp2[i] =
                OPERATOR_GPU_64::mult(temp3, inv_m_tilde_mod_Bsk[i], obase[i]);
        }

        int location2 = idx + ((idy * (obase_size + ibase_size)) << n_power);
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            out1[location2 + (i << n_power)] = temp_[i];
        }
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            out1[location2 + ((i + ibase_size) << n_power)] = temp2[i];
        }
    }

    __global__ void cross_multiplication(Data64* in1, Data64* in2, Data64* out,
                                         Modulus64* modulus, int n_power,
                                         int decomp_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // decomp size + bsk size

        int location = idx + (idy << n_power);

        Data64 ct0_0 = in1[location];
        Data64 ct0_1 = in1[location + (decomp_size << n_power)];

        Data64 ct1_0 = in2[location];
        Data64 ct1_1 = in2[location + (decomp_size << n_power)];

        Data64 out_0 = OPERATOR_GPU_64::mult(ct0_0, ct1_0, modulus[idy]);
        Data64 out_1_0 = OPERATOR_GPU_64::mult(ct0_0, ct1_1, modulus[idy]);
        Data64 out_1_1 = OPERATOR_GPU_64::mult(ct0_1, ct1_0, modulus[idy]);
        Data64 out_2 = OPERATOR_GPU_64::mult(ct0_1, ct1_1, modulus[idy]);
        Data64 out_1 = OPERATOR_GPU_64::add(out_1_0, out_1_1, modulus[idy]);

        out[location] = out_0;
        out[location + (decomp_size << n_power)] = out_1;
        out[location + (decomp_size << (n_power + 1))] = out_2;
    }

    __global__ void fast_floor(
        Data64* in_baseq_Bsk, Data64* out1, Modulus64* ibase, Modulus64* obase,
        Modulus64 plain_modulus, Data64* inv_punctured_prod_mod_base_array,
        Data64* base_change_matrix_Bsk, Data64* inv_prod_q_mod_Bsk,
        Data64* inv_punctured_prod_mod_B_array, Data64* base_change_matrix_q,
        Data64* base_change_matrix_msk, Data64 inv_prod_B_mod_m_sk,
        Data64* prod_B_mod_q, int n_power, int ibase_size, int obase_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // 3

        int location_q =
            idx + ((idy * (ibase_size + obase_size))
                   << n_power); // ibase_size = decomp_modulus_count
        int location_Bsk =
            idx + ((idy * (ibase_size + obase_size)) << n_power) +
            (ibase_size << n_power); // ibase_size = decomp_modulus_count

        Data64 reg_q[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            reg_q[i] =
                OPERATOR_GPU_64::mult(in_baseq_Bsk[location_q + (i << n_power)],
                                      plain_modulus.value, ibase[i]);
            reg_q[i] = OPERATOR_GPU_64::mult(
                reg_q[i], inv_punctured_prod_mod_base_array[i], ibase[i]);
        }

        Data64 reg_Bsk[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            reg_Bsk[i] = OPERATOR_GPU_64::mult(
                in_baseq_Bsk[location_Bsk + (i << n_power)],
                plain_modulus.value, obase[i]);
        }

        // for Bsk
        Data64 temp[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            temp[i] = 0;
            for (int j = 0; j < ibase_size; j++)
            {
                Data64 mult = OPERATOR_GPU_64::mult(
                    reg_q[j], base_change_matrix_Bsk[j + (i * ibase_size)],
                    obase[i]);
                temp[i] = OPERATOR_GPU_64::add(temp[i], mult, obase[i]);
            }
        }

#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            Data64 temp2 =
                OPERATOR_GPU_64::sub(obase[i].value, temp[i], obase[i]);
            temp2 = OPERATOR_GPU_64::add(temp2, reg_Bsk[i], obase[i]);
            reg_Bsk[i] =
                OPERATOR_GPU_64::mult(temp2, inv_prod_q_mod_Bsk[i], obase[i]);
        }

        Data64 temp3[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size - 1; i++)
        { // only B bases
            temp3[i] = OPERATOR_GPU_64::mult(
                reg_Bsk[i], inv_punctured_prod_mod_B_array[i], obase[i]);
        }

        Data64 temp4[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            temp4[i] = 0;
#pragma unroll
            for (int j = 0; j < obase_size - 1; j++)
            {
                Data64 temp3_ =
                    OPERATOR_GPU_64::reduce_forced(temp3[j], ibase[i]); // extra

                Data64 mult = OPERATOR_GPU_64::mult(
                    temp3_, base_change_matrix_q[j + (i * (obase_size - 1))],
                    ibase[i]); // extra
                mult = OPERATOR_GPU_64::reduce_forced(mult, ibase[i]); // extra
                temp4[i] = OPERATOR_GPU_64::add(temp4[i], mult, ibase[i]);
            }
        }

        // for m_sk
        temp4[ibase_size] = 0;
#pragma unroll
        for (int j = 0; j < obase_size - 1; j++)
        {
            Data64 mult = OPERATOR_GPU_64::mult(
                temp3[j], base_change_matrix_msk[j], obase[obase_size - 1]);
            temp4[ibase_size] = OPERATOR_GPU_64::add(temp4[ibase_size], mult,
                                                     obase[obase_size - 1]);
        }

        Data64 alpha_sk = OPERATOR_GPU_64::sub(obase[obase_size - 1].value,
                                               reg_Bsk[obase_size - 1],
                                               obase[obase_size - 1]);
        alpha_sk = OPERATOR_GPU_64::add(alpha_sk, temp4[ibase_size],
                                        obase[obase_size - 1]);
        alpha_sk = OPERATOR_GPU_64::mult(alpha_sk, inv_prod_B_mod_m_sk,
                                         obase[obase_size - 1]);

        Data64 m_sk_div_2 = obase[obase_size - 1].value >> 1;

#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            Data64 obase_ = OPERATOR_GPU_64::reduce_forced(
                obase[obase_size - 1].value, ibase[i]);
            Data64 temp4_ = OPERATOR_GPU_64::reduce_forced(temp4[i], ibase[i]);
            Data64 alpha_sk_ =
                OPERATOR_GPU_64::reduce_forced(alpha_sk, ibase[i]);
            if (alpha_sk > m_sk_div_2)
            {
                Data64 inner =
                    OPERATOR_GPU_64::sub(obase_, alpha_sk_, ibase[i]); // extra
                inner = OPERATOR_GPU_64::mult(inner, prod_B_mod_q[i], ibase[i]);
                temp4[i] = OPERATOR_GPU_64::add(temp4_, inner, ibase[i]);
            }
            else
            {
                Data64 inner = OPERATOR_GPU_64::sub(ibase[i].value,
                                                    prod_B_mod_q[i], ibase[i]);
                inner =
                    OPERATOR_GPU_64::mult(inner, alpha_sk_, ibase[i]); // extra
                temp4[i] = OPERATOR_GPU_64::add(temp4_, inner, ibase[i]);
            }
        }

        int location_out =
            idx + ((idy * ibase_size)
                   << n_power); // ibase_size = decomp_modulus_count
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            out1[location_out + (i << n_power)] = temp4[i];
        }
    }

    __global__ void threshold_kernel(Data64* plain_in, Data64* output,
                                     Modulus64* modulus,
                                     Data64* plain_upper_half_increment,
                                     Data64 plain_upper_half_threshold,
                                     int n_power, int decomp_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // decomp_mod

        Data64 plain_reg = plain_in[idx];

        if (plain_reg >= plain_upper_half_threshold)
        {
            output[idx + (block_y << n_power)] = OPERATOR_GPU_64::add(
                plain_reg, plain_upper_half_increment[block_y],
                modulus[block_y]); // plain_reg +
                                   // plain_upper_half_increment[block_y];
        }
        else
        {
            output[idx + (block_y << n_power)] = plain_reg;
        }
    }

    __global__ void cipherplain_kernel(Data64* cipher, Data64* plain_in,
                                       Data64* output, Modulus64* modulus,
                                       int n_power, int decomp_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // decomp_mod
        int block_z = blockIdx.z; // cipher size

        int index1 = idx + (block_y << n_power);
        int index2 = index1 + ((decomp_size << n_power) * block_z);

        output[index2] = OPERATOR_GPU_64::mult(cipher[index2], plain_in[index1],
                                               modulus[block_y]);
    }

    __global__ void cipherplain_multiplication_kernel(Data64* in1, Data64* in2,
                                                      Data64* out,
                                                      Modulus64* modulus,
                                                      int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher count

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        int location_pt = idx + (block_y << n_power);

        Data64 ct = in1[location_ct];
        Data64 pt = in2[location_pt];

        ct = OPERATOR_GPU_64::mult(ct, pt, modulus[block_y]);
        out[location_ct] = ct;
    }

    __global__ void cipherplain_multiply_accumulate_kernel(
        Data64* in1, Data64* in2, Data64* out, Modulus64* modulus,
        int iteration_count, int current_decomp_count, int first_decomp_count,
        int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns base count
        int block_z = blockIdx.z; // cipher count

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);

        int location_pt = idx + (block_y << n_power);

        int offset_ct = (current_decomp_count << (n_power + 1));
        int offset_pt = (first_decomp_count << n_power);

        Data64 sum_ctpt = 0ULL;
        for (int i = 0; i < iteration_count; i++)
        {
            Data64 ct = in1[location_ct + (i * offset_ct)];
            Data64 pt = in2[location_pt + (i * offset_pt)];
            Data64 mul_ctpt = OPERATOR_GPU_64::mult(ct, pt, modulus[block_y]);
            sum_ctpt =
                OPERATOR_GPU_64::add(sum_ctpt, mul_ctpt, modulus[block_y]);
        }

        out[location_ct] = sum_ctpt;
    }

    __global__ void cipherplain_multiply_accumulate_ctpacked_ptindexed_kernel(
        const Data64* __restrict__ in1,             // packed baby-steps (per-iter)
        const Data64* __restrict__ base_diagonals,  // ALL diagonals (blob)
        const int*    __restrict__ k_of_iter,       // [iteration_count]
        Data64*             out,                    // NTT domain ciphertext for this bucket
        const Modulus64*    modulus,
        int iteration_count,
        int current_decomp_count,   // = Q_size_
        int first_decomp_count,     // = Q_size_ (diag stride)
        int n_power)
    {
        const int N    = 1 << n_power;
        const int idx  = blockIdx.x * blockDim.x + threadIdx.x; // coeff
        const int limb = blockIdx.y;                             // 0..Q-1
        const int comp = blockIdx.z;                             // 0 or 1
        if (idx >= N) return;

        // 동일한 메모리 레이아웃 가정([comp][limb][coeff])에 맞춘 stride
        const int comp_stride = (current_decomp_count << n_power);        // jump comp
        const int ct_stride   = (current_decomp_count << (n_power + 1));  // next iter (2*N*Q)
        const int pt_stride   = (first_decomp_count   << n_power);        // next diag (N*Q)

        const int ct_base = idx + (limb << n_power) + (comp * comp_stride);
        const int pt_base = idx + (limb << n_power);

        Data64 acc = 0ULL;
        for (int it = 0; it < iteration_count; ++it) {
            const int kdiag = k_of_iter[it];
            const Data64 ct = in1[ct_base + it * ct_stride];
            const Data64 pt = base_diagonals[pt_base + kdiag * pt_stride];
            const Data64 prod = OPERATOR_GPU_64::mult(ct, pt, modulus[limb]);
            acc = OPERATOR_GPU_64::add(acc, prod, modulus[limb]);
        }
        out[ct_base] = acc;
    }

    __global__ void cipherplain_multiply_add_one_idx_kernel(
        const Data64* __restrict__ packed_baby_steps, // [g2 * (2*N*Q)]
        const Data64* __restrict__ one_diag_ntt,      // [N*Q] (이번 대각선 NTT)
        int jslot,                                    // baby-step index
        Data64* __restrict__ out_ntt,                 // 누적 대상 (NTT domain)
        const Modulus64* __restrict__ modulus,        // [Q]
        int current_decomp_count, int n_power)
    {
        const int N    = 1 << n_power;
        const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
        const int limb = blockIdx.y;                   // 0..Q-1
        const int comp = blockIdx.z;                   // 0..1 (2 polys)
        if (idx >= N) return;

        const int comp_stride = (current_decomp_count << n_power);        // N * Q
        const int ct_stride   = (current_decomp_count << (n_power + 1));  // 2 * N * Q

        const int ct_base = idx + (limb << n_power) + (comp * comp_stride);
        const int pt_base = idx + (limb << n_power);

        const Data64 ct   = packed_baby_steps[ct_base + jslot * ct_stride];
        const Data64 pt   = one_diag_ntt[pt_base];

        const Data64 prod = OPERATOR_GPU_64::mult(ct, pt, modulus[limb]);
        const Data64 acc  = OPERATOR_GPU_64::add(out_ntt[ct_base], prod, modulus[limb]);
        out_ntt[ct_base]  = acc;
    }

    __global__ void cipherplain_multiply_accumulate_block_kernel(
        const Data64* __restrict__ packed_baby_steps, // [g2 * (2*N*Q)], pack_baby_steps_all 결과
        const Data64* __restrict__ diags_ntt_slab,    // [M * (N*Q)] 이번 배치의 NTT된 대각선들
        const int*    __restrict__ j_of_m,            // [M] 각 대각선에 대응되는 baby-step j
        Data64*       __restrict__ out_ntt,           // 누적 대상 (NTT domain, 2*N*Q)
        const Modulus64* __restrict__ modulus,        // [Q]
        int M,                                        // 이번 배치 크기
        int current_decomp_count,                     // = Q_size_
        int n_power)
    {
        const int N    = 1 << n_power;
        const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
        const int limb = blockIdx.y;   // 0..Q-1
        const int comp = blockIdx.z;   // 0..1 (2 polys)
        if (idx >= N) return;

        const int comp_stride     = (current_decomp_count << n_power);        // N * Q
        const int ct_stride       = (current_decomp_count << (n_power + 1));  // 2 * N * Q
        const int per_diag_stride = (current_decomp_count << n_power);        // N * Q  (diag NTT 슬랩 stride)

        const int ct_base = idx + (limb << n_power) + (comp * comp_stride);

        Data64 acc = out_ntt[ct_base];

        // 이번 배치의 모든 (j,m) 항을 한 번에 누적
        for (int m = 0; m < M; ++m) {
            const int jslot = j_of_m[m];
            const int pt_base = idx + (limb << n_power) + (m * per_diag_stride);

            const Data64 ct = packed_baby_steps[ct_base + jslot * ct_stride];
            const Data64 pt = diags_ntt_slab[pt_base];

            const Data64 prod = OPERATOR_GPU_64::mult(ct, pt, modulus[limb]);
            acc = OPERATOR_GPU_64::add(acc, prod, modulus[limb]);
        }

        out_ntt[ct_base] = acc;
    }

    // [B, N]  ->  [Q, B, N]  (limb-major)
    __global__ void threshold_kernel_batched(
        const Data64* __restrict__ plains,            // [B * N]
        Data64*       __restrict__ out_rns,           // [Q * B * N] limb-major
        const Modulus64* __restrict__ modulus,        // [Q]
        const Data64* __restrict__ upper_inc,         // [Q]
        const Data64  upper_thresh,
        int n_power, int Q, int B)
    {
        const int N = 1 << n_power;
        const int idx  = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1
        const int limb = blockIdx.y;                            // 0..Q-1
        const int b    = blockIdx.z;                            // 0..B-1
        if (idx >= N || limb >= Q || b >= B) return;

        const Data64 v = plains[b * N + idx];

        Data64 w = v;
        if (v >= upper_thresh)
            w = OPERATOR_GPU_64::add(v, upper_inc[limb], modulus[limb]);

        // limb-major layout: [limb][b][idx]
        out_rns[ ((size_t)limb * B + b) * N + idx ] = w;
    }
    // [Q, B, N] limb-major  ->  [B, Q*N] batch-major
    __global__ void repack_limbMajor_to_batchMajor(
        const Data64* __restrict__ in_rns,   // [Q * B * N]
        Data64*       __restrict__ out_slab, // [B * (Q*N)]
        int n_power, int Q, int B)
    {
        const int N = 1 << n_power;
        const int idx  = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1
        const int limb = blockIdx.y;                             // 0..Q-1
        const int b    = blockIdx.z;                             // 0..B-1
        if (idx >= N || limb >= Q || b >= B) return;

        const Data64 v =
            in_rns[ ((size_t)limb * B + b) * N + idx ];

        // batch-major: [b][limb*N + idx]
        out_slab[ (size_t)b * (Q * N) + (size_t)limb * N + idx ] = v;
    }
    __global__ void threshold_to_polymajor_kernel(
        const Data64* __restrict__ plains_BN,   // [B*N], b단위로 N씩
        Data64*       __restrict__ out_polyBQ_N,// [(B*Q)*N]
        const Modulus64* __restrict__ modulus,  // [Q]
        const Data64* __restrict__ upper_inc,   // [Q]
        Data64 upper_thresh, int n_power, int Q, int B)
    {
        const int N = 1 << n_power;
        int idx  = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1
        int limb = blockIdx.y; // 0..Q-1
        int b    = blockIdx.z; // 0..B-1
        if (idx >= N) return;

        Data64 v = plains_BN[(size_t)b*N + idx];
        if (v >= upper_thresh)
            v = OPERATOR_GPU_64::add(v, upper_inc[limb], modulus[limb]);

        int p = b*Q + limb;
        out_polyBQ_N[(size_t)p*N + idx] = v;
    }
    __global__ void repack_poly_to_slab_kernel(
        const Data64* __restrict__ in_polyBQ_N, // [(B*Q)*N]
        Data64*       __restrict__ out_slab_B_QN,// [B*(Q*N)]
        int n_power, int Q, int B)
    {
        const int N = 1 << n_power;
        int idx  = blockIdx.x * blockDim.x + threadIdx.x;
        int limb = blockIdx.y;
        int b    = blockIdx.z;
        if (idx >= N) return;

        int p = b*Q + limb; // poly index
        Data64 v = in_polyBQ_N[(size_t)p*N + idx];

        // slab: [b][limb*N + idx]
        out_slab_B_QN[(size_t)b*(Q*N) + (size_t)limb*N + idx] = v;
    }

    // base_plain[k_list[b]*N + idx] -> in_polyBQ_N[(b*Q + limb)*N + idx]
    __global__ void gather_threshold_to_polymajor_kernel(
        const Data64* __restrict__ base_plain,   // [num_diags * N]
        const int*   __restrict__ k_list,        // [B]
        Data64*      __restrict__ in_polyBQ_N,   // [(B*Q) * N] (poly-major)
        const Modulus64* __restrict__ modulus,   // [Q]
        const Data64* __restrict__ upper_inc,    // [Q]
        const Data64  upper_thresh,
        int n_power, int Q, int B)
    {
        const int N = 1 << n_power;
        const int idx  = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1
        const int limb = blockIdx.y;                             // 0..Q-1
        const int b    = blockIdx.z;                             // 0..B-1
        if (idx >= N || limb >= Q || b >= B) return;

        const int k = k_list[b];
        Data64 v = base_plain[(size_t)k * N + idx];

        if (v >= upper_thresh)
            v = OPERATOR_GPU_64::add(v, upper_inc[limb], modulus[limb]);

        const int p = b * Q + limb; // poly index
        in_polyBQ_N[(size_t)p * N + idx] = v;
    }
} // namespace heongpu