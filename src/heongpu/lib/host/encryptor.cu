﻿// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "encryptor.cuh"

namespace heongpu
{

    __host__ HEEncryptor::HEEncryptor(Parameters& context,
                                      Publickey& public_key)
    {
        scheme_ = context.scheme_;

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();
        offset_ = gen();

        public_key_ = public_key.data();

        n = context.n;
        n_power = context.n_power;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;
        P_size_ = context.P_size;

        modulus_ = context.modulus_;

        last_q_modinv_ = context.last_q_modinv_;

        ntt_table_ = context.ntt_table_;
        intt_table_ = context.intt_table_;

        n_inverse_ = context.n_inverse_;

        half_ = context.half_p_;

        half_mod_ = context.half_mod_;

        n = context.n;
        n_power = context.n_power;

        if (scheme_ == scheme_type::bfv)
        {
            plain_modulus_ = context.plain_modulus_;

            Q_mod_t_ = context.Q_mod_t_;

            upper_threshold_ = context.upper_threshold_;

            coeeff_div_plainmod_ = context.coeeff_div_plainmod_;
        }
        else
        {
        }
    }

    __host__ void HEEncryptor::encrypt_bfv(Ciphertext& ciphertext,
                                           Plaintext& plaintext)
    {
        DeviceVector<Data> gpu_space(5 * Q_prime_size_ * n);
        Data* u_poly = gpu_space.data();
        Data* error_poly = u_poly + (Q_prime_size_ * n);
        Data* pk_u_poly = error_poly + (2 * Q_prime_size_ * n);

        modular_ternary_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256>>>(
            u_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<dim3((n >> 8), 2, 1),
                                                           256>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(u_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                        Q_prime_size_, Q_prime_size_);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256>>>(
            public_key_, u_poly, pk_u_poly, modulus_->data(), n_power,
            Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(pk_u_poly, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        enc_div_lastq_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            pk_u_poly, error_poly, plaintext.data(), ciphertext.data(),
            modulus_->data(), half_->data(), half_mod_->data(),
            last_q_modinv_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncryptor::encrypt_bfv(Ciphertext& ciphertext,
                                           Plaintext& plaintext,
                                           HEStream& stream)
    {
        DeviceVector<Data> gpu_space(5 * Q_prime_size_ * n, stream.stream);
        Data* u_poly = gpu_space.data();
        Data* error_poly = u_poly + (Q_prime_size_ * n);
        Data* pk_u_poly = error_poly + (2 * Q_prime_size_ * n);

        modular_ternary_random_number_generation_kernel<<<
            dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            u_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), 2, 1), 256, 0, stream.stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(u_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                        Q_prime_size_, Q_prime_size_);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256, 0,
                      stream.stream>>>(public_key_, u_poly, pk_u_poly,
                                       modulus_->data(), n_power,
                                       Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(pk_u_poly, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        enc_div_lastq_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                   stream.stream>>>(
            pk_u_poly, error_poly, plaintext.data(), ciphertext.data(),
            modulus_->data(), half_->data(), half_mod_->data(),
            last_q_modinv_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncryptor::encrypt_ckks(Ciphertext& ciphertext,
                                            Plaintext& plaintext)
    {
        DeviceVector<Data> gpu_space(5 * Q_prime_size_ * n);
        Data* u_poly = gpu_space.data();
        Data* error_poly = u_poly + (Q_prime_size_ * n);
        Data* pk_u_poly = error_poly + (2 * Q_prime_size_ * n);

        modular_ternary_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256>>>(
            u_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<dim3((n >> 8), 2, 1),
                                                           256>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(u_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                        Q_prime_size_, Q_prime_size_);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256>>>(
            public_key_, u_poly, pk_u_poly, modulus_->data(), n_power,
            Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(pk_u_poly, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        enc_div_lastq_ckks_kernel<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            pk_u_poly, error_poly, ciphertext.data(), modulus_->data(),
            half_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(ciphertext.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_size_, Q_size_);

        cipher_message_add_kernel<<<dim3((n >> 8), Q_size_, 1), 256>>>(
            ciphertext.data(), plaintext.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncryptor::encrypt_ckks(Ciphertext& ciphertext,
                                            Plaintext& plaintext,
                                            HEStream& stream)
    {
        DeviceVector<Data> gpu_space(5 * Q_prime_size_ * n, stream.stream);
        Data* u_poly = gpu_space.data();
        Data* error_poly = u_poly + (Q_prime_size_ * n);
        Data* pk_u_poly = error_poly + (2 * Q_prime_size_ * n);

        modular_ternary_random_number_generation_kernel<<<
            dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            u_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), 2, 1), 256, 0, stream.stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(u_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                        Q_prime_size_, Q_prime_size_);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256, 0,
                      stream.stream>>>(public_key_, u_poly, pk_u_poly,
                                       modulus_->data(), n_power,
                                       Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(pk_u_poly, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        enc_div_lastq_ckks_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                    stream.stream>>>(
            pk_u_poly, error_poly, ciphertext.data(), modulus_->data(),
            half_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(ciphertext.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_size_, Q_size_);

        cipher_message_add_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                    stream.stream>>>(
            ciphertext.data(), plaintext.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu