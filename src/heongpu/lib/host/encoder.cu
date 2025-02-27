﻿// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "encoder.cuh"

namespace heongpu
{

    __host__ HEEncoder::HEEncoder(Parameters& context)
    {
        scheme_ = context.scheme_;

        n = context.n;
        n_power = context.n_power;

        if (scheme_ == scheme_type::bfv)
        {
            slot_count_ = n;

            plain_modulus_ = context.plain_modulus2_;

            plain_ntt_tables_ = context.plain_ntt_tables_;
            plain_intt_tables_ = context.plain_intt_tables_;

            n_plain_inverse_ = context.n_plain_inverse_;

            // Encode - Decode Index
            std::vector<Data> encode_index;

            int m = n << 1;
            int gen = 3;
            int pos = 1;
            int index = 0;
            int location = 0;
            for (int i = 0; i < int(n / 2); i++)
            {
                index = (pos - 1) >> 1;
                location = bitreverse(index, n_power);
                encode_index.push_back(location);
                pos *= gen;
                pos &= (m - 1);
            }
            for (int i = int(n / 2); i < n; i++)
            {
                index = (m - pos - 1) >> 1;
                location = bitreverse(index, n_power);
                encode_index.push_back(location);
                pos *= gen;
                pos &= (m - 1);
            }

            encoding_location_ =
                std::make_shared<DeviceVector<Data>>(encode_index);
        }
        else
        { // for CKKS

            slot_count_ = n >> 1;
            log_slot_count_ = int(log2(slot_count_));
            fft_length = n * 2;

            two_pow_64 = std::pow(2.0, 64);

            Q_size_ = context.Q_size;

            total_coeff_bit_count_ = context.total_coeff_bit_count;

            modulus_ = context.modulus_;

            ntt_table_ = context.ntt_table_;
            intt_table_ = context.intt_table_;

            n_inverse_ = context.n_inverse_;

            temp_complex = DeviceVector<COMPLEX>(n);
            special_root = static_cast<COMPLEX_C>(2.0) *
                           static_cast<COMPLEX_C>(M_PI) /
                           static_cast<COMPLEX_C>(fft_length);
            COMPLEX_C j(0.0, 1.0); // Define the complex unit (imaginary part)
            COMPLEX_C one(1.0); // Define the complex unit (imaginary part)

            // forward fft root table generation
            std::vector<COMPLEX_C> special_root_tables;
            for (int i = 0; i < fft_length; i++)
            {
                COMPLEX_C element =
                    std::exp(j * static_cast<COMPLEX_C>(i) * special_root);
                special_root_tables.push_back(element);
            }

            // inverse fft root table generation
            std::vector<COMPLEX_C> special_inverse_root_tables;
            for (int i = 0; i < fft_length; i++)
            {
                COMPLEX_C element = one / special_root_tables[i];
                special_inverse_root_tables.push_back(element);
            }

            std::vector<int> rot_group;
            rot_group.push_back(1);
            for (int i = 1; i < slot_count_; i++)
            {
                rot_group.push_back((5 * rot_group[i - 1]) % fft_length);
            }

            std::vector<COMPLEX_C> new_ordered_root_tables(slot_count_,
                                                           COMPLEX_C(0));
            for (int logm = 1; logm <= log_slot_count_; ++logm)
            {
                int idx_mod = 1 << (logm + 2);
                int gap = fft_length / idx_mod;

                int offset = 1 << (logm - 1);
                for (int i = 0; i < (1 << (logm - 1)); ++i)
                {
                    int rou_idx = (rot_group[i] % idx_mod) * gap;
                    new_ordered_root_tables[offset + i] =
                        special_root_tables[rou_idx];
                }
            }

            std::vector<COMPLEX_C> new_ordered_inverse_root_tables(
                slot_count_, COMPLEX_C(0));
            for (int logm = log_slot_count_; logm > 0; logm--)
            {
                int idx_mod = 1 << (logm + 2);
                int gap = fft_length / idx_mod;

                int offset = 1 << (logm - 1);
                for (int i = 0; i < (1 << (logm - 1)); ++i)
                {
                    int rou_idx = (rot_group[i] % idx_mod) * gap;
                    new_ordered_inverse_root_tables[offset + i] =
                        special_inverse_root_tables[rou_idx];
                }
            }

            special_fft_roots_table_ = std::make_shared<DeviceVector<COMPLEX>>(
                new_ordered_root_tables, sizeof(COMPLEX));

            special_ifft_roots_table_ = std::make_shared<DeviceVector<COMPLEX>>(
                new_ordered_inverse_root_tables, sizeof(COMPLEX));

            std::vector<int> bit_reverse_vec(slot_count_);
            for (int i = 0; i < slot_count_; i++)
            {
                bit_reverse_vec[i] = bitreverse(i, log_slot_count_);
            }

            reverse_order =
                std::make_shared<DeviceVector<int>>(bit_reverse_vec);

            Mi_ = context.Mi_;
            Mi_inv_ = context.Mi_inv_;
            upper_half_threshold_ = context.upper_half_threshold_;
            decryption_modulus_ = context.decryption_modulus_;
        }
    }

    ///////////////////////////////////////////////////

    __host__ void HEEncoder::encode_bfv(Plaintext& plain,
                                        const std::vector<uint64_t>& message)
    {
        DeviceVector<Data> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), message.data(),
                   message.size() * sizeof(Data), cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256>>>(
            plain.data(), message_gpu.data(), encoding_location_->data(),
            plain_modulus_->data(), message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_plain_inverse_->data(),
            .stream = 0};

        GPU_NTT_Inplace(plain.data(), plain_intt_tables_->data(),
                        plain_modulus_->data(), cfg_intt, 1, 1);
    }

    __host__ void HEEncoder::encode_bfv(Plaintext& plain,
                                        const std::vector<uint64_t>& message,
                                        HEStream& stream)
    {
        DeviceVector<Data> message_gpu(slot_count_, stream.stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Data), cudaMemcpyHostToDevice,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            plain.data(), message_gpu.data(), encoding_location_->data(),
            plain_modulus_->data(), message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_plain_inverse_->data(),
            .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), plain_intt_tables_->data(),
                        plain_modulus_->data(), cfg_intt, 1, 1);
    }

    __host__ void HEEncoder::encode_bfv(Plaintext& plain,
                                        const std::vector<int64_t>& message)
    {
        DeviceVector<Data> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), message.data(),
                   message.size() * sizeof(Data), cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256>>>(
            plain.data(), message_gpu.data(), encoding_location_->data(),
            plain_modulus_->data(), message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_plain_inverse_->data(),
            .stream = 0};

        GPU_NTT_Inplace(plain.data(), plain_intt_tables_->data(),
                        plain_modulus_->data(), cfg_intt, 1, 1);
    }

    __host__ void HEEncoder::encode_bfv(Plaintext& plain,
                                        const std::vector<int64_t>& message,
                                        HEStream& stream)
    {
        DeviceVector<Data> message_gpu(slot_count_, stream.stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Data), cudaMemcpyHostToDevice,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            plain.data(), message_gpu.data(), encoding_location_->data(),
            plain_modulus_->data(), message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_plain_inverse_->data(),
            .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), plain_intt_tables_->data(),
                        plain_modulus_->data(), cfg_intt, 1, 1);
    }

    //

    __host__ void HEEncoder::encode_bfv(Plaintext& plain,
                                        const HostVector<uint64_t>& message)
    {
        DeviceVector<Data> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), message.data(),
                   message.size() * sizeof(Data), cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256>>>(
            plain.data(), message_gpu.data(), encoding_location_->data(),
            plain_modulus_->data(), message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_plain_inverse_->data(),
            .stream = 0};

        GPU_NTT_Inplace(plain.data(), plain_intt_tables_->data(),
                        plain_modulus_->data(), cfg_intt, 1, 1);
    }

    __host__ void HEEncoder::encode_bfv(Plaintext& plain,
                                        const HostVector<uint64_t>& message,
                                        HEStream& stream)
    {
        DeviceVector<Data> message_gpu(slot_count_, stream.stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Data), cudaMemcpyHostToDevice,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            plain.data(), message_gpu.data(), encoding_location_->data(),
            plain_modulus_->data(), message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_plain_inverse_->data(),
            .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), plain_intt_tables_->data(),
                        plain_modulus_->data(), cfg_intt, 1, 1);
    }

    __host__ void HEEncoder::encode_bfv(Plaintext& plain,
                                        const HostVector<int64_t>& message)
    {
        DeviceVector<Data> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), message.data(),
                   message.size() * sizeof(Data), cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256>>>(
            plain.data(), message_gpu.data(), encoding_location_->data(),
            plain_modulus_->data(), message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_plain_inverse_->data(),
            .stream = 0};

        GPU_NTT_Inplace(plain.data(), plain_intt_tables_->data(),
                        plain_modulus_->data(), cfg_intt, 1, 1);
    }

    __host__ void HEEncoder::encode_bfv(Plaintext& plain,
                                        const HostVector<int64_t>& message,
                                        HEStream& stream)
    {
        DeviceVector<Data> message_gpu(slot_count_, stream.stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Data), cudaMemcpyHostToDevice,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            plain.data(), message_gpu.data(), encoding_location_->data(),
            plain_modulus_->data(), message.size());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_plain_inverse_->data(),
            .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), plain_intt_tables_->data(),
                        plain_modulus_->data(), cfg_intt, 1, 1);
    }

    //

    __host__ void HEEncoder::decode_bfv(std::vector<uint64_t>& message,
                                        Plaintext& plain)
    {
        DeviceVector<Data> message_gpu(slot_count_);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(plain.data(), plain_ntt_tables_->data(),
                        plain_modulus_->data(), cfg_ntt, 1, 1);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256>>>(
            message_gpu.data(), plain.data(), encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count_);

        cudaMemcpy(message.data(), message_gpu.data(),
                   slot_count_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_bfv(std::vector<uint64_t>& message,
                                        Plaintext& plain, HEStream& stream)
    {
        DeviceVector<Data> message_gpu(slot_count_, stream.stream);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), plain_ntt_tables_->data(),
                        plain_modulus_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            message_gpu.data(), plain.data(), encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_bfv(std::vector<int64_t>& message,
                                        Plaintext& plain)
    {
        DeviceVector<Data> message_gpu(slot_count_);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(plain.data(), plain_ntt_tables_->data(),
                        plain_modulus_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256>>>(
            message_gpu.data(), plain.data(), encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        unsigned_signed_convert<<<dim3((slot_count_ >> 8), 1, 1), 256>>>(
            message_gpu.data(), message_gpu.data(), plain_modulus_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count_);

        cudaMemcpy(message.data(), message_gpu.data(),
                   slot_count_ * sizeof(int64_t), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_bfv(std::vector<int64_t>& message,
                                        Plaintext& plain, HEStream& stream)
    {
        DeviceVector<Data> message_gpu(slot_count_, stream.stream);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), plain_ntt_tables_->data(),
                        plain_modulus_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            message_gpu.data(), plain.data(), encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        unsigned_signed_convert<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                  stream.stream>>>(
            message_gpu.data(), message_gpu.data(), plain_modulus_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(int64_t), cudaMemcpyDeviceToHost,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    //

    __host__ void HEEncoder::decode_bfv(HostVector<uint64_t>& message,
                                        Plaintext& plain)
    {
        DeviceVector<Data> message_gpu(slot_count_);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(plain.data(), plain_ntt_tables_->data(),
                        plain_modulus_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256>>>(
            message_gpu.data(), plain.data(), encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count_);

        cudaMemcpy(message.data(), message_gpu.data(),
                   slot_count_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_bfv(HostVector<uint64_t>& message,
                                        Plaintext& plain, HEStream& stream)
    {
        DeviceVector<Data> message_gpu(slot_count_, stream.stream);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), plain_ntt_tables_->data(),
                        plain_modulus_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            message_gpu.data(), plain.data(), encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_bfv(HostVector<int64_t>& message,
                                        Plaintext& plain)
    {
        DeviceVector<Data> message_gpu(slot_count_);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(plain.data(), plain_ntt_tables_->data(),
                        plain_modulus_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256>>>(
            message_gpu.data(), plain.data(), encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        unsigned_signed_convert<<<dim3((slot_count_ >> 8), 1, 1), 256>>>(
            message_gpu.data(), message_gpu.data(), plain_modulus_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count_);

        cudaMemcpy(message.data(), message_gpu.data(),
                   slot_count_ * sizeof(int64_t), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_bfv(HostVector<int64_t>& message,
                                        Plaintext& plain, HEStream& stream)
    {
        DeviceVector<Data> message_gpu(slot_count_, stream.stream);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), plain_ntt_tables_->data(),
                        plain_modulus_->data(), cfg_ntt, 1, 1);

        decode_kernel_bfv<<<dim3((n >> 8), 1, 1), 256, 0, stream.stream>>>(
            message_gpu.data(), plain.data(), encoding_location_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        unsigned_signed_convert<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                  stream.stream>>>(
            message_gpu.data(), message_gpu.data(), plain_modulus_->data());
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(int64_t), cudaMemcpyDeviceToHost,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    ///////////////////////////////////////////////////

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const std::vector<double>& message,
                                         const double scale)
    {
        DeviceVector<double> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), message.data(),
                   message.size() * sizeof(double), cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double_to_complex_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256>>>(
            message_gpu.data(), temp_complex.data());

        double fix = scale / static_cast<double>(slot_count_);

        fft::fft_configuration cfg_ifft = {.n_power = log_slot_count_,
                                           .ntt_type = fft::type::INVERSE,
                                           .mod_inverse = COMPLEX(fix, 0.0),
                                           .stream = 0};

        fft::GPU_Special_FFT(temp_complex.data(),
                             special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1),
                                        256>>>(
            plain.data(), temp_complex.data(), modulus_->data(), Q_size_,
            two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(plain.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_, Q_size_);

        plain.scale_ = scale;
    }

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const std::vector<double>& message,
                                         const double scale, HEStream& stream)
    {
        DeviceVector<double> message_gpu(slot_count_, stream.stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(double), cudaMemcpyHostToDevice,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double_to_complex_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream.stream>>>(message_gpu.data(),
                                                    stream.temp_complex.data());

        double fix = scale / static_cast<double>(slot_count_);

        fft::fft_configuration cfg_ifft = {.n_power = log_slot_count_,
                                           .ntt_type = fft::type::INVERSE,
                                           .mod_inverse = COMPLEX(fix, 0.0),
                                           .stream = stream.stream};

        fft::GPU_Special_FFT(stream.temp_complex.data(),
                             special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream.stream>>>(
            plain.data(), stream.temp_complex.data(), modulus_->data(), Q_size_,
            two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_, Q_size_);

        plain.scale_ = scale;
    }

    //

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const HostVector<double>& message,
                                         const double scale)
    {
        DeviceVector<double> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), message.data(),
                   message.size() * sizeof(double), cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double_to_complex_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256>>>(
            message_gpu.data(), temp_complex.data());

        double fix = scale / static_cast<double>(slot_count_);

        fft::fft_configuration cfg_ifft = {.n_power = log_slot_count_,
                                           .ntt_type = fft::type::INVERSE,
                                           .mod_inverse = COMPLEX(fix, 0.0),
                                           .stream = 0};

        fft::GPU_Special_FFT(temp_complex.data(),
                             special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1),
                                        256>>>(
            plain.data(), temp_complex.data(), modulus_->data(), Q_size_,
            two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(plain.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_, Q_size_);

        plain.scale_ = scale;
    }

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const HostVector<double>& message,
                                         const double scale, HEStream& stream)
    {
        DeviceVector<double> message_gpu(slot_count_, stream.stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(double), cudaMemcpyHostToDevice,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double_to_complex_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream.stream>>>(message_gpu.data(),
                                                    stream.temp_complex.data());

        double fix = scale / static_cast<double>(slot_count_);

        fft::fft_configuration cfg_ifft = {.n_power = log_slot_count_,
                                           .ntt_type = fft::type::INVERSE,
                                           .mod_inverse = COMPLEX(fix, 0.0),
                                           .stream = stream.stream};

        fft::GPU_Special_FFT(stream.temp_complex.data(),
                             special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream.stream>>>(
            plain.data(), stream.temp_complex.data(), modulus_->data(), Q_size_,
            two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_, Q_size_);

        plain.scale_ = scale;
    }

    //

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const std::vector<COMPLEX_C>& message,
                                         const double scale)
    {
        DeviceVector<COMPLEX> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), message.data(),
                   message.size() * sizeof(COMPLEX), cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double fix = scale / static_cast<double>(slot_count_);

        fft::fft_configuration cfg_ifft = {.n_power = log_slot_count_,
                                           .ntt_type = fft::type::INVERSE,
                                           .mod_inverse = COMPLEX(fix, 0.0),
                                           .stream = 0};

        fft::GPU_Special_FFT(message_gpu.data(),
                             special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1),
                                        256>>>(
            plain.data(), message_gpu.data(), modulus_->data(), Q_size_,
            two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(plain.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_, Q_size_);

        plain.scale_ = scale;
    }

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const std::vector<COMPLEX_C>& message,
                                         const double scale, HEStream& stream)
    {
        DeviceVector<COMPLEX> message_gpu(slot_count_, stream.stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(COMPLEX),
                        cudaMemcpyHostToDevice, stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double fix = scale / static_cast<double>(slot_count_);

        fft::fft_configuration cfg_ifft = {.n_power = log_slot_count_,
                                           .ntt_type = fft::type::INVERSE,
                                           .mod_inverse = COMPLEX(fix, 0.0),
                                           .stream = stream.stream};

        fft::GPU_Special_FFT(message_gpu.data(),
                             special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream.stream>>>(
            plain.data(), message_gpu.data(), modulus_->data(), Q_size_,
            two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_, Q_size_);

        plain.scale_ = scale;
    }

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const HostVector<COMPLEX_C>& message,
                                         const double scale)
    {
        DeviceVector<COMPLEX> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), message.data(),
                   message.size() * sizeof(COMPLEX), cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double fix = scale / static_cast<double>(slot_count_);

        fft::fft_configuration cfg_ifft = {.n_power = log_slot_count_,
                                           .ntt_type = fft::type::INVERSE,
                                           .mod_inverse = COMPLEX(fix, 0.0),
                                           .stream = 0};

        fft::GPU_Special_FFT(message_gpu.data(),
                             special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1),
                                        256>>>(
            plain.data(), message_gpu.data(), modulus_->data(), Q_size_,
            two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(plain.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_, Q_size_);

        plain.scale_ = scale;
    }

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const HostVector<COMPLEX_C>& message,
                                         const double scale, HEStream& stream)
    {
        DeviceVector<COMPLEX> message_gpu(slot_count_, stream.stream);
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(COMPLEX),
                        cudaMemcpyHostToDevice, stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double fix = scale / static_cast<double>(slot_count_);

        fft::fft_configuration cfg_ifft = {.n_power = log_slot_count_,
                                           .ntt_type = fft::type::INVERSE,
                                           .mod_inverse = COMPLEX(fix, 0.0),
                                           .stream = stream.stream};

        fft::GPU_Special_FFT(message_gpu.data(),
                             special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream.stream>>>(
            plain.data(), message_gpu.data(), modulus_->data(), Q_size_,
            two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_, Q_size_);

        plain.scale_ = scale;
    }

    //

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const double& message,
                                         const double scale)
    {
        double value = message * scale;

        encode_kernel_double_ckks_conversion<<<dim3((n >> 8), 1, 1), 256>>>(
            plain.data(), value, modulus_->data(), Q_size_, two_pow_64,
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const double& message,
                                         const double scale, HEStream& stream)
    {
        double value = message * scale;

        encode_kernel_double_ckks_conversion<<<dim3((n >> 8), 1, 1), 256, 0,
                                               stream.stream>>>(
            plain.data(), value, modulus_->data(), Q_size_, two_pow_64,
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const std::int64_t& message,
                                         const double scale)
    {
        double value = static_cast<double>(message) * scale;

        encode_kernel_double_ckks_conversion<<<dim3((n >> 8), 1, 1), 256>>>(
            plain.data(), value, modulus_->data(), Q_size_, two_pow_64,
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::encode_ckks(Plaintext& plain,
                                         const std::int64_t& message,
                                         const double scale, HEStream& stream)
    {
        double value = static_cast<double>(message) * scale;

        encode_kernel_double_ckks_conversion<<<dim3((n >> 8), 1, 1), 256, 0,
                                               stream.stream>>>(
            plain.data(), value, modulus_->data(), Q_size_, two_pow_64,
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    //////////////////////

    __host__ void HEEncoder::decode_ckks(std::vector<double>& message,
                                         Plaintext& plain)
    {
        int current_modulus_count = Q_size_ - plain.depth_;

        DeviceVector<double> message_gpu(slot_count_);

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(plain.data(), intt_table_->data(), modulus_->data(),
                        cfg_intt, current_modulus_count, current_modulus_count);

        int counter = Q_size_;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256>>>(
            temp_complex.data(), plain.data(), modulus_->data(),
            Mi_inv_->data() + location1, Mi_->data() + location2,
            upper_half_threshold_->data() + location1,
            decryption_modulus_->data() + location1, current_modulus_count,
            plain.scale_, two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        fft::fft_configuration cfg_fft = {.n_power = log_slot_count_,
                                          .ntt_type = fft::type::FORWARD,
                                          .stream = 0};

        fft::GPU_Special_FFT(temp_complex.data(),
                             special_fft_roots_table_->data(), cfg_fft, 1);

        complex_to_double_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256>>>(
            temp_complex.data(), message_gpu.data());

        message.resize(slot_count_);

        cudaMemcpy(message.data(), message_gpu.data(),
                   slot_count_ * sizeof(double), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_ckks(std::vector<double>& message,
                                         Plaintext& plain, HEStream& stream)
    {
        int current_modulus_count = Q_size_ - plain.depth_;

        DeviceVector<double> message_gpu(slot_count_, stream.stream);

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), intt_table_->data(), modulus_->data(),
                        cfg_intt, current_modulus_count, current_modulus_count);

        int counter = Q_size_;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                stream.stream>>>(
            stream.temp_complex.data(), plain.data(), modulus_->data(),
            Mi_inv_->data() + location1, Mi_->data() + location2,
            upper_half_threshold_->data() + location1,
            decryption_modulus_->data() + location1, current_modulus_count,
            plain.scale_, two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        fft::fft_configuration cfg_fft = {.n_power = log_slot_count_,
                                          .ntt_type = fft::type::FORWARD,
                                          .stream = stream.stream};

        fft::GPU_Special_FFT(stream.temp_complex.data(),
                             special_fft_roots_table_->data(), cfg_fft, 1);

        complex_to_double_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream.stream>>>(stream.temp_complex.data(),
                                                    message_gpu.data());

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(double), cudaMemcpyDeviceToHost,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_ckks(HostVector<double>& message,
                                         Plaintext& plain)
    {
        int current_modulus_count = Q_size_ - plain.depth_;

        DeviceVector<double> message_gpu(slot_count_);

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(plain.data(), intt_table_->data(), modulus_->data(),
                        cfg_intt, current_modulus_count, current_modulus_count);

        int counter = Q_size_;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256>>>(
            temp_complex.data(), plain.data(), modulus_->data(),
            Mi_inv_->data() + location1, Mi_->data() + location2,
            upper_half_threshold_->data() + location1,
            decryption_modulus_->data() + location1, current_modulus_count,
            plain.scale_, two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        fft::fft_configuration cfg_fft = {.n_power = log_slot_count_,
                                          .ntt_type = fft::type::FORWARD,
                                          .stream = 0};

        fft::GPU_Special_FFT(temp_complex.data(),
                             special_fft_roots_table_->data(), cfg_fft, 1);

        complex_to_double_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256>>>(
            temp_complex.data(), message_gpu.data());

        message.resize(slot_count_);

        cudaMemcpy(message.data(), message_gpu.data(),
                   slot_count_ * sizeof(double), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_ckks(HostVector<double>& message,
                                         Plaintext& plain, HEStream& stream)
    {
        int current_modulus_count = Q_size_ - plain.depth_;

        DeviceVector<double> message_gpu(slot_count_, stream.stream);

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), intt_table_->data(), modulus_->data(),
                        cfg_intt, current_modulus_count, current_modulus_count);

        int counter = Q_size_;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                stream.stream>>>(
            stream.temp_complex.data(), plain.data(), modulus_->data(),
            Mi_inv_->data() + location1, Mi_->data() + location2,
            upper_half_threshold_->data() + location1,
            decryption_modulus_->data() + location1, current_modulus_count,
            plain.scale_, two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        fft::fft_configuration cfg_fft = {.n_power = log_slot_count_,
                                          .ntt_type = fft::type::FORWARD,
                                          .stream = stream.stream};

        fft::GPU_Special_FFT(stream.temp_complex.data(),
                             special_fft_roots_table_->data(), cfg_fft, 1);

        complex_to_double_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream.stream>>>(stream.temp_complex.data(),
                                                    message_gpu.data());

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(double), cudaMemcpyDeviceToHost,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    //

    __host__ void HEEncoder::decode_ckks(std::vector<COMPLEX_C>& message,
                                         Plaintext& plain)
    {
        int current_modulus_count = Q_size_ - plain.depth_;

        DeviceVector<COMPLEX> message_gpu(slot_count_);

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(plain.data(), intt_table_->data(), modulus_->data(),
                        cfg_intt, current_modulus_count, current_modulus_count);

        int counter = Q_size_;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256>>>(
            message_gpu.data(), plain.data(), modulus_->data(),
            Mi_inv_->data() + location1, Mi_->data() + location2,
            upper_half_threshold_->data() + location1,
            decryption_modulus_->data() + location1, current_modulus_count,
            plain.scale_, two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        fft::fft_configuration cfg_fft = {.n_power = log_slot_count_,
                                          .ntt_type = fft::type::FORWARD,
                                          .stream = 0};

        fft::GPU_Special_FFT(message_gpu.data(),
                             special_fft_roots_table_->data(), cfg_fft, 1);

        message.resize(slot_count_);

        cudaMemcpy(message.data(), message_gpu.data(),
                   slot_count_ * sizeof(COMPLEX), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_ckks(std::vector<COMPLEX_C>& message,
                                         Plaintext& plain, HEStream& stream)
    {
        int current_modulus_count = Q_size_ - plain.depth_;

        DeviceVector<COMPLEX> message_gpu(slot_count_, stream.stream);

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), intt_table_->data(), modulus_->data(),
                        cfg_intt, current_modulus_count, current_modulus_count);

        int counter = Q_size_;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                stream.stream>>>(
            message_gpu.data(), plain.data(), modulus_->data(),
            Mi_inv_->data() + location1, Mi_->data() + location2,
            upper_half_threshold_->data() + location1,
            decryption_modulus_->data() + location1, current_modulus_count,
            plain.scale_, two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        fft::fft_configuration cfg_fft = {.n_power = log_slot_count_,
                                          .ntt_type = fft::type::FORWARD,
                                          .stream = stream.stream};

        fft::GPU_Special_FFT(message_gpu.data(),
                             special_fft_roots_table_->data(), cfg_fft, 1);

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(COMPLEX), cudaMemcpyDeviceToHost,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_ckks(HostVector<COMPLEX_C>& message,
                                         Plaintext& plain)
    {
        int current_modulus_count = Q_size_ - plain.depth_;

        DeviceVector<COMPLEX> message_gpu(slot_count_);

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(plain.data(), intt_table_->data(), modulus_->data(),
                        cfg_intt, current_modulus_count, current_modulus_count);

        int counter = Q_size_;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256>>>(
            message_gpu.data(), plain.data(), modulus_->data(),
            Mi_inv_->data() + location1, Mi_->data() + location2,
            upper_half_threshold_->data() + location1,
            decryption_modulus_->data() + location1, current_modulus_count,
            plain.scale_, two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        fft::fft_configuration cfg_fft = {.n_power = log_slot_count_,
                                          .ntt_type = fft::type::FORWARD,
                                          .stream = 0};

        fft::GPU_Special_FFT(message_gpu.data(),
                             special_fft_roots_table_->data(), cfg_fft, 1);

        message.resize(slot_count_);

        cudaMemcpy(message.data(), message_gpu.data(),
                   slot_count_ * sizeof(COMPLEX), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncoder::decode_ckks(HostVector<COMPLEX_C>& message,
                                         Plaintext& plain, HEStream& stream)
    {
        int current_modulus_count = Q_size_ - plain.depth_;

        DeviceVector<COMPLEX> message_gpu(slot_count_, stream.stream);

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(plain.data(), intt_table_->data(), modulus_->data(),
                        cfg_intt, current_modulus_count, current_modulus_count);

        int counter = Q_size_;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                stream.stream>>>(
            message_gpu.data(), plain.data(), modulus_->data(),
            Mi_inv_->data() + location1, Mi_->data() + location2,
            upper_half_threshold_->data() + location1,
            decryption_modulus_->data() + location1, current_modulus_count,
            plain.scale_, two_pow_64, reverse_order->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        fft::fft_configuration cfg_fft = {.n_power = log_slot_count_,
                                          .ntt_type = fft::type::FORWARD,
                                          .stream = stream.stream};

        fft::GPU_Special_FFT(message_gpu.data(),
                             special_fft_roots_table_->data(), cfg_fft, 1);

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(COMPLEX), cudaMemcpyDeviceToHost,
                        stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu