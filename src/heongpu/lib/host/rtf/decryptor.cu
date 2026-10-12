// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "rtf/decryptor.cuh"

namespace heongpu
{
    __host__
    HEDecryptor<Scheme::RTF>::HEDecryptor(HEContext<Scheme::RTF>& context,
                                          Secretkey<Scheme::RTF>& secret_key)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();
        offset_ = gen();

        if (secret_key.storage_type_ == storage_type::DEVICE)
        {
            secret_key_ = secret_key.device_locations_;
        }
        else
        {
            secret_key.store_in_device();
            secret_key_ = secret_key.device_locations_;
        }

        n = context.n;
        n_power = context.n_power;

        Q_size_ = context.Q_size;

        modulus_ = context.modulus_;

        ntt_table_ = context.ntt_table_;
        intt_table_ = context.intt_table_;

        n_inverse_ = context.n_inverse_;

        if (scheme_ == scheme_type::bfv)
        {
            plain_modulus_ = context.plain_modulus_;

            gamma_ = context.gamma_;

            Qi_t_ = context.Qi_t_;

            Qi_gamma_ = context.Qi_gamma_;

            Qi_inverse_ = context.Qi_inverse_;

            mulq_inv_t_ = context.mulq_inv_t_;

            mulq_inv_gamma_ = context.mulq_inv_gamma_;

            inv_gamma_ = context.inv_gamma_;

            // Noise budget calculation

            Mi_ = context.Mi_;
            Mi_inv_ = context.Mi_inv_;
            upper_half_threshold_ = context.upper_half_threshold_;
            decryption_modulus_ = context.decryption_modulus_;
            total_bit_count_ = context.total_bit_count_;
        }
    }

    __host__ void
    HEDecryptor<Scheme::RTF>::decrypt_bfv(Plaintext<Scheme::RTF>& plaintext,
                                          Ciphertext<Scheme::RTF>& ciphertext,
                                          const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(n, stream);

        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (Q_size_ << n_power);

        DeviceVector<Data64> temp_memory(2 * n * Q_size_, stream);
        Data64* ct0_temp = temp_memory.data();
        Data64* ct1_temp = temp_memory.data() + (Q_size_ << n_power);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};
        if (!ciphertext.in_ntt_domain_)
        {
            gpuntt::GPU_NTT(ct1, ct1_temp, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, Q_size_, Q_size_);

            sk_multiplication<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                ct1_temp, secret_key_.data(), ct1_temp, modulus_->data(),
                n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            sk_multiplication<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                ct1, secret_key_.data(), ct1_temp, modulus_->data(), n_power,
                Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        if (ciphertext.in_ntt_domain_)
        {
            // TODO: merge these NTTs
            gpuntt::GPU_NTT(ct0, ct0_temp, intt_table_->data(),
                            modulus_->data(), cfg_intt, Q_size_, Q_size_);

            gpuntt::GPU_NTT_Inplace(ct1_temp, intt_table_->data(),
                                    modulus_->data(), cfg_intt, Q_size_,
                                    Q_size_);

            ct0 = ct0_temp;
        }
        else
        {
            gpuntt::GPU_NTT_Inplace(ct1_temp, intt_table_->data(),
                                    modulus_->data(), cfg_intt, Q_size_,
                                    Q_size_);
        }

        decryption_kernel<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            ct0, ct1_temp, output_memory.data(), modulus_->data(),
            plain_modulus_, gamma_, Qi_t_->data(), Qi_gamma_->data(),
            Qi_inverse_->data(), mulq_inv_t_, mulq_inv_gamma_, inv_gamma_,
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plaintext.memory_set(std::move(output_memory));
    }

    __host__ void
    HEDecryptor<Scheme::RTF>::decryptx3_bfv(Plaintext<Scheme::RTF>& plaintext,
                                            Ciphertext<Scheme::RTF>& ciphertext,
                                            const cudaStream_t stream)
    {
        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (Q_size_ << n_power);
        Data64* ct2 = ciphertext.data() + (Q_size_ << (n_power + 1));

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(ct1, ntt_table_->data(), modulus_->data(),
                                cfg_ntt, 2 * Q_size_, Q_size_);

        sk_multiplicationx3<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            ct1, secret_key_.data(), modulus_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(ct1, intt_table_->data(), modulus_->data(),
                                cfg_intt, 2 * Q_size_, Q_size_);

        decryption_kernelx3<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            ct0, ct1, ct2, plaintext.data(), modulus_->data(), plain_modulus_,
            gamma_, Qi_t_->data(), Qi_gamma_->data(), Qi_inverse_->data(),
            mulq_inv_t_, mulq_inv_gamma_, inv_gamma_, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ int HEDecryptor<Scheme::RTF>::noise_budget_calculation(
        Ciphertext<Scheme::RTF>& ciphertext, const ExecutionOptions& options)
    {
        // Invariant noise budget as in SEAL: log2(Q) - log2(||t * (c0 + c1 s)
        // mod Q||_inf) - 1, with Q the ciphertext modulus (P excluded).
        //
        // The previous version read the result of an asynchronous copy into
        // pinned memory (HostVector) without synchronizing, so it saw zeros
        // and returned log2(Q) + 57 or so; its block reduction also only
        // looked at every 16th coefficient. The max-norm reduction is now
        // done on the host over all N coefficients.
        if (ciphertext.coeff_modulus_count() != Q_size_)
        {
            throw std::invalid_argument(
                "noise budget needs a ciphertext at the full modulus Q");
        }
        if (ciphertext.in_ntt_domain())
        {
            throw std::invalid_argument(
                "noise budget needs a ciphertext in coefficient form");
        }

        std::vector<Data64> composed(static_cast<size_t>(n) * Q_size_);
        std::vector<Data64> threshold(Q_size_), modulus_big(Q_size_);

        input_storage_manager(
            ciphertext,
            [&](Ciphertext<Scheme::RTF>& ciphertext_)
            {
                Data64* ct0 = ciphertext_.data();
                Data64* ct1 = ciphertext_.data() + (Q_size_ << n_power);

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = options.stream_};

                DeviceVector<Data64> temp_memory(n * Q_size_, options.stream_);
                gpuntt::GPU_NTT(ct1, temp_memory.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, Q_size_, Q_size_);

                sk_multiplication<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                    options.stream_>>>(
                    temp_memory.data(), secret_key_.data(), temp_memory.data(),
                    modulus_->data(), n_power, Q_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::INVERSE,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .mod_inverse = n_inverse_->data(),
                    .stream = options.stream_};

                gpuntt::GPU_NTT_Inplace(temp_memory.data(), intt_table_->data(),
                                        modulus_->data(), cfg_intt, Q_size_,
                                        Q_size_);

                coeff_multadd<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                options.stream_>>>(
                    ct0, temp_memory.data(), temp_memory.data(), plain_modulus_,
                    modulus_->data(), n_power, Q_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                // CRT composition: coefficient i becomes a Q_size-word
                // integer in [0, Q), word j at temp[i + (j << n_power)].
                compose_kernel<<<dim3((n >> 8), 1, 1), 256, 0,
                                 options.stream_>>>(
                    temp_memory.data(), temp_memory.data(), modulus_->data(),
                    Mi_inv_->data(), Mi_->data(), decryption_modulus_->data(),
                    Q_size_, n_power);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    composed.data(), temp_memory.data(),
                    composed.size() * sizeof(Data64), cudaMemcpyDeviceToHost,
                    options.stream_));
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    threshold.data(), upper_half_threshold_->data(),
                    Q_size_ * sizeof(Data64), cudaMemcpyDeviceToHost,
                    options.stream_));
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    modulus_big.data(), decryption_modulus_->data(),
                    Q_size_ * sizeof(Data64), cudaMemcpyDeviceToHost,
                    options.stream_));
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
            },
            options, false);

        // Little-endian multiword compare / subtract.
        auto compare = [&](const Data64* a, const Data64* b) -> int
        {
            for (int w = Q_size_ - 1; w >= 0; --w)
            {
                if (a[w] != b[w])
                    return (a[w] > b[w]) ? 1 : -1;
            }
            return 0;
        };

        std::vector<Data64> value(Q_size_), max_norm(Q_size_, 0);
        for (int i = 0; i < n; ++i)
        {
            for (int w = 0; w < Q_size_; ++w)
                value[w] = composed[i + (static_cast<size_t>(w) << n_power)];
            if (compare(value.data(), threshold.data()) >= 0)
            {
                // |value| = Q - value
                unsigned __int128 borrow = 0;
                for (int w = 0; w < Q_size_; ++w)
                {
                    unsigned __int128 d =
                        (unsigned __int128) modulus_big[w] - value[w] - borrow;
                    value[w] = static_cast<Data64>(d);
                    borrow = (d >> 64) ? 1 : 0;
                }
            }
            if (compare(value.data(), max_norm.data()) > 0)
                max_norm = value;
        }

        int norm_bits = 0;
        for (int w = Q_size_ - 1; w >= 0; --w)
        {
            if (max_norm[w] != 0)
            {
                norm_bits = 64 * w + (64 - __builtin_clzll(max_norm[w]));
                break;
            }
        }

        const int budget = total_bit_count_ - norm_bits - 1;
        return (budget > 0) ? budget : 0;
    }

} // namespace heongpu