// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "rtf/operator.cuh"

template <typename T>
void display_matrix(const std::vector<T>& matrix, std::size_t row_size)
{
    const std::size_t display_count = 5;

    auto print_row = [&](std::size_t start_index)
    {
        std::cout << "    [";
        for (std::size_t i = 0; i < display_count; ++i)
        {
            std::cout << std::setw(3) << std::right << matrix[start_index + i]
                      << ",";
        }
        std::cout << std::setw(3) << " ...,";
        for (std::size_t i = row_size - display_count; i < row_size; ++i)
        {
            std::cout << std::setw(3) << matrix[start_index + i];
            if (i != row_size - 1)
            {
                std::cout << ",";
            }
            else
            {
                std::cout << " ]\n";
            }
        }
    };

    std::cout << "\n";
    print_row(0);
    print_row(row_size);
    std::cout << std::endl;
}

namespace heongpu
{
    __host__
    HEOperator<Scheme::RTF>::HEOperator(HEContext<Scheme::RTF>& context,
                                        HEEncoder<Scheme::RTF>& encoder)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;

        n = context.n;

        n_power = context.n_power;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;
        P_size_ = context.P_size;

        bsk_mod_count_ = context.bsk_modulus;

        modulus_ = context.modulus_;

        ntt_table_ = context.ntt_table_;

        intt_table_ = context.intt_table_;

        n_inverse_ = context.n_inverse_;

        last_q_modinv_ = context.last_q_modinv_;

        base_Bsk_ = context.base_Bsk_;

        bsk_ntt_tables_ = context.bsk_ntt_tables_;

        bsk_intt_tables_ = context.bsk_intt_tables_;

        bsk_n_inverse_ = context.bsk_n_inverse_;

        m_tilde_ = context.m_tilde_;

        base_change_matrix_Bsk_ = context.base_change_matrix_Bsk_;

        inv_punctured_prod_mod_base_array_ =
            context.inv_punctured_prod_mod_base_array_;

        base_change_matrix_m_tilde_ = context.base_change_matrix_m_tilde_;

        inv_prod_q_mod_m_tilde_ = context.inv_prod_q_mod_m_tilde_;

        inv_m_tilde_mod_Bsk_ = context.inv_m_tilde_mod_Bsk_;

        prod_q_mod_Bsk_ = context.prod_q_mod_Bsk_;

        inv_prod_q_mod_Bsk_ = context.inv_prod_q_mod_Bsk_;

        plain_modulus_ = context.plain_modulus_;

        base_change_matrix_q_ = context.base_change_matrix_q_;

        base_change_matrix_msk_ = context.base_change_matrix_msk_;

        inv_punctured_prod_mod_B_array_ =
            context.inv_punctured_prod_mod_B_array_;

        inv_prod_B_mod_m_sk_ = context.inv_prod_B_mod_m_sk_;

        prod_B_mod_q_ = context.prod_B_mod_q_;

        q_Bsk_merge_modulus_ = context.q_Bsk_merge_modulus_;

        q_Bsk_merge_ntt_tables_ = context.q_Bsk_merge_ntt_tables_;

        q_Bsk_merge_intt_tables_ = context.q_Bsk_merge_intt_tables_;

        q_Bsk_n_inverse_ = context.q_Bsk_n_inverse_;

        half_p_ = context.half_p_;

        half_mod_ = context.half_mod_;

        upper_threshold_ = context.upper_threshold_;

        upper_halfincrement_ = context.upper_halfincrement_;

        Q_mod_t_ = context.Q_mod_t_;

        coeeff_div_plainmod_ = context.coeeff_div_plainmod_;

        //////

        d = context.d;
        d_tilda = context.d_tilda;
        r_prime = context.r_prime;

        B_prime_ = context.B_prime_;
        B_prime_ntt_tables_ = context.B_prime_ntt_tables_;
        B_prime_intt_tables_ = context.B_prime_intt_tables_;
        B_prime_n_inverse_ = context.B_prime_n_inverse_;

        base_change_matrix_D_to_B_ = context.base_change_matrix_D_to_B_;
        base_change_matrix_B_to_D_ = context.base_change_matrix_B_to_D_;
        Mi_inv_D_to_B_ = context.Mi_inv_D_to_B_;
        Mi_inv_B_to_D_ = context.Mi_inv_B_to_D_;
        prod_D_to_B_ = context.prod_D_to_B_;
        prod_B_to_D_ = context.prod_B_to_D_;

        base_change_matrix_D_to_Q_tilda_ =
            context.base_change_matrix_D_to_Q_tilda_;
        Mi_inv_D_to_Q_tilda_ = context.Mi_inv_D_to_Q_tilda_;
        prod_D_to_Q_tilda_ = context.prod_D_to_Q_tilda_;

        I_j_ = context.I_j_;
        I_location_ = context.I_location_;
        Sk_pair_ = context.Sk_pair_;

        prime_vector_ = context.prime_vector_;

        std::vector<int> prime_loc;
        std::vector<int> input_loc;

        int counter = Q_size_;
        for (int i = 0; i < Q_size_ - 1; i++)
        {
            for (int j = 0; j < counter; j++)
            {
                prime_loc.push_back(j);
            }
            counter--;
            for (int j = 0; j < P_size_; j++)
            {
                prime_loc.push_back(Q_size_ + j);
            }
        }

        counter = Q_prime_size_;
        for (int i = 0; i < Q_prime_size_ - 1; i++)
        {
            int sum = counter - 1;
            for (int j = 0; j < 2; j++)
            {
                input_loc.push_back(sum);
                sum += counter;
            }
            counter--;
        }

        new_prime_locations_ = DeviceVector<int>(prime_loc);
        new_input_locations_ = DeviceVector<int>(input_loc);
        new_prime_locations = new_prime_locations_.data();
        new_input_locations = new_input_locations_.data();

        // Encode params
        slot_count_ = encoder.slot_count_;
        plain_modulus_pointer_ = encoder.plain_modulus_;
        n_plain_inverse_ = encoder.n_plain_inverse_;
        plain_intt_tables_ = encoder.plain_intt_tables_;
        encoding_location_ = encoder.encoding_location_;
    }

    __host__ void HEOperator<Scheme::RTF>::add(Ciphertext<Scheme::RTF>& input1,
                                               Ciphertext<Scheme::RTF>& input2,
                                               Ciphertext<Scheme::RTF>& output,
                                               const ExecutionOptions& options)
    {
        if (input1.relinearization_required_ !=
            input2.relinearization_required_)
        {
            throw std::invalid_argument("Ciphertexts can not be added because "
                                        "ciphertext sizes have to be equal!");
        }

        if (input1.in_ntt_domain_ != input2.in_ntt_domain_)
        {
            throw std::invalid_argument(
                "Both Ciphertexts should be in same domain");
        }

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * Q_size_) ||
            input2.memory_size() < (cipher_size * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::RTF>& input1_)
            {
                input_storage_manager(
                    input2,
                    [&](Ciphertext<Scheme::RTF>& input2_)
                    {
                        output_storage_manager(
                            output,
                            [&](Ciphertext<Scheme::RTF>& output_)
                            {
                                DeviceVector<Data64> output_memory(
                                    (cipher_size * n * Q_size_),
                                    options.stream_);

                                addition<<<dim3((n >> 8), Q_size_, cipher_size),
                                           256, 0, options.stream_>>>(
                                    input1_.data(), input2_.data(),
                                    output_memory.data(), modulus_->data(),
                                    n_power);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                output_.scheme_ = scheme_;
                                output_.ring_size_ = n;
                                output_.coeff_modulus_count_ = Q_size_;
                                output_.cipher_size_ = cipher_size;
                                output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                                output_.relinearization_required_ =
                                    input1_.relinearization_required_;
                                output_.ciphertext_generated_ = true;

                                output_.memory_set(std::move(output_memory));
                            },
                            options);
                    },
                    options, (&input2 == &output));
            },
            options, (&input1 == &output));
    }

    __host__ void HEOperator<Scheme::RTF>::sub(Ciphertext<Scheme::RTF>& input1,
                                               Ciphertext<Scheme::RTF>& input2,
                                               Ciphertext<Scheme::RTF>& output,
                                               const ExecutionOptions& options)
    {
        if (input1.relinearization_required_ !=
            input2.relinearization_required_)
        {
            throw std::invalid_argument("Ciphertexts can not be added because "
                                        "ciphertext sizes have to be equal!");
        }

        if (input1.in_ntt_domain_ != input2.in_ntt_domain_)
        {
            throw std::invalid_argument(
                "Both Ciphertexts should be in same domain");
        }

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * Q_size_) ||
            input2.memory_size() < (cipher_size * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::RTF>& input1_)
            {
                input_storage_manager(
                    input2,
                    [&](Ciphertext<Scheme::RTF>& input2_)
                    {
                        output_storage_manager(
                            output,
                            [&](Ciphertext<Scheme::RTF>& output_)
                            {
                                DeviceVector<Data64> output_memory(
                                    (cipher_size * n * Q_size_),
                                    options.stream_);

                                substraction<<<dim3((n >> 8), Q_size_,
                                                    cipher_size),
                                               256, 0, options.stream_>>>(
                                    input1_.data(), input2_.data(),
                                    output_memory.data(), modulus_->data(),
                                    n_power);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                output_.scheme_ = scheme_;
                                output_.ring_size_ = n;
                                output_.coeff_modulus_count_ = Q_size_;
                                output_.cipher_size_ = cipher_size;
                                output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                                output_.relinearization_required_ =
                                    input1_.relinearization_required_;
                                output_.ciphertext_generated_ = true;

                                output_.memory_set(std::move(output_memory));
                            },
                            options);
                    },
                    options, (&input2 == &output));
            },
            options, (&input1 == &output));
    }

    __host__ void
    HEOperator<Scheme::RTF>::negate(Ciphertext<Scheme::RTF>& input1,
                                    Ciphertext<Scheme::RTF>& output,
                                    const ExecutionOptions& options)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        input_storage_manager(
            input1,
            [&](Ciphertext<Scheme::RTF>& input1_)
            {
                output_storage_manager(
                    output,
                    [&](Ciphertext<Scheme::RTF>& output_)
                    {
                        DeviceVector<Data64> output_memory(
                            (cipher_size * n * Q_size_), options.stream_);

                        negation<<<dim3((n >> 8), Q_size_, cipher_size), 256, 0,
                                   options.stream_>>>(
                            input1_.data(), output_memory.data(),
                            modulus_->data(), n_power);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        output_.scheme_ = scheme_;
                        output_.ring_size_ = n;
                        output_.coeff_modulus_count_ = Q_size_;
                        output_.cipher_size_ = cipher_size;
                        output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                        output_.relinearization_required_ =
                            input1_.relinearization_required_;
                        output_.ciphertext_generated_ = true;

                        output_.memory_set(std::move(output_memory));
                    },
                    options);
            },
            options, (&input1 == &output));
    }

    __host__ void HEOperator<Scheme::RTF>::add_plain_bfv(
        Ciphertext<Scheme::RTF>& input1, Plaintext<Scheme::RTF>& input2,
        Ciphertext<Scheme::RTF>& output, const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        DeviceVector<Data64> output_memory((cipher_size * n * Q_size_), stream);

        addition_plain_bfv_poly<<<dim3((n >> 8), Q_size_, cipher_size), 256, 0,
                                  stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            modulus_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::add_plain_bfv_inplace(
        Ciphertext<Scheme::RTF>& input1, Plaintext<Scheme::RTF>& input2,
        const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        addition_plain_bfv_poly_inplace<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                          stream>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::RTF>::sub_plain_bfv(
        Ciphertext<Scheme::RTF>& input1, Plaintext<Scheme::RTF>& input2,
        Ciphertext<Scheme::RTF>& output, const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        DeviceVector<Data64> output_memory((cipher_size * n * Q_size_), stream);

        substraction_plain_bfv_poly<<<dim3((n >> 8), Q_size_, cipher_size), 256,
                                      0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            modulus_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::sub_plain_bfv_inplace(
        Ciphertext<Scheme::RTF>& input1, Plaintext<Scheme::RTF>& input2,
        const cudaStream_t stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        substraction_plain_bfv_poly_inplace<<<dim3((n >> 8), Q_size_, 1), 256,
                                              0, stream>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::RTF>::multiply_bfv(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& input2,
        Ciphertext<Scheme::RTF>& output, const cudaStream_t stream)
    {
        if ((input1.in_ntt_domain_ != false) ||
            (input2.in_ntt_domain_ != false))
        {
            throw std::invalid_argument("Ciphertexts should be in same domain");
        }

        if (input1.memory_size() < (2 * n * Q_size_) ||
            input2.memory_size() < (2 * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        DeviceVector<Data64> output_memory((3 * n * Q_size_), stream);

        DeviceVector<Data64> temp_mul((4 * n * (bsk_mod_count_ + Q_size_)) +
                                          (3 * n * (bsk_mod_count_ + Q_size_)),
                                      stream);
        Data64* temp1_mul = temp_mul.data();
        Data64* temp2_mul = temp1_mul + (4 * n * (bsk_mod_count_ + Q_size_));

        fast_convertion<<<dim3((n >> 8), 4, 1), 256, 0, stream>>>(
            input1.data(), input2.data(), temp1_mul, modulus_->data(),
            base_Bsk_->data(), m_tilde_, inv_prod_q_mod_m_tilde_,
            inv_m_tilde_mod_Bsk_->data(), prod_q_mod_Bsk_->data(),
            base_change_matrix_Bsk_->data(),
            base_change_matrix_m_tilde_->data(),
            inv_punctured_prod_mod_base_array_->data(), n_power, Q_size_,
            bsk_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = q_Bsk_n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_mul, q_Bsk_merge_ntt_tables_->data(),
                                q_Bsk_merge_modulus_->data(), cfg_ntt,
                                ((bsk_mod_count_ + Q_size_) * 4),
                                (bsk_mod_count_ + Q_size_));

        cross_multiplication<<<dim3((n >> 8), (bsk_mod_count_ + Q_size_), 1),
                               256, 0, stream>>>(
            temp1_mul, temp1_mul + (((bsk_mod_count_ + Q_size_) * 2) * n),
            temp2_mul, q_Bsk_merge_modulus_->data(), n_power,
            (bsk_mod_count_ + Q_size_));
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp2_mul, q_Bsk_merge_intt_tables_->data(),
                                q_Bsk_merge_modulus_->data(), cfg_intt,
                                (3 * (bsk_mod_count_ + Q_size_)),
                                (bsk_mod_count_ + Q_size_));

        fast_floor<<<dim3((n >> 8), 3, 1), 256, 0, stream>>>(
            temp2_mul, output_memory.data(), modulus_->data(),
            base_Bsk_->data(), plain_modulus_,
            inv_punctured_prod_mod_base_array_->data(),
            base_change_matrix_Bsk_->data(), inv_prod_q_mod_Bsk_->data(),
            inv_punctured_prod_mod_B_array_->data(),
            base_change_matrix_q_->data(), base_change_matrix_msk_->data(),
            inv_prod_B_mod_m_sk_, prod_B_mod_q_->data(), n_power, Q_size_,
            bsk_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::multiply_plain_bfv(
        Ciphertext<Scheme::RTF>& input1, Plaintext<Scheme::RTF>& input2,
        Ciphertext<Scheme::RTF>& output, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        if (input1.in_ntt_domain_)
        {
            cipherplain_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
                input1.data(), input2.data(), output_memory.data(),
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> temp_plain_mul(n * Q_size_, stream);
            Data64* temp1_plain_mul = temp_plain_mul.data();

            threshold_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                input2.data(), temp1_plain_mul, modulus_->data(),
                upper_halfincrement_->data(), upper_threshold_, n_power,
                Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                .n_power = n_power,
                .ntt_type = gpuntt::INVERSE,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .mod_inverse = n_inverse_->data(),
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(temp1_plain_mul, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt, Q_size_,
                                    Q_size_);

            gpuntt::GPU_NTT(input1.data(), output_memory.data(),
                            ntt_table_->data(), modulus_->data(), cfg_ntt,
                            2 * Q_size_, Q_size_);

            cipherplain_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
                output_memory.data(), temp1_plain_mul, output_memory.data(),
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::GPU_NTT_Inplace(output_memory.data(), intt_table_->data(),
                                    modulus_->data(), cfg_intt, 2 * Q_size_,
                                    Q_size_);
        }

        output.memory_set(std::move(output_memory));
    }

    __host__ heongpu::Ciphertext<heongpu::Scheme::RTF>
    heongpu::HEOperator<heongpu::Scheme::RTF>::multiply_matrix(
        heongpu::Ciphertext<heongpu::Scheme::RTF>& cipher,
        std::vector<heongpu::DeviceVector<Data64>>& matrix, // concatenated diags
        std::vector<std::vector<int>>& diags, // per-group rotation list
        heongpu::Galoiskey<heongpu::Scheme::RTF>& galois_key,
        const ExecutionOptions& options)
    {
        cudaStream_t stream = options.stream_;
        if (stream == cudaStreamDefault)
            stream = cipher.stream();

        ExecutionOptions opt = ExecutionOptions()
                                   .set_stream(stream)
                                   .set_storage_type(storage_type::DEVICE)
                                   .set_initial_location(true);

        heongpu::Ciphertext<heongpu::Scheme::RTF> ct = cipher;
        if (ct.in_ntt_domain_)
        {
            transform_from_ntt_inplace(ct, opt);
        }
        if (ct.relinearization_required_)
        {
            throw std::invalid_argument("multiply_matrix: relinearize first.");
        }

        const int N = static_cast<int>(n);

        heongpu::Ciphertext<heongpu::Scheme::RTF> acc_ntt =
            operator_ciphertext(stream);
        bool first_term = true;

        for (size_t g = 0; g < diags.size(); ++g)
        {
            if (diags[g].empty())
                continue;
            auto& blob = matrix[g];

            const size_t num_diags = diags[g].size(); // 16
            const size_t per_diag_elems = blob.size() / num_diags; // must be N
            if (per_diag_elems < static_cast<size_t>(N))
            {
                throw std::invalid_argument(
                    "multiply_matrix: diagonal size too small.");
            }
            Data64* base_ptr = blob.data();

            for (size_t k = 0; k < num_diags; ++k)
            {
                int rot = diags[g][k];

                heongpu::Ciphertext<heongpu::Scheme::RTF> rotated = ct;
                if (rot != 0)
                {
                    rotate_rows_inplace(rotated, galois_key, rot, opt);
                }

                // Load this diagonal (coeff domain)
                heongpu::Plaintext<heongpu::Scheme::RTF> pt_diag;
                {
                    const size_t diag_len = static_cast<size_t>(N);
                    heongpu::DeviceVector<Data64> tmp(diag_len, stream);

                    HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                        tmp.data(), base_ptr + k * per_diag_elems,
                        diag_len * sizeof(Data64), cudaMemcpyDeviceToDevice,
                        stream));

                    pt_diag.memory_set(std::move(tmp));
                    pt_diag.scheme_ = scheme_;
                    pt_diag.plain_size_ = static_cast<int>(diag_len);
                    pt_diag.in_ntt_domain_ = false; // coeff form
                    pt_diag.plaintext_generated_ = true;
                }

                // NTT multiply-accumulate
                transform_to_ntt_inplace(rotated, opt);
                transform_to_ntt_inplace(pt_diag, opt);
                multiply_plain_inplace(rotated, pt_diag, opt);

                if (first_term)
                {
                    acc_ntt = rotated;
                    first_term = false;
                }
                else
                {
                    add_inplace(acc_ntt, rotated, opt);
                }
            }
        }

        if (!first_term)
        {
            transform_from_ntt_inplace(acc_ntt, opt);
        }
        return acc_ntt;
    }

    // Helper inside HEOperator<Scheme::RTF> (or a free function with access to n, Q_size_, tables)
    __host__ heongpu::DeviceVector<Data64>
    HEOperator<Scheme::RTF>::batched_plain_to_ntt_slab(
        const Data64* base_plain,                 // big blob of all plain diagonals
        const std::vector<int>& k_list,           // size B: which diagonals to pick
        cudaStream_t stream)
    {
        const int N = static_cast<int>(n);
        const int Q = static_cast<int>(Q_size_);
        const int B = static_cast<int>(k_list.size());

        heongpu::DeviceVector<Data64> empty(0, stream);
        if (B <= 0) return empty;

        // 1) k_list 업로드
        
        nvtxRangeId_t rangeA = nvtxRangeStartA("upload k_list");
        heongpu::DeviceVector<int> d_k_list(B, stream);
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
            d_k_list.data(), k_list.data(), B * sizeof(int),
            cudaMemcpyHostToDevice, stream));
        nvtxRangeEnd(rangeA);

        // 2) gather+threshold → poly-major in [(B*Q)*N]
        nvtxRangeId_t rangeB = nvtxRangeStartA("gather+threshold");
        heongpu::DeviceVector<Data64> in_polyBQ_N((size_t)B * Q * N, stream);
        {
            dim3 grid((N + 255) / 256, Q, B), blk(256);
            gather_threshold_to_polymajor_kernel<<<grid, blk, 0, stream>>>(
                base_plain, d_k_list.data(), in_polyBQ_N.data(),
                modulus_->data(), upper_halfincrement_->data(), upper_threshold_,
                n_power, Q, B);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        nvtxRangeEnd(rangeB);

        // 3) RNS batched NTT: batch_size=B*Q, mod_count=Q (옵션 1의 핵심)
        heongpu::DeviceVector<Data64> out_polyBQ_N((size_t)B * Q * N, stream);
        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = nullptr, // forward에선 사용 안 함
            .stream = stream
        };

        gpuntt::GPU_NTT(
            /*device_in=*/in_polyBQ_N.data(),
            /*device_out=*/out_polyBQ_N.data(),
            /*root_of_unity_table=*/ntt_table_->data(),  // 전체 테이블 베이스 포인터 하나면 충분
            /*modulus=*/modulus_->data(),                // [Q]
            /*cfg=*/cfg_ntt,
            /*batch_size=*/B * Q,
            /*mod_count=*/Q);
        GPUNTT_CUDA_CHECK(cudaGetLastError());

        // 4) poly-major → slab [B, Q*N]
        heongpu::DeviceVector<Data64> slab_B_QN((size_t)B * (Q * N), stream);
        {
            dim3 grid2((N + 255) / 256, Q, B), blk2(256);
            repack_poly_to_slab_kernel<<<grid2, blk2, 0, stream>>>(
                out_polyBQ_N.data(), slab_B_QN.data(),
                n_power, Q, B);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        return slab_B_QN;
    }


    __host__ heongpu::DeviceVector<Data64>
     heongpu::HEOperator<heongpu::Scheme::RTF>::pack_baby_steps_all_hoisted_ntt(
        heongpu::HEOperator<heongpu::Scheme::RTF>& op,            // 연산자 (rotate/NTT 호출용)
        heongpu::Ciphertext<heongpu::Scheme::RTF>  input,         // current_input_ct (by value; 내부에서 변형)
        int g2,                                                   // baby-step 개수
        heongpu::Galoiskey<heongpu::Scheme::RTF>& galois_key,
        Modulus64* modulus_array,                                 // ← gpuntt::Modulus<Data64>* (비-const)
        Root<Data64>* root_table,                         // ← gpuntt::Root<Data64>*
        int n_power, int Q,                                       // 링/모듈 수
        cudaStream_t stream)
    {
        const int    N          = 1 << n_power;
        const size_t per_ct_len = (size_t)Q << (n_power + 1); // 2 * N * Q

        auto local_opt = heongpu::ExecutionOptions()
                            .set_stream(stream)
                            .set_storage_type(heongpu::storage_type::DEVICE)
                            .set_initial_location(true);

        // 0) 입력을 계수영역으로
        if (input.in_ntt_domain_) {
            op.transform_from_ntt_inplace(input, local_opt);
        }

        // 1) 회전 결과를 계수영역 버퍼에 이어붙이기
        heongpu::DeviceVector<Data64> baby_coeff_concat(per_ct_len * g2, stream);

        // Prepare once per ciphertext:
        BFVKeySwitchHoistedMethodI hoisted;
        op.prepare_keyswitch_hoisted_method_I(input, hoisted, stream);

        nvtxRangeId_t rangeA = nvtxRangeStartA("Copy");

        // j = 0 : 그대로 복사
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
            baby_coeff_concat.data() + 0 * per_ct_len,
            input.data(),
            per_ct_len * sizeof(Data64),
            cudaMemcpyDeviceToDevice, stream));

        // j = 1..g2-1 : **hoisted** rotations
        for (int j = 1; j < g2; ++j) {
            heongpu::Ciphertext<heongpu::Scheme::RTF> rotated;
            op.rotate_rows_method_I_hoisted(hoisted, input, rotated, galois_key, j, stream);

            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                baby_coeff_concat.data() + (size_t)j * per_ct_len,
                rotated.data(),
                per_ct_len * sizeof(Data64),
                cudaMemcpyDeviceToDevice, stream));
        }
        nvtxRangeEnd(rangeA);

        // 2) RNS batched NTT: batch_size = g2 * 2 * Q, mod_count = Q (한 번 호출)
        heongpu::DeviceVector<Data64> baby_ntt_concat(per_ct_len * g2, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = nullptr,  // forward에선 사용 안 함
            .stream = stream
        };

        nvtxRangeId_t rangeB = nvtxRangeStartA("Batched NTT");
        gpuntt::GPU_NTT(
            /*device_in=*/baby_coeff_concat.data(),
            /*device_out=*/baby_ntt_concat.data(),
            /*root_of_unity_table=*/root_table,      // gpuntt::Root<Data64>*
            /*modulus=*/modulus_array,               // Modulus64*  (= gpuntt::Modulus<Data64>*)
            /*cfg=*/cfg_ntt,
            /*batch_size=*/g2 * 2 * Q,
            /*mod_count=*/Q);
        GPUNTT_CUDA_CHECK(cudaGetLastError());
        nvtxRangeEnd(rangeB);

        // 3) 결과(=NTT 후 baby-steps 이어붙인 것)를 그대로 packed_baby로 리턴
        return baby_ntt_concat;
    }


    __host__ void HEOperator<Scheme::RTF>::apply_galois_method_I_with_hoisted(
        BFVKeySwitchHoistedMethodI& ws,
        Ciphertext<Scheme::RTF>& output,
        Galoiskey<Scheme::RTF>& galois_key,
        int galois_elt,
        const cudaStream_t stream)
    {
        if (!ws.prepared) {
            throw std::logic_error("Hoisted Method I workspace not prepared.");
        }

        // Local aliases
        Data64* temp0_rotation = const_cast<Data64*>(ws.temp0_dup.data());   // used read-only here
        Data64* temp1_rotation = const_cast<Data64*>(ws.temp1_ntt.data());   // used read-only here

        // Per-rotation scratch (accumulators in P)
        Data64* temp2_rotation = ws.temp2_acc.data();

        // Output buffer (Q-domain, 2 components)
        heongpu::DeviceVector<Data64> output_memory((size_t)2 * n * Q_size_, stream);

        // === Multiply-accumulate in NTT domain with the *rotation-specific* key ===
        int iteration_count_1 = Q_size_ / 4;
        int iteration_count_2 = Q_size_ % 4;

        if (galois_key.storage_type_ == storage_type::DEVICE) {
            keyswitch_multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_rotation,
                galois_key.device_location_[galois_elt].data(),
                temp2_rotation,
                modulus_->data(),
                n_power,
                Q_prime_size_,
                iteration_count_1,
                iteration_count_2
            );
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        } else {
            heongpu::DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt],
                stream
            );
            keyswitch_multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_rotation,
                key_location.data(),
                temp2_rotation,
                modulus_->data(),
                n_power,
                Q_prime_size_,
                iteration_count_1,
                iteration_count_2
            );
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        // === Inverse NTT over P ===
        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream
        };

        gpuntt::GPU_NTT_Inplace(
            temp2_rotation,
            intt_table_->data(),
            modulus_->data(),
            cfg_intt,
            2 * Q_prime_size_,   // two components, each over |P|
            Q_prime_size_
        );
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // === Mod-down + logical permutation (depends on galois_elt) ===
        divide_round_lastq_permute_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            temp2_rotation,            // from P
            temp0_rotation,            // dup of original c0 (and/or needed constants)
            output_memory.data(),      // to Q
            modulus_->data(),
            half_p_->data(),
            half_mod_->data(),
            last_q_modinv_->data(),
            galois_elt,
            n_power,
            Q_prime_size_,
            Q_size_,
            P_size_
        );
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // Hand back as a ciphertext object (same as your original)
        output.memory_set(std::move(output_memory));
        output.scheme_ = scheme_;
        output.ring_size_ = n;
        output.coeff_modulus_count_ = Q_size_;
        output.cipher_size_ = 2;
        output.in_ntt_domain_ = false;
        output.relinearization_required_ = false;
        output.ciphertext_generated_ = true;
    }

    __host__ void HEOperator<Scheme::RTF>::prepare_keyswitch_hoisted_method_I(
        Ciphertext<Scheme::RTF>& input_ct,
        BFVKeySwitchHoistedMethodI& ws,
        const cudaStream_t stream)
    {
        if (input_ct.in_ntt_domain_ != false) {
            throw std::invalid_argument("Ciphertext should be in INT domain for Method I.");
        }
        if (input_ct.memory_size() < (2 * n * Q_size_)) {
            throw std::invalid_argument("Invalid ciphertext size for prepare.");
        }

        // Allocate once
        ws.temp0_dup.resize((size_t)2 * n * Q_size_, stream);
        ws.temp1_ntt.resize((size_t)n * Q_size_ * Q_prime_size_, stream);
        ws.temp2_acc.resize((size_t)2 * n * Q_prime_size_, stream);

        Data64* temp0_rotation = ws.temp0_dup.data();
        Data64* temp1_rotation = ws.temp1_ntt.data();

        // 1) Duplicate/layout (c0 → temp0_dup, c1 → expanded across P into temp1_ntt)
        bfv_duplicate_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            input_ct.data(),
            temp0_rotation,                 // output1 (z==0)
            temp1_rotation,                 // output2 (z>0; laid out as [Q][P][N])
            modulus_->data(),
            n_power,
            Q_prime_size_ /* rns_mod_count = |P| */
        );
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // 2) Forward NTT on temp1_ntt (Q * P blocks)
        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream
        };

        gpuntt::GPU_NTT_Inplace(
            temp1_rotation,
            ntt_table_->data(),
            modulus_->data(),
            cfg_ntt,
            Q_size_ * Q_prime_size_,   // number of length-N transforms
            Q_prime_size_              // modulus-stride (your NTT wrapper’s convention)
        );
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ws.prepared = true;
    }

    __host__ void HEOperator<Scheme::RTF>::rotate_rows_method_I_hoisted(
        BFVKeySwitchHoistedMethodI& ws,                    // prepared once
        Ciphertext<Scheme::RTF>& input1,                   // same ct used in prepare
        Ciphertext<Scheme::RTF>& output,                   // per-rotation result
        Galoiskey<Scheme::RTF>& galois_key,
        int shift,
        const cudaStream_t stream)
    {
        if (!ws.prepared) {
            throw std::logic_error("Hoisted workspace not prepared.");
        }
        if (input1.relinearization_required_) {
            throw std::invalid_argument("Ciphertext cannot be rotated (relin required).");
        }
        if (input1.in_ntt_domain_ != false) {
            throw std::invalid_argument("Ciphertext should be in INT domain for Method I.");
        }

        if (shift == 0) {
            output = input1;
            return;
        }

        int galoiselt = steps_to_galois_elt(shift, n, galois_key.group_order_);

        bool key_exist = (galois_key.storage_type_ == storage_type::DEVICE)
            ? (galois_key.device_location_.find(galoiselt) != galois_key.device_location_.end())
            : (galois_key.host_location_.find(galoiselt) != galois_key.host_location_.end());

        if (key_exist) {
            apply_galois_method_I_with_hoisted(ws, output, galois_key, galoiselt, stream);
            return;
        }

        // Decompose shift into powers-of-two rotations (as your original)
        std::vector<int> required_galoiselt;
        int shift_num = std::abs(shift);
        int negative = (shift < 0) ? (-1) : 1;

        while (shift_num != 0) {
            int power = int(std::log2(shift_num));
            int power_2 = 1 << power;
            shift_num -= power_2;

            int index_in = power_2 * negative;

            if (!(galois_key.galois_elt.find(index_in) != galois_key.galois_elt.end())) {
                throw std::logic_error("Galois key not present!");
            }
            required_galoiselt.push_back(galois_key.galois_elt[index_in]);
        }

        // Chain rotations using *the same hoisted ws*.
        Ciphertext<Scheme::RTF> *in_ptr = &input1;
        Ciphertext<Scheme::RTF>  mid, *out_ptr = &output;

        for (size_t k = 0; k < required_galoiselt.size(); ++k) {
            // write into mid (except last -> output)
            bool last = (k + 1 == required_galoiselt.size());
            out_ptr = last ? &output : &mid;

            apply_galois_method_I_with_hoisted(ws, *out_ptr, galois_key, required_galoiselt[k], stream);

            // next hop
            in_ptr = out_ptr;
        }
    }


    __host__ static heongpu::DeviceVector<Data64> pack_baby_steps_all(
        const std::vector<heongpu::Ciphertext<heongpu::Scheme::RTF>>& baby_steps,
        int Q_size_, int n_power, cudaStream_t stream)
    {
        const size_t per_ct_len = static_cast<size_t>(Q_size_) << (n_power + 1); // 2 * N * Q
        const size_t total_len  = per_ct_len * baby_steps.size();

        heongpu::DeviceVector<Data64> packed(total_len, stream);

        for (size_t j = 0; j < baby_steps.size(); ++j) {
            const Data64* src = baby_steps[j].data(); 
            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                packed.data() + j * per_ct_len,
                src,
                per_ct_len * sizeof(Data64),
                cudaMemcpyDeviceToDevice,
                stream));
        }
        return packed;
    }

    __host__ heongpu::Ciphertext<heongpu::Scheme::RTF>
    heongpu::HEOperator<heongpu::Scheme::RTF>::multiply_matrix_bsgs(
        heongpu::Ciphertext<heongpu::Scheme::RTF>& input,
        const std::vector<std::vector<heongpu::DeviceVector<Data64>>>& matrix_groups_caller,
        const std::vector<std::vector<int>>& shifts_caller,
        heongpu::Galoiskey<heongpu::Scheme::RTF>& galois_key,
        const ExecutionOptions& opt)
    {
        nvtx3::scoped_range data_processing_range("bsgs");


        cudaStream_t stream = opt.stream_;
        if (stream == cudaStreamDefault) stream = input.stream();

        ExecutionOptions local_opt = ExecutionOptions()
                                        .set_stream(stream)
                                        .set_storage_type(storage_type::DEVICE)
                                        .set_initial_location(true);

        // 회전은 계수영역
        if (input.in_ntt_domain_) {
            transform_from_ntt_inplace(input, local_opt);
        }

        const int N  = static_cast<int>(n);
        const int H  = N >> 1;
        const int g2 = static_cast<int>(ceil(sqrt(static_cast<double>(H))));
        const int Q  = static_cast<int>(Q_size_);

        const size_t per_diag_len_plain = static_cast<size_t>(N);                 // PLAIN
        const size_t per_ct_len         = static_cast<size_t>(Q) << (n_power + 1);// 2*N*Q

        // 원본 + column-swapped
        heongpu::Ciphertext<heongpu::Scheme::RTF> ct_orig   = input;
        heongpu::Ciphertext<heongpu::Scheme::RTF> ct_swapped;
        rotate_columns(input, ct_swapped, galois_key, local_opt);

        std::vector<heongpu::Ciphertext<heongpu::Scheme::RTF>> inputs;
        inputs.reserve(2);
        inputs.emplace_back(std::move(ct_orig));
        inputs.emplace_back(std::move(ct_swapped));

        heongpu::Ciphertext<heongpu::Scheme::RTF> final_result;
        bool result_initialized = false;

        for (int group_idx = 0; group_idx < 2; ++group_idx) {
            auto& current_input_ct = inputs[group_idx];

            nvtxRangeId_t rangeBS = nvtxRangeStartA("BS");
            auto packed_baby = pack_baby_steps_all_hoisted_ntt(
                *this,
                current_input_ct,              // 회전의 기준
                g2,
                galois_key,
                modulus_->data(),              // Modulus64*   (비-const 포인터)
                ntt_table_->data(),            // Root<Data64>*
                n_power,
                (int)Q_size_,
                stream);
            nvtxRangeEnd(rangeBS);

            // (B) 이제는 PLAIN 대각선 블롭을 읽는다
            const auto& blob_plain   = matrix_groups_caller[group_idx][0];   // PLAIN diag들
            const auto& shifts       = shifts_caller[group_idx];
            const Data64* base_plain = reinterpret_cast<const Data64*>(blob_plain.data());

            // (C) giant index별 버킷
            struct Item { int j; int k; };
            std::unordered_map<int, std::vector<Item>> buckets;
            buckets.reserve(shifts.size());
            for (int k = 0; k < static_cast<int>(shifts.size()); ++k) {
                const int s = shifts[k];
                const int r = (s < H) ? s : (s - H);
                const int j = r % g2;
                const int i = (r - j) / g2;
                buckets[i].push_back({j, k});
            }
            
            // (D) plain 대각선용 디바이스 버퍼 1개를 미리 잡아두고(길이 N) 매 아이템마다 래핑
            heongpu::DeviceVector<Data64> diag_plain_buf(static_cast<size_t>(N), stream);

            heongpu::Ciphertext<heongpu::Scheme::RTF> group_result;
            bool group_result_initialized = false;

            // (각 버킷 반복)
            for (auto& kv : buckets) {
                nvtxRangeId_t rangeGS = nvtxRangeStartA("GS");

                const int i_giant = kv.first;
                auto& items = kv.second;
                if (items.empty()) continue;

                // NTT domain 누적자 초기화 (0)
                heongpu::Ciphertext<heongpu::Scheme::RTF> inner_sum = operator_ciphertext(0);
                inner_sum.scheme_               = scheme_;
                inner_sum.ring_size_            = n;
                inner_sum.coeff_modulus_count_  = Q_size_;
                inner_sum.cipher_size_          = 2;
                inner_sum.in_ntt_domain_        = true;
                inner_sum.ciphertext_generated_ = true;
                const size_t per_ct_len = (size_t)Q_size_ << (n_power + 1); // 2*N*Q
                HEONGPU_CUDA_CHECK(cudaMemsetAsync(inner_sum.data(), 0, per_ct_len * sizeof(Data64), stream));

                // 배치 버퍼(재사용 가능하게 밖으로 올려도 OK)
                const int kBatch = g2; //
                std::vector<int>  h_j_of_batch(kBatch);
                std::vector<int>  k_list; k_list.reserve(kBatch);
                heongpu::DeviceVector<int> d_j_of_batch(kBatch, stream);

                for (int start = 0; start < (int)items.size(); start += kBatch) {
                    const int B = std::min(kBatch, (int)items.size() - start);

                    k_list.clear();
                    for (int m = 0; m < B; ++m) {
                        h_j_of_batch[m] = items[start + m].j;  // baby-step index
                        k_list.push_back(items[start + m].k);  // which plain diag
                    }
                    
                    nvtxRangeId_t rangeA = nvtxRangeStartA("batch copy diag");
                    HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                        d_j_of_batch.data(), h_j_of_batch.data(), B*sizeof(int),
                        cudaMemcpyHostToDevice, stream));
                    nvtxRangeEnd(rangeA);

                    // === 핵심: 이번 배치의 [B]개 PLAIN 대각선을 RNS-NTT로 한 번에 ===
                    heongpu::DeviceVector<Data64> diags_ntt_slab =
                        batched_plain_to_ntt_slab(/*base_plain=*/base_plain, /*k_list=*/k_list, stream);
                    // diags_ntt_slab: [B, Q*N] (batch-major)

                    // === 한 번의 MAC 커널로 B개 대각선 누적 ===
                    nvtxRangeId_t rangeC = nvtxRangeStartA("accum");
                    dim3 grid((N + 255) / 256, Q, 2), blk(256);
                    cipherplain_multiply_accumulate_block_kernel<<<grid, blk, 0, stream>>>(
                        packed_baby.data(),            // [g2 * (2*N*Q)]
                        diags_ntt_slab.data(),         // [B * (Q*N)]
                        d_j_of_batch.data(),           // [B], 각 대각선의 baby-step j
                        inner_sum.data(),              // 누적 대상 (NTT domain)
                        modulus_->data(),              // [Q]
                        /*M=*/B, (int)Q_size_, n_power);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                    nvtxRangeEnd(rangeC);
                }

                
                nvtxRangeId_t rangeD = nvtxRangeStartA("rotate and add");
                // coeff domain 복귀, giant 회전, 그룹 합산
                transform_from_ntt_inplace(inner_sum, local_opt);
                const int giant_rotation = i_giant * g2;
                if (giant_rotation) rotate_rows_inplace(inner_sum, galois_key, giant_rotation, local_opt);

                if (!group_result_initialized) {
                    group_result = std::move(inner_sum);
                    group_result_initialized = true;
                } else {
                    add_inplace(group_result, inner_sum, local_opt);
                }
                nvtxRangeEnd(rangeD);
                nvtxRangeEnd(rangeGS);
            }

            // (F) 그룹 누적
            if (group_result_initialized) {
                if (!result_initialized) {
                    final_result = std::move(group_result);
                    result_initialized = true;
                } else {
                    add_inplace(final_result, group_result, local_opt);
                }
            }
        }

        return final_result;
    }

    __host__ void HEOperator<Scheme::RTF>::relinearize_seal_method_inplace(
        Ciphertext<Scheme::RTF>& input1, Relinkey<Scheme::RTF>& relin_key,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> temp_relin(
            (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_), stream);
        Data64* temp1_relin = temp_relin.data();
        Data64* temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

        cipher_broadcast_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            input1.data() + (Q_size_ << (n_power + 1)), temp1_relin,
            modulus_->data(), n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_relin, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);

        int iteration_count_1 = Q_size_ / 4;
        int iteration_count_2 = Q_size_ % 4;
        // TODO: make it efficient
        if (relin_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                n_power, Q_prime_size_, iteration_count_1, iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                n_power, Q_prime_size_, iteration_count_1, iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_relin, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        divide_round_lastq_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                    stream>>>(
            temp2_relin, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEOperator<Scheme::RTF>::relinearize_external_product_method_inplace(
        Ciphertext<Scheme::RTF>& input1, Relinkey<Scheme::RTF>& relin_key,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> temp_relin_new((n * d * r_prime) +
                                                (2 * n * d_tilda * r_prime) +
                                                (2 * n * Q_prime_size_),
                                            stream);
        Data64* temp1_relin_new = temp_relin_new.data();
        Data64* temp2_relin_new = temp1_relin_new + (n * d * r_prime);
        Data64* temp3_relin_new = temp2_relin_new + (2 * n * d_tilda * r_prime);

        base_conversion_DtoB_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                            stream>>>(
            input1.data() + (Q_size_ << (n_power + 1)), temp1_relin_new,
            modulus_->data(), B_prime_->data(),
            base_change_matrix_D_to_B_->data(), Mi_inv_D_to_B_->data(),
            prod_D_to_B_->data(), I_j_->data(), I_location_->data(), n_power,
            Q_size_, d_tilda, d, r_prime);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_relin_new, B_prime_ntt_tables_->data(),
                                B_prime_->data(), cfg_ntt, d * r_prime,
                                r_prime);

        // TODO: make it efficient
        if (relin_key.storage_type_ == storage_type::DEVICE)
        {
            multiply_accumulate_extended_kernel<<<
                dim3((n >> 8), r_prime, d_tilda), 256, 0, stream>>>(
                temp1_relin_new, relin_key.data(), temp2_relin_new,
                B_prime_->data(), n_power, d_tilda, d, r_prime);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            multiply_accumulate_extended_kernel<<<
                dim3((n >> 8), r_prime, d_tilda), 256, 0, stream>>>(
                temp1_relin_new, key_location.data(), temp2_relin_new,
                B_prime_->data(), n_power, d_tilda, d, r_prime);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = B_prime_n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_relin_new, B_prime_intt_tables_->data(),
                                B_prime_->data(), cfg_intt,
                                2 * r_prime * d_tilda, r_prime);

        base_conversion_BtoD_relin_kernel<<<dim3((n >> 8), d_tilda, 2), 256, 0,
                                            stream>>>(
            temp2_relin_new, temp3_relin_new, modulus_->data(),
            B_prime_->data(), base_change_matrix_B_to_D_->data(),
            Mi_inv_B_to_D_->data(), prod_B_to_D_->data(), I_j_->data(),
            I_location_->data(), n_power, Q_prime_size_, d_tilda, d, r_prime);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        divide_round_lastq_extended_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                             stream>>>(
            temp3_relin_new, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEOperator<Scheme::RTF>::relinearize_external_product_method2_inplace(
        Ciphertext<Scheme::RTF>& input1, Relinkey<Scheme::RTF>& relin_key,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> temp_relin(
            (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_), stream);
        Data64* temp1_relin = temp_relin.data();
        Data64* temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

        base_conversion_DtoQtilde_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                                 stream>>>(
            input1.data() + (Q_size_ << (n_power + 1)), temp1_relin,
            modulus_->data(), base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_relin, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d * Q_prime_size_,
                                Q_prime_size_);

        // TODO: make it efficient
        int iteration_count_1 = d / 4;
        int iteration_count_2 = d % 4;
        if (relin_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                n_power, Q_prime_size_, iteration_count_1, iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                n_power, Q_prime_size_, iteration_count_1, iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_relin, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        divide_round_lastq_extended_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                             stream>>>(
            temp2_relin, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator<Scheme::RTF>::rotate_method_I(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        Galoiskey<Scheme::RTF>& galois_key, int shift,
        const cudaStream_t stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n, galois_key.group_order_);
        bool key_exist = (galois_key.storage_type_ == storage_type::DEVICE)
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        if (key_exist)
        {
            apply_galois_method_I(input1, output, galois_key, galoiselt,
                                  stream);
        }
        else
        {
            std::vector<int> required_galoiselt;
            int shift_num = abs(shift);
            int negative = (shift < 0) ? (-1) : 1;
            while (shift_num != 0)
            {
                int power = int(log2(shift_num));
                int power_2 = pow(2, power);
                shift_num = shift_num - power_2;

                int index_in = power_2 * negative;

                if (!(galois_key.galois_elt.find(index_in) !=
                      galois_key.galois_elt.end()))
                {
                    throw std::logic_error("Galois key not present!");
                }
                galoiselt = galois_key.galois_elt[index_in];
                required_galoiselt.push_back(galoiselt);
            }

            Ciphertext<Scheme::RTF>& in_data = input1;
            for (auto& galois_elt : required_galoiselt)
            {
                apply_galois_method_I(in_data, output, galois_key, galois_elt,
                                      stream);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator<Scheme::RTF>::rotate_method_II(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        Galoiskey<Scheme::RTF>& galois_key, int shift,
        const cudaStream_t stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n, galois_key.group_order_);
        bool key_exist = (galois_key.storage_type_ == storage_type::DEVICE)
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        if (key_exist)
        {
            apply_galois_method_II(input1, output, galois_key, galoiselt,
                                   stream);
        }
        else
        {
            std::vector<int> required_galoiselt;
            int shift_num = abs(shift);
            int negative = (shift < 0) ? (-1) : 1;
            while (shift_num != 0)
            {
                int power = int(log2(shift_num));
                int power_2 = pow(2, power);
                shift_num = shift_num - power_2;

                int index_in = power_2 * negative;

                if (!(galois_key.galois_elt.find(index_in) !=
                      galois_key.galois_elt.end()))
                {
                    throw std::logic_error("Galois key not present!");
                }
                galoiselt = galois_key.galois_elt[index_in];
                required_galoiselt.push_back(galoiselt);
            }

            Ciphertext<Scheme::RTF>& in_data = input1;
            for (auto& galois_elt : required_galoiselt)
            {
                apply_galois_method_II(in_data, output, galois_key, galois_elt,
                                       stream);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator<Scheme::RTF>::apply_galois_method_I(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        Galoiskey<Scheme::RTF>& galois_key, int galois_elt,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) +
                                               (n * Q_size_ * Q_prime_size_) +
                                               (2 * n * Q_prime_size_),
                                           stream);
        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_ * Q_prime_size_);

        bfv_duplicate_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 = Q_size_ / 4;
        int iteration_count_2 = Q_size_ % 4;
        if (galois_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_rotation, galois_key.device_location_[galois_elt].data(),
                temp2_rotation, modulus_->data(), n_power, Q_prime_size_,
                iteration_count_1, iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt], stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_rotation, key_location.data(), temp2_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // ModDown + Permute
        divide_round_lastq_permute_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            temp2_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galois_elt, n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::apply_galois_method_II(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        Galoiskey<Scheme::RTF>& galois_key, int galois_elt,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) + (n * Q_size_) +
                                               (2 * n * Q_prime_size_ * d) +
                                               (2 * n * Q_prime_size_),
                                           stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (2 * n * Q_prime_size_ * d);

        // TODO: make it efficient
        global_memory_replace_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                       stream>>>(input1.data(), temp0_rotation,
                                                 n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        base_conversion_DtoQtilde_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                                 stream>>>(
            input1.data() + (Q_size_ << n_power), temp2_rotation,
            modulus_->data(), base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d * Q_prime_size_,
                                Q_prime_size_);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 = d / 4;
        int iteration_count_2 = d % 4;
        if (galois_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, galois_key.device_location_[galois_elt].data(),
                temp3_rotation, modulus_->data(), n_power, Q_prime_size_,
                iteration_count_1, iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt], stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp3_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        // ModDown + Permute
        divide_round_lastq_permute_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galois_elt, n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::rotate_columns_method_I(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        Galoiskey<Scheme::RTF>& galois_key, const cudaStream_t stream)
    {
        int galoiselt = galois_key.galois_elt_zero;

        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) +
                                               (n * Q_size_ * Q_prime_size_) +
                                               (2 * n * Q_prime_size_),
                                           stream);
        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_ * Q_prime_size_);

        bfv_duplicate_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 = Q_size_ / 4;
        int iteration_count_2 = Q_size_ % 4;
        if (galois_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_rotation, galois_key.c_data(), temp2_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(galois_key.zero_host_location_,
                                              stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_rotation, key_location.data(), temp2_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        // ModDown + Permute
        divide_round_lastq_permute_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            temp2_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galoiselt, n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::rotate_columns_method_II(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        Galoiskey<Scheme::RTF>& galois_key, const cudaStream_t stream)
    {
        int galoiselt = galois_key.galois_elt_zero;

        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) + (n * Q_size_) +
                                               (2 * n * Q_prime_size_ * d) +
                                               (2 * n * Q_prime_size_),
                                           stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (2 * n * Q_prime_size_ * d);

        // TODO: make it efficient
        global_memory_replace_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                       stream>>>(input1.data(), temp0_rotation,
                                                 n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        base_conversion_DtoQtilde_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                                 stream>>>(
            input1.data() + (Q_size_ << n_power), temp2_rotation,
            modulus_->data(), base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d * Q_prime_size_,
                                Q_prime_size_);

        // MultSum
        // TODO: make it efficient
        int iteration_count_1 = d / 4;
        int iteration_count_2 = d % 4;
        if (galois_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, galois_key.c_data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(galois_key.zero_host_location_,
                                              stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp3_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        // ModDown + Permute
        divide_round_lastq_permute_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galoiselt, n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::switchkey_method_I(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        Switchkey<Scheme::RTF>& switch_key, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) +
                                               (n * Q_size_ * Q_prime_size_) +
                                               (2 * n * Q_prime_size_),
                                           stream);
        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_ * Q_prime_size_);

        cipher_broadcast_switchkey_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                            stream>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        int iteration_count_1 = Q_size_ / 4;
        int iteration_count_2 = Q_size_ % 4;
        if (switch_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_rotation, switch_key.data(), temp2_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(switch_key.host_location_,
                                              stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_rotation, key_location.data(), temp2_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        divide_round_lastq_switchkey_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                              0, stream>>>(
            temp2_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::switchkey_method_II(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        Switchkey<Scheme::RTF>& switch_key, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) + (n * Q_size_) +
                                               (2 * n * Q_prime_size_ * d) +
                                               (2 * n * Q_prime_size_),
                                           stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (2 * n * Q_prime_size_ * d);

        cipher_broadcast_switchkey_method_II_kernel<<<
            dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        base_conversion_DtoQtilde_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                                 stream>>>(
            temp1_rotation, temp2_rotation, modulus_->data(),
            base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d * Q_prime_size_,
                                Q_prime_size_);

        // TODO: make it efficient
        int iteration_count_1 = d / 4;
        int iteration_count_2 = d % 4;
        if (switch_key.storage_type_ == storage_type::DEVICE)
        {
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, switch_key.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(switch_key.host_location_,
                                              stream);
            keyswitch_multiply_accumulate_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, iteration_count_1,
                iteration_count_2);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp3_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        divide_round_lastq_extended_switchkey_kernel<<<
            dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::negacyclic_shift_poly_coeffmod(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        int index, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        DeviceVector<Data64> temp(2 * n * Q_size_, stream);

        negacyclic_shift_poly_coeffmod_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            input1.data(), temp.data(), modulus_->data(), index, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // TODO: do with efficient way!
        global_memory_replace_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                       stream>>>(temp.data(),
                                                 output_memory.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::transform_to_ntt_bfv_plain(
        Plaintext<Scheme::RTF>& input1, Plaintext<Scheme::RTF>& output,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(n * Q_size_, stream);

        DeviceVector<Data64> temp_plain_mul(n * Q_size_, stream);
        Data64* temp1_plain_mul = temp_plain_mul.data();

        threshold_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            input1.data(), temp1_plain_mul, modulus_->data(),
            upper_halfincrement_->data(), upper_threshold_, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(temp1_plain_mul, output_memory.data(),
                        ntt_table_->data(), modulus_->data(), cfg_ntt, Q_size_,
                        Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::transform_to_ntt_bfv_cipher(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), output_memory.data(), ntt_table_->data(),
                        modulus_->data(), cfg_ntt, 2 * Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator<Scheme::RTF>::transform_from_ntt_bfv_cipher(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), output_memory.data(),
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        2 * Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    //                       BOOTSRAPPING                         //
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    __host__ Ciphertext<Scheme::RTF>
    HEOperator<Scheme::RTF>::operator_ciphertext(cudaStream_t stream)
    {
        Ciphertext<Scheme::RTF> cipher;

        cipher.coeff_modulus_count_ = Q_size_;
        cipher.cipher_size_ = 2; // default
        cipher.ring_size_ = n; // n
        // cipher.depth_ = 0;

        cipher.scheme_ = scheme_;
        cipher.in_ntt_domain_ = true;
        cipher.storage_type_ = storage_type::DEVICE;

        // cipher.rescale_required_ = false;
        cipher.relinearization_required_ = false;
        // cipher.scale_ = scale;
        cipher.ciphertext_generated_ = true;

        // int cipher_memory_size = 2 * (Q_size_ - cipher.depth_) * n;
        int cipher_memory_size = 2 * Q_size_ * n;

        cipher.device_locations_ =
            DeviceVector<Data64>(cipher_memory_size, stream);

        return cipher;
    }

    __host__ Ciphertext<Scheme::RTF>
    HEOperator<Scheme::RTF>::operator_from_ciphertext(
        Ciphertext<Scheme::RTF>& input, cudaStream_t stream)
    {
        Ciphertext<Scheme::RTF> cipher;

        cipher.coeff_modulus_count_ = input.coeff_modulus_count_;
        cipher.cipher_size_ = input.cipher_size_;
        cipher.ring_size_ = input.ring_size_;

        cipher.scheme_ = input.scheme_;
        cipher.in_ntt_domain_ = input.in_ntt_domain_;

        cipher.storage_type_ = storage_type::DEVICE;

        cipher.relinearization_required_ = input.relinearization_required_;
        cipher.ciphertext_generated_ = true;

        int cipher_memory_size = 2 * Q_size_ * n;

        cipher.device_locations_ =
            DeviceVector<Data64>(cipher_memory_size, stream);

        return cipher;
    }

    __host__ heongpu::Plaintext<heongpu::Scheme::RTF>
    heongpu::HEOperator<heongpu::Scheme::RTF>::operator_plaintext(cudaStream_t stream)
    {
        const int N = static_cast<int>(n);

        DeviceVector<Data64> buf(static_cast<size_t>(N), stream);

        heongpu::Plaintext<heongpu::Scheme::RTF> pt(scheme_, buf, /*is_ntt=*/false);
        return pt;
    }

    HEArithmeticOperator<Scheme::RTF>::HEArithmeticOperator(
        HEContext<Scheme::RTF>& context, HEEncoder<Scheme::RTF>& encoder)
        : HEOperator<Scheme::RTF>(context, encoder)
    {
    }

    HELogicOperator<Scheme::RTF>::HELogicOperator(
        HEContext<Scheme::RTF>& context, HEEncoder<Scheme::RTF>& encoder)
        : HEOperator<Scheme::RTF>(context, encoder)
    {
        // TODO: make it efficinet
        Data64 constant_1 = 1ULL;
        encoded_constant_one_ = DeviceVector<Data64>(slot_count_);
        fill_device_vector<<<dim3((n >> 8), 1, 1), 256>>>(
            encoded_constant_one_.data(), constant_1, slot_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_plain_inverse_->data(),
            .stream = 0};

        gpuntt::GPU_NTT_Inplace(encoded_constant_one_.data(),
                                plain_intt_tables_->data(),
                                plain_modulus_pointer_->data(), cfg_intt, 1, 1);
    }

    __host__ void HELogicOperator<Scheme::RTF>::one_minus_cipher(
        Ciphertext<Scheme::RTF>& input1, Ciphertext<Scheme::RTF>& output,
        const ExecutionOptions& options)
    {
        // TODO: make it efficient
        negate_inplace(input1, options);

        addition_plain_bfv_poly<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                  options.stream_>>>(
            input1.data(), encoded_constant_one_.data(), output.data(),
            modulus_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HELogicOperator<Scheme::RTF>::one_minus_cipher_inplace(
        Ciphertext<Scheme::RTF>& input1, const ExecutionOptions& options)
    {
        // TODO: make it efficient
        negate_inplace(input1, options);

        addition_plain_bfv_poly_inplace<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                          options.stream_>>>(
            input1.data(), encoded_constant_one_.data(), input1.data(),
            modulus_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu
