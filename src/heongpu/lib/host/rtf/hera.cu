
#include "rtf/hera.cuh"


namespace heongpu
{
    __host__
    HEHERA<Scheme::RTF>::HEHERA(HEContext<Scheme::RTF>& context,
                                HEEncoder<Scheme::RTF>& encoder,
                                HEOperator<Scheme::RTF>& op)
        : context_(context), encoder_(encoder), operator_(op)
    {
        if (!context.context_generated_){
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

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::gen_stream_key(
        heongpu::DeviceVector<Data64>& nonce, Ciphertext<Scheme::RTF>& ctkey,
        const ExecutionOptions& options)
    {
        return;
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::add_round_key(
        Ciphertext<Scheme::RTF>& input, Ciphertext<Scheme::RTF>& round_key,
        Ciphertext<Scheme::RTF>& output, const ExecutionOptions& options)
    {
        operator_.add(input, round_key, output, options);
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::feistel(
        Ciphertext<Scheme::RTF>& input, Ciphertext<Scheme::RTF>& output,
        Galoiskey<Scheme::RTF>& galois_key, const ExecutionOptions& options)
    {
        // square, rotate and add
        auto stream = options.stream_;
        heongpu::Ciphertext<heongpu::Scheme::RTF> temp =
            operator_.operator_ciphertext(stream);

        operator_.multiply_bfv(input, input, temp, options.stream_);
        operator_.rotate_rows_inplace(temp, galois_key, -1, options); // right shift by 1
        operator_.add(input, temp, output, options);
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::cube(
        Ciphertext<Scheme::RTF>& input, Ciphertext<Scheme::RTF>& output,
        const ExecutionOptions& options)
    {
        operator_.multiply_bfv(input, input, output, options.stream_);
        operator_.multiply_bfv(output, input, output, options.stream_);
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::linear(
        heongpu::Ciphertext<heongpu::Scheme::RTF>& input,
        heongpu::Ciphertext<heongpu::Scheme::RTF>& output,
        heongpu::HEEncoder<heongpu::Scheme::RTF>& encoder,
        heongpu::HEContext<heongpu::Scheme::RTF>& context,
        heongpu::Galoiskey<heongpu::Scheme::RTF>& galois_key,
        const ExecutionOptions& options)
    {
        cudaStream_t stream = options.stream_;
        if (stream == cudaStreamDefault)
            stream = input.stream();

        ExecutionOptions opt = ExecutionOptions()
                                   .set_stream(stream)
                                   .set_storage_type(storage_type::DEVICE)
                                   .set_initial_location(true);

        if (input.relinearization_required_)
        {
            throw std::invalid_argument(
                "linear(): input has non-linear part; relinearize first.");
        }

        // Coefficient domain (row rotations)
        heongpu::Ciphertext<heongpu::Scheme::RTF> x = input;
        if (x.in_ntt_domain_)
            operator_.transform_from_ntt_inplace(x, opt);

        const size_t N = static_cast<size_t>(n);
        const size_t row_len = N >> 1; // BFV batching row length
        const size_t B = 16; // 16x16 block
        if (row_len % B != 0)
        {
            throw std::invalid_argument(
                "linear(): (n/2) must be a multiple of 16.");
        }
        const size_t blocks_per_row = row_len / B;

        static const uint64_t M16[16][16] = {
            {4, 6, 2, 2, 6, 9, 3, 3, 2, 3, 1, 1, 2, 3, 1, 1},
            {2, 4, 6, 2, 3, 6, 9, 3, 1, 2, 3, 1, 1, 2, 3, 1},
            {2, 2, 4, 6, 3, 3, 6, 9, 1, 1, 2, 3, 1, 1, 2, 3},
            {6, 2, 2, 4, 9, 3, 3, 6, 3, 1, 1, 2, 3, 1, 1, 2},
            {2, 3, 1, 1, 4, 6, 2, 2, 6, 9, 3, 3, 2, 3, 1, 1},
            {1, 2, 3, 1, 2, 4, 6, 2, 3, 6, 9, 3, 1, 2, 3, 1},
            {1, 1, 2, 3, 2, 2, 4, 6, 3, 3, 6, 9, 1, 1, 2, 3},
            {3, 1, 1, 2, 6, 2, 2, 4, 9, 3, 3, 6, 3, 1, 1, 2},
            {2, 3, 1, 1, 2, 3, 1, 1, 4, 6, 2, 2, 6, 9, 3, 3},
            {1, 2, 3, 1, 1, 2, 3, 1, 2, 4, 6, 2, 3, 6, 9, 3},
            {1, 1, 2, 3, 1, 1, 2, 3, 2, 2, 4, 6, 3, 3, 6, 9},
            {3, 1, 1, 2, 3, 1, 1, 2, 6, 2, 2, 4, 9, 3, 3, 6},
            {6, 9, 3, 3, 2, 3, 1, 1, 2, 3, 1, 1, 4, 6, 2, 2},
            {3, 6, 9, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 4, 6, 2},
            {3, 3, 6, 9, 1, 1, 2, 3, 1, 1, 2, 3, 2, 2, 4, 6},
            {9, 3, 3, 6, 3, 1, 1, 2, 3, 1, 1, 2, 6, 2, 2, 4}};

        std::vector<std::vector<int>> diags(1);
        diags[0].reserve(31);
        diags[0].push_back(0);
        // Swapped: Lower diagonals < 0 (left-rot), Upper diagonals > 0
        // (right-rot)
        for (int s = 1; s < (int) B; ++s)
            diags[0].push_back(-s);
        for (int s = 1; s < (int) B; ++s)
            diags[0].push_back(s);

        const size_t D = diags[0].size(); // 31
        heongpu::DeviceVector<Data64> blob(N * D, stream);

        // Temporary plaintext for encoding each diagonal
        heongpu::Plaintext<heongpu::Scheme::RTF> pt_diag(context);

        for (size_t k = 0; k < D; ++k)
        {
            const int rot = diags[0][k];
            std::vector<uint64_t> hdiag_vec(N, 0ULL);

            for (size_t row = 0; row < 2; ++row)
            {
                const size_t row_base = row * row_len;
                for (size_t blk = 0; blk < blocks_per_row; ++blk)
                {
                    const size_t base = row_base + blk * B;
                    if (rot == 0)
                    { // Main diagonal
                        for (int r = 0; r < (int) B; ++r)
                        {
                            hdiag_vec[base + r] = M16[r][r];
                        }
                    }
                    else if (rot > 0)
                    { // Upper diagonals (Shift > 0)
                        const int s = rot;
                        for (int r = 0; r < (int) B - s; ++r)
                        {
                            hdiag_vec[base + r] = M16[r][r + s];
                        }
                    }
                    else
                    { // Lower diagonals (Shift < 0)
                        const int s = -rot;
                        for (int r = s; r < (int) B; ++r)
                        {
                            hdiag_vec[base + r] = M16[r][r - s];
                        }
                    }
                }
            }

            // Encode the plaintext diagonal before copying to the blob
            encoder.encode(pt_diag, hdiag_vec);

            // Copy the properly encoded diagonal (from device) to the
            // concatenated blob (on device)
            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                blob.data() + k * N, pt_diag.data(), N * sizeof(Data64),
                cudaMemcpyDeviceToDevice, stream));
        }

        // Single group with 31 diagonals
        std::vector<heongpu::DeviceVector<Data64>> matrices;
        matrices.emplace_back(std::move(blob));

        auto y = operator_.multiply_matrix(x, matrices, diags, galois_key, opt);
        output = y;
    }


}