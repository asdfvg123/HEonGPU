
#include "rtf/hera.cuh"


namespace heongpu
{
    __host__
    HEHERA<Scheme::RTF>::HEHERA(HEContext<Scheme::RTF>& context,
                                HEContext<Scheme::CKKS>& contextckks,
                                HEEncoder<Scheme::RTF>& encoder,
                                HEEncryptor<Scheme::RTF>& encryptor,
                                HEOperator<Scheme::RTF>& op,
                                Galoiskey<Scheme::RTF>& galois_key,
                                Relinkey<Scheme::RTF>& relin_key,
                                const ExecutionOptions& options
                            )
        : 
        contextbfv_(context), 
        contextckks_(contextckks), 
        encoder_(encoder), 
        encryptor_(encryptor), 
        operator_(op), 
        galois_key_(galois_key),
        relin_key_(relin_key)
    {
        if (!context.context_generated_){
            throw std::invalid_argument("HEContext is not generated!");
        }
        cudaStream_t stream = options.stream_;
        cudaStreamSynchronize(stream);

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


        // hera setup
        gen_FV_moddown_params();

        plain_psi_ = context.plain_psi_;
        messageScaling_ = plain_modulus_.value / message_ratio_;
        
        rc_vec_size_ = static_cast<size_t>(round_ + 1) * n;
        // initialize IC
        icVec.resize(n);
        for (int i = 0; i < n; i++)
        {
            icVec[i] = (i % 16) + 1;
        }
        icPt_ = Plaintext<Scheme::RTF>(context);
        icCt_ = Ciphertext<Scheme::RTF>(context);
        
        encoder_.encode(icPt_, icVec, options);
        encryptor_.encrypt(icCt_, icPt_, options);
        
        icCt_.store_in_device(stream);

        // initialize round constants
        // TODO : this part takes time and thus synchronization issue

        // heongpu::DeviceVector<Data64> d_rc(rc_vec_size_, stream);
        // heongpu::DeviceVector<Modulus64> d_mod(1, stream);

        // HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
        //     d_mod.data(), &plain_modulus_, sizeof(Modulus64), cudaMemcpyHostToDevice, stream));

        // heongpu::RandomNumberGenerator::instance()
        //     .modular_uniform_random_number_generation(
        //         d_rc.data(), d_mod.data(), static_cast<Data64>(n_power),
        //         /*mod_count=*/1, /*repeat_count=*/round_, stream);

        // HEONGPU_CUDA_CHECK(cudaGetLastError());
        // HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        std::vector<uint64_t> flat(rc_vec_size_, 2ULL);
        // HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
        //     flat.data(), d_rc.data(), rc_vec_size_ * sizeof(Data64),
        //     cudaMemcpyDeviceToHost, stream));
        // HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        rcVec.assign(static_cast<size_t>(round_ + 1), std::vector<uint64_t>(n));
        for (int r = 0; r < round_+1; ++r) {
            rcVec[r].assign(flat.begin() + r * n, flat.begin() + (r + 1) * n);
        }

        
        rcPt.reserve(round_+1);
        rckCt.reserve(round_+1);
        // encode rc
        
        for (int r = 0; r < round_+1; ++r) {
            Plaintext<Scheme::RTF> pt(context);
            encoder_.encode(pt, rcVec[r], options);
            pt.store_in_device(options.stream_);
            rcPt.emplace_back(std::move(pt));
        }
        

        const size_t row_len = n >> 1; // BFV batching row length
        const size_t B = 64;           // 16x16 block, NEEDS TO BE CHANGED
        if (row_len % B != 0)
        {
            throw std::invalid_argument(
                "HEHERA::ctor: (n/2) must be a multiple of 16.");
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
        
        static const uint64_t M64[64][64] = {
            {25, 15, 20, 15, 30, 10,  5,  5, 15,  9, 12,  9, 18,  6,  3,  3, 20, 12,
            16, 12, 24,  8,  4,  4, 15,  9, 12,  9, 18,  6,  3,  3, 30, 18, 24, 18,
            36, 12,  6,  6, 10,  6,  8,  6, 12,  4,  2,  2,  5,  3,  4,  3,  6,  2,
            1,  1,  5,  3,  4,  3,  6,  2,  1,  1},
            { 5, 25, 15, 20, 15, 30, 10,  5,  3, 15,  9, 12,  9, 18,  6,  3,  4, 20,
            12, 16, 12, 24,  8,  4,  3, 15,  9, 12,  9, 18,  6,  3,  6, 30, 18, 24,
            18, 36, 12,  6,  2, 10,  6,  8,  6, 12,  4,  2,  1,  5,  3,  4,  3,  6,
            2,  1,  1,  5,  3,  4,  3,  6,  2,  1},
            { 5,  5, 25, 15, 20, 15, 30, 10,  3,  3, 15,  9, 12,  9, 18,  6,  4,  4,
            20, 12, 16, 12, 24,  8,  3,  3, 15,  9, 12,  9, 18,  6,  6,  6, 30, 18,
            24, 18, 36, 12,  2,  2, 10,  6,  8,  6, 12,  4,  1,  1,  5,  3,  4,  3,
            6,  2,  1,  1,  5,  3,  4,  3,  6,  2},
            {10,  5,  5, 25, 15, 20, 15, 30,  6,  3,  3, 15,  9, 12,  9, 18,  8,  4,
            4, 20, 12, 16, 12, 24,  6,  3,  3, 15,  9, 12,  9, 18, 12,  6,  6, 30,
            18, 24, 18, 36,  4,  2,  2, 10,  6,  8,  6, 12,  2,  1,  1,  5,  3,  4,
            3,  6,  2,  1,  1,  5,  3,  4,  3,  6},
            {30, 10,  5,  5, 25, 15, 20, 15, 18,  6,  3,  3, 15,  9, 12,  9, 24,  8,
            4,  4, 20, 12, 16, 12, 18,  6,  3,  3, 15,  9, 12,  9, 36, 12,  6,  6,
            30, 18, 24, 18, 12,  4,  2,  2, 10,  6,  8,  6,  6,  2,  1,  1,  5,  3,
            4,  3,  6,  2,  1,  1,  5,  3,  4,  3},
            {15, 30, 10,  5,  5, 25, 15, 20,  9, 18,  6,  3,  3, 15,  9, 12, 12, 24,
            8,  4,  4, 20, 12, 16,  9, 18,  6,  3,  3, 15,  9, 12, 18, 36, 12,  6,
            6, 30, 18, 24,  6, 12,  4,  2,  2, 10,  6,  8,  3,  6,  2,  1,  1,  5,
            3,  4,  3,  6,  2,  1,  1,  5,  3,  4},
            {20, 15, 30, 10,  5,  5, 25, 15, 12,  9, 18,  6,  3,  3, 15,  9, 16, 12,
            24,  8,  4,  4, 20, 12, 12,  9, 18,  6,  3,  3, 15,  9, 24, 18, 36, 12,
            6,  6, 30, 18,  8,  6, 12,  4,  2,  2, 10,  6,  4,  3,  6,  2,  1,  1,
            5,  3,  4,  3,  6,  2,  1,  1,  5,  3},
            {15, 20, 15, 30, 10,  5,  5, 25,  9, 12,  9, 18,  6,  3,  3, 15, 12, 16,
            12, 24,  8,  4,  4, 20,  9, 12,  9, 18,  6,  3,  3, 15, 18, 24, 18, 36,
            12,  6,  6, 30,  6,  8,  6, 12,  4,  2,  2, 10,  3,  4,  3,  6,  2,  1,
            1,  5,  3,  4,  3,  6,  2,  1,  1,  5},
            { 5,  3,  4,  3,  6,  2,  1,  1, 25, 15, 20, 15, 30, 10,  5,  5, 15,  9,
            12,  9, 18,  6,  3,  3, 20, 12, 16, 12, 24,  8,  4,  4, 15,  9, 12,  9,
            18,  6,  3,  3, 30, 18, 24, 18, 36, 12,  6,  6, 10,  6,  8,  6, 12,  4,
            2,  2,  5,  3,  4,  3,  6,  2,  1,  1},
            { 1,  5,  3,  4,  3,  6,  2,  1,  5, 25, 15, 20, 15, 30, 10,  5,  3, 15,
            9, 12,  9, 18,  6,  3,  4, 20, 12, 16, 12, 24,  8,  4,  3, 15,  9, 12,
            9, 18,  6,  3,  6, 30, 18, 24, 18, 36, 12,  6,  2, 10,  6,  8,  6, 12,
            4,  2,  1,  5,  3,  4,  3,  6,  2,  1},
            { 1,  1,  5,  3,  4,  3,  6,  2,  5,  5, 25, 15, 20, 15, 30, 10,  3,  3,
            15,  9, 12,  9, 18,  6,  4,  4, 20, 12, 16, 12, 24,  8,  3,  3, 15,  9,
            12,  9, 18,  6,  6,  6, 30, 18, 24, 18, 36, 12,  2,  2, 10,  6,  8,  6,
            12,  4,  1,  1,  5,  3,  4,  3,  6,  2},
            { 2,  1,  1,  5,  3,  4,  3,  6, 10,  5,  5, 25, 15, 20, 15, 30,  6,  3,
            3, 15,  9, 12,  9, 18,  8,  4,  4, 20, 12, 16, 12, 24,  6,  3,  3, 15,
            9, 12,  9, 18, 12,  6,  6, 30, 18, 24, 18, 36,  4,  2,  2, 10,  6,  8,
            6, 12,  2,  1,  1,  5,  3,  4,  3,  6},
            { 6,  2,  1,  1,  5,  3,  4,  3, 30, 10,  5,  5, 25, 15, 20, 15, 18,  6,
            3,  3, 15,  9, 12,  9, 24,  8,  4,  4, 20, 12, 16, 12, 18,  6,  3,  3,
            15,  9, 12,  9, 36, 12,  6,  6, 30, 18, 24, 18, 12,  4,  2,  2, 10,  6,
            8,  6,  6,  2,  1,  1,  5,  3,  4,  3},
            { 3,  6,  2,  1,  1,  5,  3,  4, 15, 30, 10,  5,  5, 25, 15, 20,  9, 18,
            6,  3,  3, 15,  9, 12, 12, 24,  8,  4,  4, 20, 12, 16,  9, 18,  6,  3,
            3, 15,  9, 12, 18, 36, 12,  6,  6, 30, 18, 24,  6, 12,  4,  2,  2, 10,
            6,  8,  3,  6,  2,  1,  1,  5,  3,  4},
            { 4,  3,  6,  2,  1,  1,  5,  3, 20, 15, 30, 10,  5,  5, 25, 15, 12,  9,
            18,  6,  3,  3, 15,  9, 16, 12, 24,  8,  4,  4, 20, 12, 12,  9, 18,  6,
            3,  3, 15,  9, 24, 18, 36, 12,  6,  6, 30, 18,  8,  6, 12,  4,  2,  2,
            10,  6,  4,  3,  6,  2,  1,  1,  5,  3},
            { 3,  4,  3,  6,  2,  1,  1,  5, 15, 20, 15, 30, 10,  5,  5, 25,  9, 12,
            9, 18,  6,  3,  3, 15, 12, 16, 12, 24,  8,  4,  4, 20,  9, 12,  9, 18,
            6,  3,  3, 15, 18, 24, 18, 36, 12,  6,  6, 30,  6,  8,  6, 12,  4,  2,
            2, 10,  3,  4,  3,  6,  2,  1,  1,  5},
            { 5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1, 25, 15,
            20, 15, 30, 10,  5,  5, 15,  9, 12,  9, 18,  6,  3,  3, 20, 12, 16, 12,
            24,  8,  4,  4, 15,  9, 12,  9, 18,  6,  3,  3, 30, 18, 24, 18, 36, 12,
            6,  6, 10,  6,  8,  6, 12,  4,  2,  2},
            { 1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  5, 25,
            15, 20, 15, 30, 10,  5,  3, 15,  9, 12,  9, 18,  6,  3,  4, 20, 12, 16,
            12, 24,  8,  4,  3, 15,  9, 12,  9, 18,  6,  3,  6, 30, 18, 24, 18, 36,
            12,  6,  2, 10,  6,  8,  6, 12,  4,  2},
            { 1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  5,  5,
            25, 15, 20, 15, 30, 10,  3,  3, 15,  9, 12,  9, 18,  6,  4,  4, 20, 12,
            16, 12, 24,  8,  3,  3, 15,  9, 12,  9, 18,  6,  6,  6, 30, 18, 24, 18,
            36, 12,  2,  2, 10,  6,  8,  6, 12,  4},
            { 2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6, 10,  5,
            5, 25, 15, 20, 15, 30,  6,  3,  3, 15,  9, 12,  9, 18,  8,  4,  4, 20,
            12, 16, 12, 24,  6,  3,  3, 15,  9, 12,  9, 18, 12,  6,  6, 30, 18, 24,
            18, 36,  4,  2,  2, 10,  6,  8,  6, 12},
            { 6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3, 30, 10,
            5,  5, 25, 15, 20, 15, 18,  6,  3,  3, 15,  9, 12,  9, 24,  8,  4,  4,
            20, 12, 16, 12, 18,  6,  3,  3, 15,  9, 12,  9, 36, 12,  6,  6, 30, 18,
            24, 18, 12,  4,  2,  2, 10,  6,  8,  6},
            { 3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4, 15, 30,
            10,  5,  5, 25, 15, 20,  9, 18,  6,  3,  3, 15,  9, 12, 12, 24,  8,  4,
            4, 20, 12, 16,  9, 18,  6,  3,  3, 15,  9, 12, 18, 36, 12,  6,  6, 30,
            18, 24,  6, 12,  4,  2,  2, 10,  6,  8},
            { 4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3, 20, 15,
            30, 10,  5,  5, 25, 15, 12,  9, 18,  6,  3,  3, 15,  9, 16, 12, 24,  8,
            4,  4, 20, 12, 12,  9, 18,  6,  3,  3, 15,  9, 24, 18, 36, 12,  6,  6,
            30, 18,  8,  6, 12,  4,  2,  2, 10,  6},
            { 3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5, 15, 20,
            15, 30, 10,  5,  5, 25,  9, 12,  9, 18,  6,  3,  3, 15, 12, 16, 12, 24,
            8,  4,  4, 20,  9, 12,  9, 18,  6,  3,  3, 15, 18, 24, 18, 36, 12,  6,
            6, 30,  6,  8,  6, 12,  4,  2,  2, 10},
            {10,  6,  8,  6, 12,  4,  2,  2,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,
            4,  3,  6,  2,  1,  1, 25, 15, 20, 15, 30, 10,  5,  5, 15,  9, 12,  9,
            18,  6,  3,  3, 20, 12, 16, 12, 24,  8,  4,  4, 15,  9, 12,  9, 18,  6,
            3,  3, 30, 18, 24, 18, 36, 12,  6,  6},
            { 2, 10,  6,  8,  6, 12,  4,  2,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,
            3,  4,  3,  6,  2,  1,  5, 25, 15, 20, 15, 30, 10,  5,  3, 15,  9, 12,
            9, 18,  6,  3,  4, 20, 12, 16, 12, 24,  8,  4,  3, 15,  9, 12,  9, 18,
            6,  3,  6, 30, 18, 24, 18, 36, 12,  6},
            { 2,  2, 10,  6,  8,  6, 12,  4,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,
            5,  3,  4,  3,  6,  2,  5,  5, 25, 15, 20, 15, 30, 10,  3,  3, 15,  9,
            12,  9, 18,  6,  4,  4, 20, 12, 16, 12, 24,  8,  3,  3, 15,  9, 12,  9,
            18,  6,  6,  6, 30, 18, 24, 18, 36, 12},
            { 4,  2,  2, 10,  6,  8,  6, 12,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,
            1,  5,  3,  4,  3,  6, 10,  5,  5, 25, 15, 20, 15, 30,  6,  3,  3, 15,
            9, 12,  9, 18,  8,  4,  4, 20, 12, 16, 12, 24,  6,  3,  3, 15,  9, 12,
            9, 18, 12,  6,  6, 30, 18, 24, 18, 36},
            {12,  4,  2,  2, 10,  6,  8,  6,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,
            1,  1,  5,  3,  4,  3, 30, 10,  5,  5, 25, 15, 20, 15, 18,  6,  3,  3,
            15,  9, 12,  9, 24,  8,  4,  4, 20, 12, 16, 12, 18,  6,  3,  3, 15,  9,
            12,  9, 36, 12,  6,  6, 30, 18, 24, 18},
            { 6, 12,  4,  2,  2, 10,  6,  8,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,
            2,  1,  1,  5,  3,  4, 15, 30, 10,  5,  5, 25, 15, 20,  9, 18,  6,  3,
            3, 15,  9, 12, 12, 24,  8,  4,  4, 20, 12, 16,  9, 18,  6,  3,  3, 15,
            9, 12, 18, 36, 12,  6,  6, 30, 18, 24},
            { 8,  6, 12,  4,  2,  2, 10,  6,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,
            6,  2,  1,  1,  5,  3, 20, 15, 30, 10,  5,  5, 25, 15, 12,  9, 18,  6,
            3,  3, 15,  9, 16, 12, 24,  8,  4,  4, 20, 12, 12,  9, 18,  6,  3,  3,
            15,  9, 24, 18, 36, 12,  6,  6, 30, 18},
            { 6,  8,  6, 12,  4,  2,  2, 10,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,
            3,  6,  2,  1,  1,  5, 15, 20, 15, 30, 10,  5,  5, 25,  9, 12,  9, 18,
            6,  3,  3, 15, 12, 16, 12, 24,  8,  4,  4, 20,  9, 12,  9, 18,  6,  3,
            3, 15, 18, 24, 18, 36, 12,  6,  6, 30},
            {30, 18, 24, 18, 36, 12,  6,  6, 10,  6,  8,  6, 12,  4,  2,  2,  5,  3,
            4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1, 25, 15, 20, 15,
            30, 10,  5,  5, 15,  9, 12,  9, 18,  6,  3,  3, 20, 12, 16, 12, 24,  8,
            4,  4, 15,  9, 12,  9, 18,  6,  3,  3},
            { 6, 30, 18, 24, 18, 36, 12,  6,  2, 10,  6,  8,  6, 12,  4,  2,  1,  5,
            3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  5, 25, 15, 20,
            15, 30, 10,  5,  3, 15,  9, 12,  9, 18,  6,  3,  4, 20, 12, 16, 12, 24,
            8,  4,  3, 15,  9, 12,  9, 18,  6,  3},
            { 6,  6, 30, 18, 24, 18, 36, 12,  2,  2, 10,  6,  8,  6, 12,  4,  1,  1,
            5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  5,  5, 25, 15,
            20, 15, 30, 10,  3,  3, 15,  9, 12,  9, 18,  6,  4,  4, 20, 12, 16, 12,
            24,  8,  3,  3, 15,  9, 12,  9, 18,  6},
            {12,  6,  6, 30, 18, 24, 18, 36,  4,  2,  2, 10,  6,  8,  6, 12,  2,  1,
            1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6, 10,  5,  5, 25,
            15, 20, 15, 30,  6,  3,  3, 15,  9, 12,  9, 18,  8,  4,  4, 20, 12, 16,
            12, 24,  6,  3,  3, 15,  9, 12,  9, 18},
            {36, 12,  6,  6, 30, 18, 24, 18, 12,  4,  2,  2, 10,  6,  8,  6,  6,  2,
            1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3, 30, 10,  5,  5,
            25, 15, 20, 15, 18,  6,  3,  3, 15,  9, 12,  9, 24,  8,  4,  4, 20, 12,
            16, 12, 18,  6,  3,  3, 15,  9, 12,  9},
            {18, 36, 12,  6,  6, 30, 18, 24,  6, 12,  4,  2,  2, 10,  6,  8,  3,  6,
            2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4, 15, 30, 10,  5,
            5, 25, 15, 20,  9, 18,  6,  3,  3, 15,  9, 12, 12, 24,  8,  4,  4, 20,
            12, 16,  9, 18,  6,  3,  3, 15,  9, 12},
            {24, 18, 36, 12,  6,  6, 30, 18,  8,  6, 12,  4,  2,  2, 10,  6,  4,  3,
            6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3, 20, 15, 30, 10,
            5,  5, 25, 15, 12,  9, 18,  6,  3,  3, 15,  9, 16, 12, 24,  8,  4,  4,
            20, 12, 12,  9, 18,  6,  3,  3, 15,  9},
            {18, 24, 18, 36, 12,  6,  6, 30,  6,  8,  6, 12,  4,  2,  2, 10,  3,  4,
            3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5, 15, 20, 15, 30,
            10,  5,  5, 25,  9, 12,  9, 18,  6,  3,  3, 15, 12, 16, 12, 24,  8,  4,
            4, 20,  9, 12,  9, 18,  6,  3,  3, 15},
            {15,  9, 12,  9, 18,  6,  3,  3, 30, 18, 24, 18, 36, 12,  6,  6, 10,  6,
            8,  6, 12,  4,  2,  2,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,
            6,  2,  1,  1, 25, 15, 20, 15, 30, 10,  5,  5, 15,  9, 12,  9, 18,  6,
            3,  3, 20, 12, 16, 12, 24,  8,  4,  4},
            { 3, 15,  9, 12,  9, 18,  6,  3,  6, 30, 18, 24, 18, 36, 12,  6,  2, 10,
            6,  8,  6, 12,  4,  2,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,
            3,  6,  2,  1,  5, 25, 15, 20, 15, 30, 10,  5,  3, 15,  9, 12,  9, 18,
            6,  3,  4, 20, 12, 16, 12, 24,  8,  4},
            { 3,  3, 15,  9, 12,  9, 18,  6,  6,  6, 30, 18, 24, 18, 36, 12,  2,  2,
            10,  6,  8,  6, 12,  4,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,
            4,  3,  6,  2,  5,  5, 25, 15, 20, 15, 30, 10,  3,  3, 15,  9, 12,  9,
            18,  6,  4,  4, 20, 12, 16, 12, 24,  8},
            { 6,  3,  3, 15,  9, 12,  9, 18, 12,  6,  6, 30, 18, 24, 18, 36,  4,  2,
            2, 10,  6,  8,  6, 12,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,
            3,  4,  3,  6, 10,  5,  5, 25, 15, 20, 15, 30,  6,  3,  3, 15,  9, 12,
            9, 18,  8,  4,  4, 20, 12, 16, 12, 24},
            {18,  6,  3,  3, 15,  9, 12,  9, 36, 12,  6,  6, 30, 18, 24, 18, 12,  4,
            2,  2, 10,  6,  8,  6,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,
            5,  3,  4,  3, 30, 10,  5,  5, 25, 15, 20, 15, 18,  6,  3,  3, 15,  9,
            12,  9, 24,  8,  4,  4, 20, 12, 16, 12},
            { 9, 18,  6,  3,  3, 15,  9, 12, 18, 36, 12,  6,  6, 30, 18, 24,  6, 12,
            4,  2,  2, 10,  6,  8,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,
            1,  5,  3,  4, 15, 30, 10,  5,  5, 25, 15, 20,  9, 18,  6,  3,  3, 15,
            9, 12, 12, 24,  8,  4,  4, 20, 12, 16},
            {12,  9, 18,  6,  3,  3, 15,  9, 24, 18, 36, 12,  6,  6, 30, 18,  8,  6,
            12,  4,  2,  2, 10,  6,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,
            1,  1,  5,  3, 20, 15, 30, 10,  5,  5, 25, 15, 12,  9, 18,  6,  3,  3,
            15,  9, 16, 12, 24,  8,  4,  4, 20, 12},
            { 9, 12,  9, 18,  6,  3,  3, 15, 18, 24, 18, 36, 12,  6,  6, 30,  6,  8,
            6, 12,  4,  2,  2, 10,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,
            2,  1,  1,  5, 15, 20, 15, 30, 10,  5,  5, 25,  9, 12,  9, 18,  6,  3,
            3, 15, 12, 16, 12, 24,  8,  4,  4, 20},
            {20, 12, 16, 12, 24,  8,  4,  4, 15,  9, 12,  9, 18,  6,  3,  3, 30, 18,
            24, 18, 36, 12,  6,  6, 10,  6,  8,  6, 12,  4,  2,  2,  5,  3,  4,  3,
            6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1, 25, 15, 20, 15, 30, 10,
            5,  5, 15,  9, 12,  9, 18,  6,  3,  3},
            { 4, 20, 12, 16, 12, 24,  8,  4,  3, 15,  9, 12,  9, 18,  6,  3,  6, 30,
            18, 24, 18, 36, 12,  6,  2, 10,  6,  8,  6, 12,  4,  2,  1,  5,  3,  4,
            3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  5, 25, 15, 20, 15, 30,
            10,  5,  3, 15,  9, 12,  9, 18,  6,  3},
            { 4,  4, 20, 12, 16, 12, 24,  8,  3,  3, 15,  9, 12,  9, 18,  6,  6,  6,
            30, 18, 24, 18, 36, 12,  2,  2, 10,  6,  8,  6, 12,  4,  1,  1,  5,  3,
            4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  5,  5, 25, 15, 20, 15,
            30, 10,  3,  3, 15,  9, 12,  9, 18,  6},
            { 8,  4,  4, 20, 12, 16, 12, 24,  6,  3,  3, 15,  9, 12,  9, 18, 12,  6,
            6, 30, 18, 24, 18, 36,  4,  2,  2, 10,  6,  8,  6, 12,  2,  1,  1,  5,
            3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6, 10,  5,  5, 25, 15, 20,
            15, 30,  6,  3,  3, 15,  9, 12,  9, 18},
            {24,  8,  4,  4, 20, 12, 16, 12, 18,  6,  3,  3, 15,  9, 12,  9, 36, 12,
            6,  6, 30, 18, 24, 18, 12,  4,  2,  2, 10,  6,  8,  6,  6,  2,  1,  1,
            5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3, 30, 10,  5,  5, 25, 15,
            20, 15, 18,  6,  3,  3, 15,  9, 12,  9},
            {12, 24,  8,  4,  4, 20, 12, 16,  9, 18,  6,  3,  3, 15,  9, 12, 18, 36,
            12,  6,  6, 30, 18, 24,  6, 12,  4,  2,  2, 10,  6,  8,  3,  6,  2,  1,
            1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4, 15, 30, 10,  5,  5, 25,
            15, 20,  9, 18,  6,  3,  3, 15,  9, 12},
            {16, 12, 24,  8,  4,  4, 20, 12, 12,  9, 18,  6,  3,  3, 15,  9, 24, 18,
            36, 12,  6,  6, 30, 18,  8,  6, 12,  4,  2,  2, 10,  6,  4,  3,  6,  2,
            1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3, 20, 15, 30, 10,  5,  5,
            25, 15, 12,  9, 18,  6,  3,  3, 15,  9},
            {12, 16, 12, 24,  8,  4,  4, 20,  9, 12,  9, 18,  6,  3,  3, 15, 18, 24,
            18, 36, 12,  6,  6, 30,  6,  8,  6, 12,  4,  2,  2, 10,  3,  4,  3,  6,
            2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5, 15, 20, 15, 30, 10,  5,
            5, 25,  9, 12,  9, 18,  6,  3,  3, 15},
            {15,  9, 12,  9, 18,  6,  3,  3, 20, 12, 16, 12, 24,  8,  4,  4, 15,  9,
            12,  9, 18,  6,  3,  3, 30, 18, 24, 18, 36, 12,  6,  6, 10,  6,  8,  6,
            12,  4,  2,  2,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,
            1,  1, 25, 15, 20, 15, 30, 10,  5,  5},
            { 3, 15,  9, 12,  9, 18,  6,  3,  4, 20, 12, 16, 12, 24,  8,  4,  3, 15,
            9, 12,  9, 18,  6,  3,  6, 30, 18, 24, 18, 36, 12,  6,  2, 10,  6,  8,
            6, 12,  4,  2,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,
            2,  1,  5, 25, 15, 20, 15, 30, 10,  5},
            { 3,  3, 15,  9, 12,  9, 18,  6,  4,  4, 20, 12, 16, 12, 24,  8,  3,  3,
            15,  9, 12,  9, 18,  6,  6,  6, 30, 18, 24, 18, 36, 12,  2,  2, 10,  6,
            8,  6, 12,  4,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,
            6,  2,  5,  5, 25, 15, 20, 15, 30, 10},
            { 6,  3,  3, 15,  9, 12,  9, 18,  8,  4,  4, 20, 12, 16, 12, 24,  6,  3,
            3, 15,  9, 12,  9, 18, 12,  6,  6, 30, 18, 24, 18, 36,  4,  2,  2, 10,
            6,  8,  6, 12,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,
            3,  6, 10,  5,  5, 25, 15, 20, 15, 30},
            {18,  6,  3,  3, 15,  9, 12,  9, 24,  8,  4,  4, 20, 12, 16, 12, 18,  6,
            3,  3, 15,  9, 12,  9, 36, 12,  6,  6, 30, 18, 24, 18, 12,  4,  2,  2,
            10,  6,  8,  6,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,  3,
            4,  3, 30, 10,  5,  5, 25, 15, 20, 15},
            { 9, 18,  6,  3,  3, 15,  9, 12, 12, 24,  8,  4,  4, 20, 12, 16,  9, 18,
            6,  3,  3, 15,  9, 12, 18, 36, 12,  6,  6, 30, 18, 24,  6, 12,  4,  2,
            2, 10,  6,  8,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,  5,
            3,  4, 15, 30, 10,  5,  5, 25, 15, 20},
            {12,  9, 18,  6,  3,  3, 15,  9, 16, 12, 24,  8,  4,  4, 20, 12, 12,  9,
            18,  6,  3,  3, 15,  9, 24, 18, 36, 12,  6,  6, 30, 18,  8,  6, 12,  4,
            2,  2, 10,  6,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,  1,
            5,  3, 20, 15, 30, 10,  5,  5, 25, 15},
            { 9, 12,  9, 18,  6,  3,  3, 15, 12, 16, 12, 24,  8,  4,  4, 20,  9, 12,
            9, 18,  6,  3,  3, 15, 18, 24, 18, 36, 12,  6,  6, 30,  6,  8,  6, 12,
            4,  2,  2, 10,  3,  4,  3,  6,  2,  1,  1,  5,  3,  4,  3,  6,  2,  1,
            1,  5, 15, 20, 15, 30, 10,  5,  5, 25}
        };

        linear_matrix_shifts_.resize(1);
        linear_matrix_shifts_[0].reserve(64+64-1);
        linear_matrix_shifts_[0].push_back(0);
        
        // Lower diagonals (< 0), Upper diagonals (> 0)
        for (int s = 1; s < (int)B; ++s) linear_matrix_shifts_[0].push_back(-s);
        for (int s = 1; s < (int)B; ++s) linear_matrix_shifts_[0].push_back(s);

        const size_t D = linear_matrix_shifts_[0].size();
        std::cout << "Diagonals count: " << D << std::endl;
        heongpu::DeviceVector<Data64> blob(n * D, stream);
        heongpu::Plaintext<heongpu::Scheme::RTF> pt_diag(context);

        for (size_t k = 0; k < D; ++k)
        {
            const int rot = linear_matrix_shifts_[0][k];
            std::vector<uint64_t> hdiag_vec(n, 0ULL);

            for (size_t row = 0; row < 2; ++row)
            {
                const size_t row_base = row * row_len;
                for (size_t blk = 0; blk < blocks_per_row; ++blk)
                {
                    const size_t base = row_base + blk * B;
                    if (rot == 0) { // Main diagonal
                        for (int r = 0; r < (int)B; ++r) hdiag_vec[base + r] = M64[r][r];
                    } else if (rot > 0) { // Upper diagonals
                        const int s = rot;
                        for (int r = 0; r < (int)B - s; ++r) hdiag_vec[base + r] = M64[r][r + s];
                    } else { // Lower diagonals
                        const int s = -rot;
                        for (int r = s; r < (int)B; ++r) hdiag_vec[base + r] = M64[r][r - s];
                    }
                }
            }
            
            encoder.encode(pt_diag, hdiag_vec, options);
            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                blob.data() + k * n, pt_diag.data(), n * sizeof(Data64),
                cudaMemcpyDeviceToDevice, stream));
        }
        
        // Store the completed blob in the member variable
        linear_matrix_diagonals_.emplace_back(std::move(blob));
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cout << "HERA initialized." << std::endl;

    }
    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::precompute(
        heongpu::Ciphertext<heongpu::Scheme::RTF>& key,
        const heongpu::ExecutionOptions& options)
    {
        rckCt.clear();
        rckCt.reserve(round_+1);  // you only fill 0..round_-1
        auto opt = ExecutionOptions().set_stream(options.stream_)
                                    .set_storage_type(storage_type::DEVICE)
                                    .set_initial_location(true);

        for (int r = 0; r < round_+1; ++r) {
            Ciphertext<Scheme::RTF> ct(contextbfv_);           // own buffers
            ct.store_in_device(options.stream_);
            operator_.multiply_plain(key, rcPt[r], ct, opt);
            rckCt.emplace_back(std::move(ct));
        }
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
    }

    __host__ Ciphertext<Scheme::RTF> heongpu::HEHERA<heongpu::Scheme::RTF>::gen_stream_key(
        // heongpu::DeviceVector<Data64>& nonce, 
        Ciphertext<Scheme::RTF>& key,
        const ExecutionOptions& options)
    {
        Ciphertext<Scheme::RTF> result(contextbfv_);
        precompute(key, options);

        // ark
        add_round_key(icCt_, rckCt[0], result, options);
        // std::cout << "here3" << std::endl;
        
        // round_function
        for (int r = 1; r < round_; ++r)
        {
            linear(result, result, options);
            cube(result, result, options);
            add_round_key(result, rckCt[r], result, options);

        }

        linear(result, result, options);
        cube(result, result, options);
        linear(result, result, options);
        add_round_key(result, rckCt[round_], result, options);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
        return result;


    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::add_round_key(
        Ciphertext<Scheme::RTF>& input, Ciphertext<Scheme::RTF>& round_key,
        Ciphertext<Scheme::RTF>& output, const ExecutionOptions& options)
    {
        cudaStream_t stream = options.stream_;
        if (stream == cudaStreamDefault) stream = input.stream();
        operator_.add(input, round_key, output, options);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

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
        Ciphertext<Scheme::RTF>& input, 
        Ciphertext<Scheme::RTF>& output,
        const ExecutionOptions& options)
    {
        cudaStream_t stream = options.stream_;
        if (stream == cudaStreamDefault) stream = input.stream();
            
        operator_.multiply_inplace(input, input, options);
        operator_.relinearize_inplace(input, relin_key_, options);
        operator_.multiply(input, input, output, options);
        operator_.relinearize_inplace(output, relin_key_, options);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::linear(
        heongpu::Ciphertext<heongpu::Scheme::RTF>& input,
        heongpu::Ciphertext<heongpu::Scheme::RTF>& output,
        const ExecutionOptions& options)
    {
        cudaStream_t stream = options.stream_;
        if (stream == cudaStreamDefault) stream = input.stream();

        ExecutionOptions opt = ExecutionOptions()
                                   .set_stream(stream)
                                   .set_storage_type(storage_type::DEVICE)
                                   .set_initial_location(true);

        if (input.relinearization_required_)
        {
            throw std::invalid_argument(
                "linear(): input has non-linear part; relinearize first.");
        }

        heongpu::Ciphertext<heongpu::Scheme::RTF> x = input;
        if (x.in_ntt_domain_)
        {
            operator_.transform_from_ntt_inplace(x, opt);
        }

        auto y = operator_.multiply_matrix(x, 
                                          this->linear_matrix_diagonals_, 
                                          this->linear_matrix_shifts_, 
                                          galois_key_hera_, 
                                          opt);
        output = y;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    }

    static inline void rotate_subrange_right_inplace(std::vector<uint64_t>& v,
                                                    int start, int len, int k)
    {
        if (len <= 0) return;
        k %= len; if (k < 0) k += len;
        if (k == 0) return;
        auto first = v.begin() + start;
        auto last  = first + len;
        std::rotate(first, last - k, last);
    }

    static inline void pre_rotate_diagonals(std::vector<uint64_t>& v, int H, int k)
    {
        const int N = static_cast<int>(v.size());
        if (N == 0 || H <= 0 || 2 * H != N || k == 0) return;
        
        const int num_swaps = k / H;
        const int small_rotation = k % H;

        if ((num_swaps % 2) != 0) {
            auto first_half_begin = v.begin();
            auto second_half_begin = v.begin() + H;
            std::rotate(first_half_begin, second_half_begin, v.end());
        }

        if (small_rotation != 0) {
            rotate_subrange_right_inplace(v, 0, H, small_rotation);
            rotate_subrange_right_inplace(v, H, H, small_rotation);
        }
    }
    static inline int row_aligned_col_idx(int i, int s, int H)
    {
        const int row_base = (i < H) ? 0 : H;
        const int in_row   = i - row_base;

        if (s < H) {
            return row_base + ((in_row + s) % H);
        } else {
            const int r = s - H;
            const int other_base = (row_base == 0) ? H : 0;
            return other_base + ((in_row + r) % H);
        }
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::gen_FV_S2C_Matrix(
        const ExecutionOptions& options)
    {
        if(is_S2C_initialized_) return; 
        
        cudaStream_t stream = options.stream_;
        const int N    = (int)contextbfv_.n;
        const int H    = N >> 1;
        const int twoN = 2 * N;

        const uint64_t t   = plain_modulus_.value;
        const uint64_t psi = plain_psi_;

        psi_pow_.resize(twoN);
        psi_pow_[0] = 1ULL;
        for (int k = 1; k < twoN; ++k)
            psi_pow_[k] = mul_mod_u128(psi_pow_[k - 1], psi, t);

        const int g2 = (int)std::ceil(std::sqrt((double)H));

        s2c_matrix_diagonals_.assign(2, {});     
        s2c_matrix_shifts_.assign(2, {});

        s2c_matrix_shifts_[0].resize(H);
        std::iota(s2c_matrix_shifts_[0].begin(), s2c_matrix_shifts_[0].end(), 0);

        s2c_matrix_shifts_[1].resize(N - H);
        std::iota(s2c_matrix_shifts_[1].begin(), s2c_matrix_shifts_[1].end(), H);

        const size_t per_diag_len_plain = (size_t)N;
        heongpu::DeviceVector<Data64> blob0(per_diag_len_plain * (size_t)H, stream);
        heongpu::DeviceVector<Data64> blob1(per_diag_len_plain * (size_t)(N - H), stream);

        heongpu::Plaintext<heongpu::Scheme::RTF> pt_diag(contextbfv_);
        std::vector<uint64_t> diag_slots(N);

        const uint64_t generator = generator_;
        std::vector<uint64_t> gen_pow(N);

        uint64_t pos = 1ULL; 

        for (int i = 0; i < N / 2; ++i) {
            gen_pow[i] = pos;
            pos = (pos * generator) % twoN;
        }

        for (int i = N / 2; i < N; ++i) {
            gen_pow[i] = twoN - pos;
            pos = (pos * generator) % twoN;
        }
        for (int s = 0; s < N; ++s)
        {
            for (int i = 0; i < N; ++i) {
                const int j_col = row_aligned_col_idx(i, s, H); 

                const uint64_t e = (static_cast<uint64_t>(j_col) * gen_pow[i])
                                % static_cast<uint64_t>(twoN);

                diag_slots[i] = psi_pow_[static_cast<size_t>(e)];
            }
            int r = (s < H) ? s : (s - H);
            int j = r % g2;
            int i_big = (r - j) / g2;
            int giant = i_big * g2;

            pre_rotate_diagonals(diag_slots, H, giant);

            encoder_.encode(pt_diag, diag_slots, options);

            // 2) NTT(RNS) 로 변환 → 내부 버퍼가 N * Q_size_ 로 바뀜
            // operator_.transform_to_ntt_inplace(pt_diag, options);

            // 3) ⬇⬇⬇ 변경: N이 아니라 N * Q_size_ 바이트를 복사
            const Data64* src_plain = pt_diag.data();                // length = N (plain)
            if (s < H) {
                Data64* dst = blob0.data() + ((size_t)s) * per_diag_len_plain;
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    dst, src_plain, per_diag_len_plain * sizeof(Data64),
                    cudaMemcpyDeviceToDevice, stream));
            } else {
                Data64* dst = blob1.data() + ((size_t)(s - H)) * per_diag_len_plain;
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    dst, src_plain, per_diag_len_plain * sizeof(Data64),
                    cudaMemcpyDeviceToDevice, stream));
            }
        }

        s2c_matrix_diagonals_[0].clear();
        s2c_matrix_diagonals_[1].clear();
        s2c_matrix_diagonals_[0].push_back(std::move(blob0));
        s2c_matrix_diagonals_[1].push_back(std::move(blob1));

        is_S2C_initialized_ = true;

        if (stream != cudaStreamDefault)
            cudaStreamSynchronize(stream);
    }
    
    __host__ heongpu::Ciphertext<heongpu::Scheme::RTF> HEHERA<heongpu::Scheme::RTF>::S2C_FV(
        heongpu::Ciphertext<heongpu::Scheme::RTF>& ct_in,
        const ExecutionOptions& options)
    {
        if (!is_S2C_initialized_) {
            throw std::invalid_argument("S2C matrix not initialized; call gen_FV_S2C_Matrix() first.");
        }
        if (ct_in.in_ntt_domain_) {
            operator_.transform_from_ntt_inplace(ct_in, options);
        }

        auto& dft_matrix_diagonals = s2c_matrix_diagonals_;
        auto& dft_matrix_shifts    = s2c_matrix_shifts_;

        return operator_.multiply_matrix_bsgs(
            ct_in,
            dft_matrix_diagonals,
            dft_matrix_shifts,
            galois_key_,
            galois_key_bs_,
            options
        );
    }

    __host__ heongpu::Plaintext<heongpu::Scheme::RTF> HEHERA<heongpu::Scheme::RTF>::vec2poly(
        const std::vector<uint64_t>& input,
        const ExecutionOptions& options
    ){
        cudaStream_t stream = options.stream_;
        if (stream == cudaStreamDefault) stream = 0;

        const int      N = static_cast<int>(contextbfv_.n);
        const uint64_t t = plain_modulus_.value;

        std::vector<uint64_t> coeffs((size_t)N, 0ULL);
        const size_t use_len = std::min(input.size(), (size_t)N);
        for (size_t i = 0; i < use_len; ++i) coeffs[i] = t ? (input[i] % t) : input[i];

        heongpu::DeviceVector<Data64> dev_coeffs;
        dev_coeffs.resize(N, stream);

        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
            dev_coeffs.data(), coeffs.data(), (size_t)N * sizeof(uint64_t),
            cudaMemcpyHostToDevice, stream));
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        heongpu::Plaintext<heongpu::Scheme::RTF> pt(scheme_, dev_coeffs, /*is_ntt=*/false);
        return pt;
    }
    
    __host__ heongpu::Ciphertext<heongpu::Scheme::RTF> HEHERA<heongpu::Scheme::RTF>::eval_decrypt(
        Ciphertext<Scheme::RTF>& keyCt, 
        Plaintext<Scheme::RTF>& SKEpt,
        const ExecutionOptions& options
    ){
        Ciphertext<Scheme::RTF> keyCT_neg(contextbfv_), result(contextbfv_);
        operator_.negate(keyCt, keyCT_neg, options);
        operator_.add_plain(keyCT_neg, SKEpt, result, options); // m + z - z

        return result;
    }
    
    __host__ heongpu::Ciphertext<heongpu::Scheme::CKKS>
    HEHERA<heongpu::Scheme::RTF>::transcipher_bfv2ckks(
        Ciphertext<Scheme::RTF>& ct_bfv,
        const ExecutionOptions& options
    )
    {
        cudaStream_t stream = options.stream_;
        moddown_FV_inplace(ct_bfv, options);

        if (contextbfv_.get_poly_modulus_degree() != contextckks_.get_poly_modulus_degree()) {
            throw std::invalid_argument("Poly modulus degree mismatch between BFV and CKKS contexts.");
        }

        if (ct_bfv.coeff_modulus_count() > contextckks_.get_ciphertext_modulus_count()) {
            throw std::invalid_argument("BFV ciphertext has more moduli than the CKKS context supports.");
        }

        Ciphertext<Scheme::CKKS> ct_ckks(contextckks_, options);
        ct_ckks.coeff_modulus_count_ = ct_bfv.coeff_modulus_count();
        ct_ckks.depth_ = contextckks_.get_ciphertext_modulus_count() - ct_bfv.coeff_modulus_count();
        std::cout << "transciphering ckks depth :" << ct_ckks.depth_ << std::endl;

        if (!ct_bfv.is_on_device()) {
            ct_bfv.store_in_device(stream);
        }

        const size_t bfv_words = static_cast<size_t>(ct_bfv.ring_size())
                            * static_cast<size_t>(ct_bfv.coeff_modulus_count())
                            * static_cast<size_t>(ct_bfv.size());
        if (ct_ckks.device_locations_.size() != bfv_words) {
            ct_ckks.device_locations_.resize(bfv_words, stream);
        }

        const size_t bytes = bfv_words * sizeof(Data64);
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
            ct_ckks.device_locations_.data(),
            ct_bfv.device_locations_.data(),
            bytes,
            cudaMemcpyDeviceToDevice,
            stream
        ));
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        // // ===== inline print (after BFV→CKKS copy) =====
        // {
        //     const int N          = ct_ckks.ring_size();
        //     const int poly_count = ct_ckks.size();                 // usually 2
        //     const int L          = ct_ckks.coeff_modulus_count();  // here likely 1

        //     // device → host copy for printing
        //     std::vector<Data64> host(bfv_words);
        //     HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
        //         host.data(),
        //         ct_ckks.device_locations_.data(),
        //         bytes,
        //         cudaMemcpyDeviceToHost,
        //         stream));
        //     HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        //     // prepare q-values (optional)
        //     const auto& ckks_moduli = contextckks_.get_key_modulus(); // host-side vector<Modulus64>
        //     std::vector<uint64_t> qvals(std::max(0, L));
        //     for (int i = 0; i < L; ++i) qvals[i] = ckks_moduli[i].value;

        //     print_ct_coeffs_inline(
        //         "[CKKS after copy] ",
        //         reinterpret_cast<const uint64_t*>(host.data()),
        //         poly_count,
        //         L,
        //         N,
        //         (L > 0 ? qvals.data() : nullptr),
        //         /*max_print=*/32
        //     );
        // }
        // // ==============================================

        ct_ckks.in_ntt_domain_ = ct_bfv.in_ntt_domain();
        ct_ckks.relinearization_required_ = ct_bfv.relinearization_required();

        const auto& ckks_moduli = contextckks_.get_key_modulus();
        const double q0 = static_cast<double>(ckks_moduli[0].value);
        const double t  = static_cast<double>(contextbfv_.get_plain_modulus().value);

        double initial_scale      = (q0 / t) * messageScaling_;
        double log2_scale         = std::log2(initial_scale);
        double rounded_log2_scale = std::round(log2_scale); // q0 / message_ratio_
        ct_ckks.scale_ = std::exp2(rounded_log2_scale);
        // ct_ckks.scale_ = 0;
        
        std::cout << "Initial scale set to: " << ct_ckks.scale_ << std::endl;
        ct_ckks.rescale_required_    = false;
        ct_ckks.ciphertext_generated_ = true;
        ct_ckks.scheme_ = heongpu::scheme_type::ckks;
        ct_ckks.ring_size_ = ct_bfv.ring_size_;
        ct_ckks.cipher_size_ = ct_bfv.size();
        
        return ct_ckks;
    }

    __host__ void HEHERA<heongpu::Scheme::RTF>::moddown_FV_inplace(
        Ciphertext<Scheme::RTF>& input,
        const ExecutionOptions& options
    )
    {
        cudaStream_t stream = options.stream_;
        if (stream == cudaStreamDefault) stream = input.stream();

        // std::cout << "Performing FV moddown..." << std::endl;
        // debug_print_ct_coeffs_bfv(input, *modulus_, stream);


        if (input.in_ntt_domain())
        {
            throw std::invalid_argument("Ciphertext must be in coefficient form for moddown.");
        }
        if (input.coeff_modulus_count() <= 1) {
            return;
        }

        input.store_in_device(stream);

        const int N = input.ring_size();
        const int poly_count = input.size();

        DeviceVector<Data64> scratch_buffer(input.memory_size(), stream);
        
        Data64* d_ptr_in = input.data();
        Data64* d_ptr_out = scratch_buffer.data();

        int initial_L_plus_1 = input.coeff_modulus_count();

        for (int L_plus_1 = initial_L_plus_1; L_plus_1 > 1; --L_plus_1) 
        {
            int current_L_idx = L_plus_1 - 1;
            int target_L = current_L_idx;

            size_t invq_base_idx = static_cast<size_t>(current_L_idx) * (current_L_idx - 1) / 2;

            const int threads = 256;
            const int blocks_x = (N + threads - 1) / threads;
            dim3 grid(blocks_x, target_L, poly_count);
            dim3 block(threads, 1, 1);

            moddown_FV_kernel<<<grid, block, 0, stream>>>(
                d_ptr_in,
                d_ptr_out,
                modulus_->data(),
                qhalf_->data(),
                invq_->data(),
                n_power,
                current_L_idx,
                invq_base_idx);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            std::swap(d_ptr_in, d_ptr_out);
        }
        
        const size_t final_size_words = static_cast<size_t>(poly_count) * 1 * N; 
        
        if (d_ptr_in != input.data()) {
            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(input.data(), d_ptr_in,
                                            final_size_words * sizeof(Data64),
                                            cudaMemcpyDeviceToDevice, stream));
        }

        input.device_locations_.resize(final_size_words, stream);
        input.coeff_modulus_count_ = 1; 

        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        // std::cout << "FV moddown completed." << std::endl;
        // debug_print_ct_coeffs_bfv(input, *modulus_, stream);

        return;
    }

    __host__ void HEHERA<heongpu::Scheme::RTF>::gen_FV_moddown_params(
        const ExecutionOptions& options
    ){
        cudaStream_t stream = options.stream_; 
        std::cout << "Generating FV moddown parameters..." << std::endl;
        
        std::vector<Modulus64> host_modulus(Q_size_);
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(host_modulus.data(), modulus_->data(),
                                        Q_size_ * sizeof(Modulus64),
                                        cudaMemcpyDeviceToHost, stream));
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<Data64> host_qhalf(Q_size_);
        for (int i = 0; i < Q_size_; ++i) {
            host_qhalf[i] = host_modulus[i].value >> 1 ;
        }

        // invq_ 계산 (Flattened 2D Array)
        size_t invq_total_size = static_cast<size_t>(Q_size_) * (Q_size_ - 1) / 2;
        std::vector<Data64> host_invq(invq_total_size);
        
        size_t flattened_idx = 0;
        for (int i = 1; i < Q_size_; ++i) {
            for (int j = 0; j < i; ++j) {
                uint64_t q_i = host_modulus[i].value;
                uint64_t q_j = host_modulus[j].value;
                Data64 inv_result = modInverseHERA(q_i, q_j);

                host_invq[flattened_idx++] = inv_result;
            }
        }

        qhalf_ = std::make_shared<DeviceVector<Data64>>(host_qhalf, stream);
        invq_ = std::make_shared<DeviceVector<Data64>>(host_invq, stream);
    }



}