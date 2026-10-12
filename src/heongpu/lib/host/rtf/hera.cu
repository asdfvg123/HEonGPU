
#include "rtf/hera.cuh"

#include <algorithm>
#include <atomic>
#include <future>
#include <sched.h>
#include <set>
#include <thread>


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
                                const ExecutionOptions& options,
                                StreamCipher cipher
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
        
        cipher_ = cipher;
        const bool is_hera = (cipher_ == StreamCipher::Hera);
        block_ = is_hera ? hera_params_.n : rubato_params_.n;
        l_ = is_hera ? hera_params_.n : rubato_params_.l;
        const int block = block_;
        if (n % block != 0 || (n >> 1) % block != 0)
        {
            throw std::invalid_argument(
                "HEHERA: N/2 must be a multiple of the cipher block size.");
        }
        if (plain_modulus_.value !=
            (is_hera ? hera_params_.q : rubato_params_.q))
        {
            throw std::invalid_argument(
                "HEHERA: plain modulus must be the cipher modulus q.");
        }
        round_ = is_hera ? hera_params_.rounds : rubato_params_.rounds;
        rc_vec_size_ = static_cast<size_t>(round_ + 1) * n;
        // ic = (1, 2, ..., n) in every block
        icVec.resize(n);
        for (int i = 0; i < n; i++)
        {
            icVec[i] = (i % block) + 1;
        }
        icPt_ = Plaintext<Scheme::RTF>(context);
        icCt_ = Ciphertext<Scheme::RTF>(context);
        
        encoder_.encode(icPt_, icVec, options);
        encryptor_.encrypt(icCt_, icPt_, options);
        
        icCt_.store_in_device(stream);

        // Round constants are set per nonce by set_nonce / set_xof_seeds.

        const size_t row_len = n >> 1; // BFV batching row length
        const size_t B = block;
        const size_t blocks_per_row = row_len / B;

        
        if (!is_hera)
        {
            std::vector<uint64_t> feistel_mask(n);
            for (size_t i = 0; i < n; ++i)
                feistel_mask[i] = (i % B == 0) ? 0 : 1;
            encoder_.encode(feistel_maskPt_, feistel_mask, options);
        }


        // MixRows o MixColumns = M8 (x) M8 on every 64-slot Rubato block
        // (M4 (x) M4 on every 16-slot HERA block), as a slot-domain transform
        // with diagonals at rotations -(B-1)..B-1. rows_kept = l zeroes the
        // truncated output words (Tr_{n,l}, Rubato only).
        auto entry = [&](int r, int c) -> uint64_t
        { return is_hera ? hera::linear_entry(r, c) : M64[r][c]; };
        auto build_diagonals = [&](int rows_kept)
        {
            SlotLinearTransform::DiagMap diags;
            for (int rot = -(int) B + 1; rot < (int) B; ++rot)
            {
                std::vector<uint64_t> d(n, 0ULL);
                bool nonzero = false;
                for (size_t row = 0; row < 2; ++row)
                    for (size_t blk = 0; blk < blocks_per_row; ++blk)
                    {
                        const size_t base = row * row_len + blk * B;
                        for (int r = 0; r < rows_kept; ++r)
                        {
                            const int c = r + rot;
                            if (c >= 0 && c < (int) B)
                            {
                                d[base + r] = entry(r, c);
                                nonzero = true;
                            }
                        }
                    }
                if (nonzero)
                    diags[rot] = std::move(d);
            }
            return diags;
        };
        // Both start their baby steps at rotation -(B - 1), so they share
        // the (hoisted) baby-step keys.
        const int base_rot = -(int) B + 1;
        linear_full_ = SlotLinearTransform(context, encoder, op,
                                           build_diagonals((int) B), options,
                                           base_rot);
        if (!is_hera)
            linear_trunc_ = SlotLinearTransform(
                context, encoder, op, build_diagonals(rubato_params_.l),
                options, base_rot);
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    }
    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::precompute(
        heongpu::Ciphertext<heongpu::Scheme::RTF>& key,
        const heongpu::ExecutionOptions& options)
    {
        if (!rc_ready_)
        {
            throw std::logic_error(
                "HEHERA: round constants not set; call set_nonce first.");
        }
        rckCt.clear();
        rckCt.reserve(round_+1);
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
        if (cipher_ == StreamCipher::Hera)
        {
            std::vector<HEHERA*> stages(round_ + 1, this);
            std::vector<Ciphertext<Scheme::RTF>*> keys(round_ + 1, &key);
            return gen_stream_key_staged(stages, keys, options);
        }
        Ciphertext<Scheme::RTF> result(contextbfv_);
        precompute(key, options);

        // ark
        add_round_key(icCt_, rckCt[0], result, options);
        // std::cout << "here3" << std::endl;
        
        // round_function
        for (int r = 1; r < round_; ++r)
        {
            linear(result, result, options);
            feistel(result, result, options);
            add_round_key(result, rckCt[r], result, options);

        }

        // Fin = Tr o ARK o MR o MC o Feistel o MR o MC
        linear(result, result, options);
        feistel(result, result, options);
        linear(result, result, options, /*truncate=*/true);
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
        const ExecutionOptions& options)
    {
        // square, rotate and add
        auto stream = options.stream_;
        heongpu::Ciphertext<heongpu::Scheme::RTF> temp =
            operator_.operator_ciphertext(stream);

        operator_.multiply(input, input, temp, options);
        operator_.relinearize_inplace(temp, relin_key_, options);

        operator_.rotate_rows_inplace(temp, galois_key_hera_, -1, options); // right shift by 1
        // mask
        operator_.multiply_plain_inplace(temp, feistel_maskPt_, options);

        operator_.add(input, temp, output, options);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::cube(
        Ciphertext<Scheme::RTF>& input, 
        Ciphertext<Scheme::RTF>& output,
        const ExecutionOptions& options)
    {
        cudaStream_t stream = options.stream_;
        if (stream == cudaStreamDefault) stream = input.stream();

        Ciphertext<Scheme::RTF> sq(contextbfv_);
        operator_.multiply(input, input, sq, options);
        operator_.relinearize_inplace(sq, relin_key_, options);
        operator_.multiply(sq, input, sq, options);
        operator_.relinearize_inplace(sq, relin_key_, options);
        output = std::move(sq);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::linear(
        heongpu::Ciphertext<heongpu::Scheme::RTF>& input,
        heongpu::Ciphertext<heongpu::Scheme::RTF>& output,
        const ExecutionOptions& options, bool truncate)
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

        const SlotLinearTransform& lt =
            (truncate && cipher_ == StreamCipher::Rubato) ? linear_trunc_
                                                          : linear_full_;
        auto y = lt.apply(operator_, contextbfv_, x, galois_key_hera_, opt);
        output = y;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::gen_FV_S2C_FFT(
        int levels_per_group, const ExecutionOptions& options)
    {
        s2c_fft_ = std::make_shared<FVSlotToCoeffFFT>(
            contextbfv_, encoder_, operator_, levels_per_group, options);
    }

    __host__ std::vector<int>
    heongpu::HEHERA<heongpu::Scheme::RTF>::s2c_fft_rotation_shifts() const
    {
        if (!s2c_fft_)
            throw std::logic_error("HEHERA: call gen_FV_S2C_FFT first.");
        return s2c_fft_->rotation_shifts();
    }

    __host__ std::vector<int>
    heongpu::HEHERA<heongpu::Scheme::RTF>::rubato_rotation_shifts() const
    {
        std::set<int> s;
        for (int v : linear_full_.rotation_shifts())
            s.insert(v);
        if (cipher_ == StreamCipher::Rubato)
        {
            s.insert(-1); // Feistel
            for (int v : linear_trunc_.rotation_shifts())
                s.insert(v);
        }
        return std::vector<int>(s.begin(), s.end());
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::set_nonce(
        const std::array<uint8_t, 8>& nonce, uint64_t counter_start,
        const ExecutionOptions& options)
    {
        set_round_constants(compute_round_constants(nonce, counter_start),
                            options);
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::set_xof_seeds(
        const std::vector<std::vector<uint8_t>>& seeds,
        const ExecutionOptions& options)
    {
        set_round_constants(compute_round_constants(seeds), options);
    }

    __host__ heongpu::HEHERA<heongpu::Scheme::RTF>::RoundConstants
    heongpu::HEHERA<heongpu::Scheme::RTF>::compute_round_constants(
        const std::array<uint8_t, 8>& nonce, uint64_t counter_start,
        int max_threads) const
    {
        std::vector<std::vector<uint8_t>> seeds(get_block_count());
        for (int j = 0; j < get_block_count(); ++j)
            seeds[j] = rubato::xof_seed(nonce, counter_start + j);
        return compute_round_constants(seeds, max_threads);
    }

    __host__ heongpu::HEHERA<heongpu::Scheme::RTF>::RoundConstants
    heongpu::HEHERA<heongpu::Scheme::RTF>::compute_round_constants(
        const std::vector<std::vector<uint8_t>>& seeds, int max_threads) const
    {
        const int block = block_;
        const int blocks = get_block_count();
        const bool is_hera = (cipher_ == StreamCipher::Hera);
        if ((int) seeds.size() != blocks)
        {
            throw std::invalid_argument(
                "HEHERA::compute_round_constants: need one seed per block.");
        }

        RoundConstants rc_all(round_ + 1, std::vector<uint64_t>(n, 0ULL));
        // One XOF per block; blocks are independent, so split them over the
        // CPUs this process may run on.
        auto fill = [&](int j0, int j1)
        {
            for (int j = j0; j < j1; ++j)
            {
                auto rc = is_hera
                              ? hera::round_constants(seeds[j], hera_params_)
                              : rubato::round_constants(seeds[j],
                                                        rubato_params_);
                for (int r = 0; r <= round_; ++r)
                    for (int i = 0; i < block; ++i)
                        rc_all[r][j * block + i] = rc[r][i];
                // Tr_{n,l} o ARK: the truncated words get no round key.
                for (int i = l_; i < block; ++i)
                    rc_all[round_][j * block + i] = 0;
            }
        };
        int n_threads = 1;
        {
            cpu_set_t set;
            CPU_ZERO(&set);
            if (sched_getaffinity(0, sizeof(set), &set) == 0)
                n_threads = CPU_COUNT(&set);
        }
        if (max_threads > 0)
            n_threads = std::min(n_threads, max_threads);
        n_threads = std::max(1, std::min(n_threads, blocks / 16));
        if (n_threads == 1)
        {
            fill(0, blocks);
        }
        else
        {
            std::vector<std::thread> pool;
            const int per = (blocks + n_threads - 1) / n_threads;
            for (int t = 0; t < n_threads; ++t)
            {
                const int j0 = t * per, j1 = std::min(blocks, j0 + per);
                if (j0 < j1)
                    pool.emplace_back(fill, j0, j1);
            }
            for (auto& th : pool)
                th.join();
        }
        return rc_all;
    }

    __host__ std::future<heongpu::HEHERA<heongpu::Scheme::RTF>::RoundConstants>
    heongpu::HEHERA<heongpu::Scheme::RTF>::prepare_nonce_async(
        const std::array<uint8_t, 8>& nonce, uint64_t counter_start,
        int max_threads) const
    {
        return std::async(std::launch::async,
                          [this, nonce, counter_start, max_threads]()
                          {
                              return compute_round_constants(
                                  nonce, counter_start, max_threads);
                          });
    }

    __host__ void heongpu::HEHERA<heongpu::Scheme::RTF>::set_round_constants(
        const RoundConstants& rc, const ExecutionOptions& options,
        const std::vector<int>& rounds)
    {
        if ((int) rc.size() != round_ + 1)
            throw std::invalid_argument("HEHERA::set_round_constants: rounds");
        rcVec = rc;
        rcPt.resize(round_ + 1);
        for (int r = 0; r < round_ + 1; ++r)
        {
            if (!rounds.empty() &&
                std::find(rounds.begin(), rounds.end(), r) == rounds.end())
                continue;
            Plaintext<Scheme::RTF> pt(contextbfv_);
            encoder_.encode(pt, rcVec[r], options);
            pt.store_in_device(options.stream_);
            rcPt[r] = std::move(pt);
        }
        rc_ready_ = true;
    }

    __host__ void
    heongpu::HEHERA<heongpu::Scheme::RTF>::set_round_constants_staged(
        const std::vector<HEHERA*>& stages, const RoundConstants& rc,
        const ExecutionOptions& options)
    {
        // Rounds of each distinct stage object.
        std::vector<std::pair<HEHERA*, std::vector<int>>> own;
        for (int i = 0; i < (int) stages.size(); ++i)
        {
            auto it = std::find_if(own.begin(), own.end(),
                                   [&](auto& p) { return p.first == stages[i]; });
            if (it == own.end())
                own.push_back({stages[i], {i}});
            else
                it->second.push_back(i);
        }
        for (auto& [st, rounds] : own)
            st->set_round_constants(rc, options, rounds);
    }

    __host__ Ciphertext<Scheme::RTF>
    heongpu::HEHERA<heongpu::Scheme::RTF>::gen_stream_key_staged(
        const std::vector<HEHERA*>& stages,
        const std::vector<Ciphertext<Scheme::RTF>*>& keys,
        const ExecutionOptions& options,
        const std::function<void(int, Ciphertext<Scheme::RTF>&)>& probe)
    {
        if (stages.empty() || stages[0]->cipher_ != StreamCipher::Hera ||
            (int) stages.size() != stages[0]->round_ + 1 ||
            keys.size() != stages.size())
            throw std::invalid_argument(
                "gen_stream_key_staged: need r + 1 HERA stages and keys");
        for (auto* st : stages)
            if (!st->rc_ready_)
                throw std::logic_error(
                    "HEHERA: round constants not set; call set_nonce first.");
        const int r_last = (int) stages.size() - 1;
        auto opt = ExecutionOptions()
                       .set_stream(options.stream_)
                       .set_storage_type(storage_type::DEVICE)
                       .set_initial_location(true);

        // ARK_i on stage i: state += k . rc_i
        auto ark = [&](Ciphertext<Scheme::RTF>& state, int i)
        {
            HEHERA& S = *stages[i];
            Ciphertext<Scheme::RTF> rk(S.contextbfv_);
            S.operator_.multiply_plain(*keys[i], S.rcPt[i], rk, opt);
            S.operator_.add_inplace(state, rk, opt);
        };

        // ARK_0(ic) = ic + k . rc_0 (ic as a plaintext)
        Ciphertext<Scheme::RTF> st(stages[0]->contextbfv_);
        stages[0]->operator_.multiply_plain(*keys[0], stages[0]->rcPt[0], st,
                                            opt);
        stages[0]->operator_.add_plain_inplace(st, stages[0]->icPt_, opt);
        for (int i = 0; i < r_last; ++i)
        {
            HEHERA& S = *stages[i];
            S.linear(st, st, opt);
            S.cube(st, st, opt);
            if (stages[i + 1] != stages[i])
                st = S.mod_switch_to(st, stages[i + 1]->contextbfv_, opt);
            if (probe)
                probe(i, st);
            if (i + 1 < r_last)
                ark(st, i + 1);
        }
        stages[r_last]->linear(st, st, opt);
        ark(st, r_last);
        if (probe)
            probe(r_last, st);
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
        return st;
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

        // Diagonal s of the slot -> coefficient matrix, pre-rotated by its
        // giant step. O(N^2) host work: the diagonals are computed by a pool
        // of threads (one batch at a time), then encoded on the GPU.
        auto build_diag = [&](int s, std::vector<uint64_t>& diag_slots)
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
        };

        int n_threads = 1;
        {
            cpu_set_t set;
            if (sched_getaffinity(0, sizeof(set), &set) == 0)
                n_threads = CPU_COUNT(&set);
            n_threads = std::max(1, std::min(n_threads, 64));
        }
        const int batch = std::min(N, 4 * n_threads);
        std::vector<std::vector<uint64_t>> diag_batch(
            batch, std::vector<uint64_t>(N));

        for (int s0 = 0; s0 < N; s0 += batch)
        {
            const int nb = std::min(batch, N - s0);
            {
                std::vector<std::thread> pool;
                std::atomic<int> next(0);
                const int nt = std::min(n_threads, nb);
                for (int t = 0; t < nt; ++t)
                {
                    pool.emplace_back([&]()
                    {
                        for (int b = next++; b < nb; b = next++)
                            build_diag(s0 + b, diag_batch[b]);
                    });
                }
                for (auto& th : pool)
                    th.join();
            }

            for (int b = 0; b < nb; ++b)
            {
                const int s = s0 + b;
                encoder_.encode(pt_diag, diag_batch[b], options);

                const Data64* src_plain = pt_diag.data();                // length = N (plain)
                Data64* dst = (s < H)
                    ? blob0.data() + ((size_t)s) * per_diag_len_plain
                    : blob1.data() + ((size_t)(s - H)) * per_diag_len_plain;
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
        if (s2c_fft_)
        {
            return s2c_fft_->apply(ct_in, galois_key_s2c_, options);
        }
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
        // word i goes where S2C_FV puts keystream slot i
        for (size_t i = 0; i < use_len; ++i)
            coeffs[s2c_coeff_index((int) i)] = t ? (input[i] % t) : input[i];

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
        {
            // The residues are copied as they are: the FV and CKKS chains
            // must share their first primes (RtF uses one chain for both).
            const auto ckks_primes = contextckks_.get_key_modulus();
            for (int i = 0; i < ct_bfv.coeff_modulus_count(); ++i)
            {
                if (ckks_primes[i].value != prime_vector_[i].value)
                {
                    throw std::invalid_argument(
                        "transcipher_bfv2ckks: FV and CKKS moduli differ.");
                }
            }
        }

        Ciphertext<Scheme::CKKS> ct_ckks(contextckks_, options);
        ct_ckks.coeff_modulus_count_ = ct_bfv.coeff_modulus_count();
        ct_ckks.depth_ = contextckks_.get_ciphertext_modulus_count() - ct_bfv.coeff_modulus_count();

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

        // After the mod-down the FV ciphertext encrypts round(q0/t * c) with
        // c = round(Delta * m), i.e. m at scale q0 * Delta / t (= q0 / ratio).
        ct_ckks.scale_ = (q0 / t) * messageScaling_;
        ct_ckks.rescale_required_    = false;
        ct_ckks.ciphertext_generated_ = true;
        ct_ckks.scheme_ = heongpu::scheme_type::ckks;
        ct_ckks.ring_size_ = ct_bfv.ring_size_;
        ct_ckks.cipher_size_ = ct_bfv.size();
        
        return ct_ckks;
    }

    __host__ Ciphertext<Scheme::RTF> HEHERA<heongpu::Scheme::RTF>::mod_switch_to(
        Ciphertext<Scheme::RTF>& input, HEContext<Scheme::RTF>& target,
        const ExecutionOptions& options)
    {
        cudaStream_t stream = options.stream_;
        if (stream == cudaStreamDefault) stream = input.stream();

        const int target_count = target.Q_size;
        if (target.n != n || target.plain_modulus_.value != plain_modulus_.value)
            throw std::invalid_argument("mod_switch_to: ring or t differs");
        if (target_count < 1 || target_count > input.coeff_modulus_count())
            throw std::invalid_argument("mod_switch_to: bad target level");
        for (int i = 0; i < target_count; i++)
            if (target.prime_vector_[i].value != prime_vector_[i].value)
                throw std::invalid_argument(
                    "mod_switch_to: target moduli are not a prefix");
        if (input.in_ntt_domain())
            operator_.transform_from_ntt_inplace(input, options);
        input.store_in_device(stream);

        const int poly_count = input.size();
        DeviceVector<Data64> buf_a(input.memory_size(), stream);
        DeviceVector<Data64> buf_b(input.memory_size(), stream);
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
            buf_a.data(), input.data(), input.memory_size() * sizeof(Data64),
            cudaMemcpyDeviceToDevice, stream));
        Data64* d_in = buf_a.data();
        Data64* d_out = buf_b.data();
        for (int L_plus_1 = input.coeff_modulus_count();
             L_plus_1 > target_count; --L_plus_1)
        {
            const int current_L_idx = L_plus_1 - 1;
            const size_t invq_base_idx =
                static_cast<size_t>(current_L_idx) * (current_L_idx - 1) / 2;
            dim3 grid((n + 255) / 256, current_L_idx, poly_count);
            moddown_FV_kernel<<<grid, dim3(256, 1, 1), 0, stream>>>(
                d_in, d_out, modulus_->data(), qhalf_->data(), invq_->data(),
                n_power, current_L_idx, invq_base_idx);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            std::swap(d_in, d_out);
        }

        const size_t words = (size_t) poly_count * target_count * n;
        Ciphertext<Scheme::RTF> out(target);
        out.device_locations_ = DeviceVector<Data64>(words, stream);
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(out.device_locations_.data(), d_in,
                                           words * sizeof(Data64),
                                           cudaMemcpyDeviceToDevice, stream));
        out.scheme_ = input.scheme_;
        out.ring_size_ = input.ring_size_;
        out.coeff_modulus_count_ = target_count;
        out.cipher_size_ = poly_count;
        out.in_ntt_domain_ = false;
        out.relinearization_required_ = input.relinearization_required_;
        out.ciphertext_generated_ = true;
        out.storage_type_ = storage_type::DEVICE;
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        return out;
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

    __host__ std::vector<int64_t> HEHERA<heongpu::Scheme::RTF>::decrypt_raw_q0(
        Ciphertext<Scheme::CKKS>& ct, Secretkey<Scheme::RTF>& sk,
        const ExecutionOptions& options)
    {
        if (ct.coeff_modulus_count() != 1 || ct.in_ntt_domain() ||
            ct.size() != 2)
        {
            throw std::invalid_argument(
                "decrypt_raw_q0: need a 2-component ciphertext at q0 in "
                "coefficient form.");
        }
        return decrypt_raw_q0_impl(ct.data(), sk, options.stream_);
    }

    __host__ std::vector<int64_t> HEHERA<heongpu::Scheme::RTF>::decrypt_raw_q0(
        Ciphertext<Scheme::RTF>& ct, Secretkey<Scheme::RTF>& sk,
        const ExecutionOptions& options)
    {
        if (ct.coeff_modulus_count() != 1 || ct.in_ntt_domain() ||
            ct.size() != 2)
        {
            throw std::invalid_argument(
                "decrypt_raw_q0: need a 2-component ciphertext at q0 in "
                "coefficient form.");
        }
        return decrypt_raw_q0_impl(ct.data(), sk, options.stream_);
    }

    __host__ std::vector<int64_t>
    HEHERA<heongpu::Scheme::RTF>::decrypt_raw_q0_impl(
        const Data64* ct_device, Secretkey<Scheme::RTF>& sk,
        cudaStream_t stream)
    {
        // The secret key is kept in NTT form over Q x P; residue 0 is s mod q0.
        if (!sk.is_on_device())
        {
            sk.store_in_device(stream);
        }

        // c1 * s at q0 (the first residue of the secret key), NTT domain.
        DeviceVector<Data64> c1s(n, stream);
        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};
        gpuntt::GPU_NTT(const_cast<Data64*>(ct_device) + n, c1s.data(),
                        ntt_table_->data(), modulus_->data(), cfg_ntt, 1, 1);
        sk_multiplication<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            c1s.data(), sk.data(), c1s.data(), modulus_->data(), n_power, 1);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};
        gpuntt::GPU_NTT_Inplace(c1s.data(), intt_table_->data(),
                                modulus_->data(), cfg_intt, 1, 1);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        std::vector<Data64> h_c0(n), h_c1s(n);
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(h_c0.data(), ct_device,
                                           n * sizeof(Data64),
                                           cudaMemcpyDeviceToHost, stream));
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(h_c1s.data(), c1s.data(),
                                           n * sizeof(Data64),
                                           cudaMemcpyDeviceToHost, stream));
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        const Data64 q0 = prime_vector_[0].value;
        std::vector<int64_t> out(n);
        for (int i = 0; i < n; ++i)
        {
            unsigned __int128 v =
                ((unsigned __int128) h_c0[i] + h_c1s[i]) % q0;
            const Data64 r = static_cast<Data64>(v);
            out[i] = (r > q0 / 2) ? -static_cast<int64_t>(q0 - r)
                                  : static_cast<int64_t>(r);
        }
        return out;
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