#ifndef HEONGPU_RTF_HERA_H
#define HEONGPU_RTF_HERA_H

#include "ntt.cuh"
#include "fft.cuh"
#include "addition.cuh"
#include "multiplication.cuh"
#include "switchkey.cuh"
#include "keygeneration.cuh"
#include "bootstrapping.cuh"

#include "rtf/context.cuh"
#include "rtf/encoder.cuh"
#include "rtf/encryptor.cuh"
#include "rtf/plaintext.cuh"
#include "rtf/ciphertext.cuh"
#include "rtf/evaluationkey.cuh"
#include "rtf/operator.cuh"

namespace heongpu
{

    /**
     * @brief HERA is responsible for generating FV encrypted stream key
     * and homomorphically decrypting ciphertext.
     *
     */
    template <> class HEHERA<Scheme::RTF>
    {
    public:
        /**
         * @brief Construct a new HEHERA object with the given parameters.
         */
        __host__ HEHERA(HEContext<Scheme::RTF>& context,
                         HEEncoder<Scheme::RTF>& encoder,
                         HEEncryptor<Scheme::RTF>& encryptor,
                         HEOperator<Scheme::RTF>& op,
                         Galoiskey<Scheme::RTF>& galois_key,
                         Relinkey<Scheme::RTF>& relin_key,
                        const ExecutionOptions& options = ExecutionOptions()
                        );


        HEHERA() = default;
        HEHERA(const HEHERA& copy) = default;
        HEHERA(HEHERA&& source) = default;
        HEHERA& operator=(const HEHERA& assign) = default;
        HEHERA& operator=(HEHERA&& assign) = default;

        __host__ void gen_FV_S2C_Matrix(
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ void precompute(Ciphertext<Scheme::RTF>& key, 
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::RTF> gen_stream_key(
            // heongpu::DeviceVector<Data64>& nonce,
            Ciphertext<Scheme::RTF>& ctkey,
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ void add_round_key(
            Ciphertext<Scheme::RTF>& input,
            Ciphertext<Scheme::RTF>& round_key,
            Ciphertext<Scheme::RTF>& output,
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ void feistel(
            Ciphertext<Scheme::RTF>& input,
            Ciphertext<Scheme::RTF>& output,
            Galoiskey<Scheme::RTF>& galois_key,
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ void cube(
            Ciphertext<Scheme::RTF>& input,
            Ciphertext<Scheme::RTF>& output,
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ void linear(
            Ciphertext<Scheme::RTF>& input,
            Ciphertext<Scheme::RTF>& output,
            const ExecutionOptions& options = ExecutionOptions()
        );
        
        __host__ Ciphertext<Scheme::RTF> S2C_FV(
            Ciphertext<Scheme::RTF>& ct_in,
            const ExecutionOptions& options = ExecutionOptions()
        );



        __host__ heongpu::Ciphertext<Scheme::RTF> get_icCt(){
            return icCt_;
        };
        __host__ [[nodiscard]]
        const std::vector<std::vector<uint64_t>>& get_rcVec() const noexcept {
            return rcVec;
        }
        __host__ std::vector<Ciphertext<Scheme::RTF>> get_rckCt(){
            return rckCt;
        };

        __host__ Data64 get_plain_psi(){
            return plain_psi_;
        };

        __host__ void print_s2c_matrix_diagonal(
            int k,
            const ExecutionOptions& options = ExecutionOptions())
        {
            cudaStream_t stream = options.stream_;
            const int N = static_cast<int>(context_.n);

            // --- 1. Validation ---
            if (k < 0 || k >= N) {
                std::cerr << "[ERROR] Diagonal index " << k << " is out of bounds [0, " 
                        << N - 1 << "]." << std::endl;
                return;
            }
            if (s2c_matrix_diagonals_.empty() || s2c_matrix_diagonals_[0].empty()) {
                std::cerr << "[ERROR] S2C matrix diagonals have not been generated. "
                        << "Call gen_FV_S2C_Matrix() first." << std::endl;
                return;
            }

            // --- 2. Access Diagonal's Coefficient Data from the Blob ---
            // All diagonals are stored contiguously in a single DeviceVector.
            const auto& blob = s2c_matrix_diagonals_[0][0];
            const Data64* diagonal_coeffs_device_ptr = blob.data() + static_cast<size_t>(k) * N;

            // --- 3. Prepare a Plaintext for Decoding ---
            // Create a Plaintext object to hold the diagonal's coefficients for the decoder.
            heongpu::Plaintext<heongpu::Scheme::RTF> pt_diag_coeffs(context_);
            {
                // Create a temporary device vector and copy the coefficient data into it.
                heongpu::DeviceVector<Data64> tmp_coeffs(N, stream);
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    tmp_coeffs.data(),
                    diagonal_coeffs_device_ptr,
                    N * sizeof(Data64),
                    cudaMemcpyDeviceToDevice,
                    stream));
                
                // Set the plaintext's memory to this vector of coefficients.
                pt_diag_coeffs.memory_set(std::move(tmp_coeffs));
            }
            
            // --- 4. Decode to Recover Original Slot Values ---
            // The 'decode' function reverses the 'encode' process. It takes the polynomial
            // in coefficient form and returns the original values that were in the slots.
            std::vector<uint64_t> diag_slots;
            encoder_.decode(diag_slots, pt_diag_coeffs, options);

            // --- 5. Synchronize and Print Results ---
            if (stream != cudaStreamDefault) {
                cudaStreamSynchronize(stream);
            }

            std::cout << "\n## S2C Matrix Diagonal k=" << k << " (Slot Values) ##" << std::endl;
            // To avoid excessive output, let's print the first 64 values.
            const int print_count = std::min(N, 64); 
            for (int i = 0; i < print_count; ++i) {
                std::cout << diag_slots[i] << (((i + 1) % 16 == 0) ? "\n" : " ");
            }
            if (print_count % 16 != 0 || print_count == 0) {
                std::cout << std::endl;
            }
            if (N > print_count) {
                std::cout << "(... and " << N - print_count << " more values)" << std::endl;
            }
        }

        __host__ void print_ntt_table_host(std::size_t count,
                                   cudaStream_t stream = cudaStreamDefault) const {
            if (stream == cudaStreamDefault) {
                // Use the vector’s stream if available
                stream = ntt_table_->stream();
            }

            const std::size_t k = std::min<std::size_t>(count, ntt_table_->size());
            std::vector<Root64> host(k);

            HEONGPU_CUDA_CHECK(cudaMemcpyAsync(host.data(),
                                    ntt_table_->data(),
                                    k * sizeof(Root64),
                                    cudaMemcpyDeviceToHost,
                                    stream));
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

            std::cout << "[ntt_table] ";
            for (std::size_t i = 0; i < k; ++i) {
                std::cout << host[i] << (i + 1 == k ? '\n' : ' ');
            }
        };

        __host__ void print_s2c_matrix_diagonals()
        {
            if (s2c_groups_no_swap_.empty()) {
                std::cout << "[S2C] No diagonals generated. Did you call gen_FV_S2C_Matrix()?\n";
                return;
            }

            const std::size_t stages = s2c_groups_no_swap_.size();
            std::cout << "[S2C] Diagonals: stages=" << stages << ", N=" << n << "\n";

            // How many entries per diagonal to print from the head/tail
            const std::size_t head = std::min<std::size_t>(16, static_cast<std::size_t>(n));
            const std::size_t tail = std::min<std::size_t>(16,  static_cast<std::size_t>(n));

            for (std::size_t s = 0; s < stages; ++s) {
                const auto& group = s2c_groups_no_swap_[s];
                std::cout << "  [stage " << s << "] terms=" << group.size()
                        << " (expected 2: diag0, diag1)\n";

                for (std::size_t j = 0; j < group.size(); ++j) {
                    const auto& dvec = group[j];
                    cudaStream_t stream = dvec.stream();

                    // Head sample
                    std::vector<Data64> h_head(head, 0);
                    HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                        h_head.data(), dvec.data(), head * sizeof(Data64),
                        cudaMemcpyDeviceToHost, stream));
                    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

                    std::cout << "    diag" << j << " [0:" << head << "): ";
                    for (std::size_t k = 0; k < head; ++k) {
                        std::cout << h_head[k] << (k + 1 == head ? '\n' : ' ');
                    }

                    // Tail sample (optional)
                    if (static_cast<std::size_t>(n) > head && tail > 0) {
                        std::vector<Data64> h_tail(tail, 0);
                        const Data64* dev_ptr_tail = dvec.data() + (n - tail);
                        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                            h_tail.data(), dev_ptr_tail, tail * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream));
                        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

                        std::cout << "    diag" << j << " [tail " << (n - static_cast<int>(tail))
                                << ":" << n << "): ";
                        for (std::size_t k = 0; k < tail; ++k) {
                            std::cout << h_tail[k] << (k + 1 == tail ? '\n' : ' ');
                        }
                    }
                }
            }
        }

        __host__ void print_s2c_matrix_shifts()
        {
            if (s2c_matrix_shifts_.empty()) {
                std::cout << "[S2C] No shifts generated. Did you call gen_FV_S2C_Matrix()?\n";
                return;
            }

            std::cout << "[S2C] Shifts per stage:\n";
            for (std::size_t s = 0; s < s2c_matrix_shifts_.size(); ++s) {
                const auto& sh = s2c_matrix_shifts_[s];
                std::cout << "  [stage " << s << "] {";
                for (std::size_t i = 0; i < sh.size(); ++i) {
                    std::cout << sh[i] << (i + 1 == sh.size() ? "" : ", ");
                }
                std::cout << "}\n";
            }
        }

        __host__ Data64 get_psi() const {
            return plain_psi_;
        }

        __host__ int get_n_power() const {
            return n_power;
        }

        static inline uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod) {
            unsigned __int128 p = (unsigned __int128)a * b;
            return (uint64_t)(p % mod);
        }
        static inline uint64_t add_mod_u64(uint64_t a, uint64_t b, uint64_t mod) {
            uint64_t c = a + b; if (c >= mod) c -= mod; return c;
        }
        static inline uint64_t sub_mod_u64(uint64_t a, uint64_t b, uint64_t mod) {
            return (a >= b) ? (a - b) : (a + mod - b);
        }
        static inline uint64_t pow_mod_u64(uint64_t base, uint64_t exp, uint64_t mod) {
            uint64_t res = 1 % mod, x = base % mod;
            while (exp) { if (exp & 1) res = mul_mod_u64(res, x, mod); x = mul_mod_u64(x, x, mod); exp >>= 1; }
            return res;
        }
        static inline int bit_reverse_int(int x, int bits) {
            unsigned r = 0, v = (unsigned)x;
            for (int i = 0; i < bits; ++i) { r = (r << 1) | (v & 1); v >>= 1; }
            return (int)r;
            }

        // Inverse NTT (radix-2 DIT, in-place), twiddle = wInv^(N/m), final scale by invN
        // Input/Output in natural order (we apply bit-reverse first).
        static void intt_inplace_dit(std::vector<uint64_t>& a, uint64_t t, uint64_t w, int N) {
            const int bits = (int)std::log2((double)N);
            // bit-reverse permutation
            for (int i = 0; i < N; ++i) {
                int j = bit_reverse_int(i, bits);
                if (j > i) std::swap(a[i], a[j]);
            }
            // use wInv twiddles
            const uint64_t wInv = pow_mod_u64(w, (uint64_t)N - 1, t);
            for (int m = 2; m <= N; m <<= 1) {
                uint64_t wm = pow_mod_u64(wInv, (uint64_t)(N / m), t);
                for (int k = 0; k < N; k += m) {
                    uint64_t wcur = 1;
                    for (int j = 0; j < m/2; ++j) {
                        uint64_t u = a[k + j];
                        uint64_t v = mul_mod_u64(a[k + j + m/2], wcur, t);
                        a[k + j]         = add_mod_u64(u, v, t);
                        a[k + j + m/2]   = sub_mod_u64(u, v, t);
                        wcur = mul_mod_u64(wcur, wm, t);
                    }
                }
            }
            // scale by invN
            const uint64_t invN = pow_mod_u64((uint64_t)N, (uint64_t)t - 2, t); // t prime → Fermat
            for (int i = 0; i < N; ++i) a[i] = mul_mod_u64(a[i], invN, t);
        }

        // 128-bit safe mul-mod
        static inline __host__ __device__ uint64_t mul_mod_u128(uint64_t a, uint64_t b, uint64_t m) {
            return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) % m);
        }
        static void bit_reverse_vector(std::vector<uint64_t>& vec) {
            const size_t n = vec.size();
            if (n == 0) return;

            // Determine number of bits
            size_t log_n = 0;
            while ((1UL << log_n) < n) {
                log_n++;
            }

            for (size_t i = 0; i < n; i++) {
                size_t reversed_i = 0;
                for (size_t j = 0; j < log_n; j++) {
                    if ((i >> j) & 1) {
                        reversed_i |= 1UL << (log_n - 1 - j);
                    }
                }
                if (i < reversed_i) {
                    std::swap(vec[i], vec[reversed_i]);
                }
            }
        }

        static inline __host__ uint64_t modInverseHERA(int64_t a, uint64_t m) {
            // a가 음수일 경우 양수로 변환
            a = (a % m + m) % m;

            int64_t m0 = m;
            int64_t y = 0, x = 1;

            if (m == 1) return 0;

            while (a > 1) {
                int64_t q = a / m0;
                int64_t t = m0;

                m0 = a % m0;
                a = t;
                t = y;

                y = x - q * y;
                x = t;
            }

            if (x < 0) x += m;

            return x;
        }

        // // Rebuilds plain_intt_natural_table_ so that plain_intt_natural_table_[k] = psi^{-k} (mod t)
        // // in NATURAL order for k = 0..(2N-1).
        // __host__ void recompute_plain_intt_table_natural(
        //     uint64_t psi, uint64_t t, int N)
        // {
        //     const int twoN = 2 * N;
        //     const uint64_t inv_psi = modInverse(psi, t);

        //     plain_intt_natural_table_.assign(twoN, 0ull);
        //     plain_intt_natural_table_[0] = 1ull % t;
        //     for (int k = 1; k < twoN; ++k) {
        //         plain_intt_natural_table_[k] = mul_mod_u128(
        //             plain_intt_natural_table_[k - 1], inv_psi, t);
        //     }
        // }
    __host__ void print_FV_S2C_Matrix(int print_size = 8)
    {
        // =========================================================================
        // ## Full DFT Matrix Visualization (Slot-to-Coefficient) ##
        //
        // This function reconstructs and prints the full DFT matrix M where
        // the element M[i, j] = ψ^(j * (2i + 1)).
        // =========================================================================
        const int N         = static_cast<int>(context_.n);
        const int twoN      = 2 * N;
        const uint64_t t   = plain_modulus_.value;
        const uint64_t psi = plain_psi_;

        std::cout << "\n## Printing DFT Matrix (Slot-to-Coefficient) ##\n"
                << "Parameters: N=" << N << ", t=" << t << ", psi=" << psi << "\n"
                << "Matrix M[i, j] = psi^(j * (2i+1))\n" << std::endl;

        if (print_size <= 0) return;

        // To avoid printing a massive matrix, we'll only show the top-left corner.
        const int effective_print_size = std::min(N, print_size);

        // --- Precompute powers of psi (same as in the original function) ---
        std::vector<uint64_t> psi_pow(twoN);
        psi_pow[0] = 1ULL;
        for (int k = 1; k < twoN; ++k) {
            psi_pow[k] = mul_mod_u128(psi_pow[k - 1], psi, t);
        }
        
        // --- Print Header ---
        std::cout << "M[i,j]|";
        for (int j = 0; j < effective_print_size; ++j) {
            std::cout << std::setw(8) << "j=" << j;
        }
        std::cout << "\n-------+";
        for (int j = 0; j < effective_print_size; ++j) {
            std::cout << "---------";
        }
        std::cout << std::endl;


        // --- Iterate through rows and columns to print the matrix ---
        for (int i = 0; i < effective_print_size; ++i) {
            std::cout << " i=" << std::setw(2) << i << " |";
            for (int j = 0; j < effective_print_size; ++j) {
                // This is the core logic from your original function,
                // calculating the exponent for the matrix element M[i, j].
                const uint64_t e = (static_cast<uint64_t>(j) * (2ULL * static_cast<uint64_t>(i) + 1ULL)) % static_cast<uint64_t>(twoN);
                
                // Look up the precomputed value
                uint64_t matrix_element = psi_pow[e];
                
                // Print the element with nice formatting
                std::cout << std::setw(8) << matrix_element << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    private:
        HEContext<Scheme::RTF>& context_;   
        HEEncoder<Scheme::RTF>& encoder_;  
        HEEncryptor<Scheme::RTF>& encryptor_; 
        HEOperator<Scheme::RTF>& operator_;
        Galoiskey<Scheme::RTF>& galois_key_;
        Relinkey<Scheme::RTF> relin_key_;
        scheme_type scheme_;

        int n;

        int n_power;

        int bsk_mod_count_;

        // New
        int Q_prime_size_;
        int Q_size_;
        int P_size_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;
        std::shared_ptr<DeviceVector<Data64>> last_q_modinv_;

        std::shared_ptr<DeviceVector<Modulus64>> base_Bsk_;
        std::shared_ptr<DeviceVector<Root64>> bsk_ntt_tables_; // check
        std::shared_ptr<DeviceVector<Root64>> bsk_intt_tables_; // check
        std::shared_ptr<DeviceVector<Ninverse64>> bsk_n_inverse_; // check

        Modulus64 m_tilde_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_Bsk_;
        std::shared_ptr<DeviceVector<Data64>>
            inv_punctured_prod_mod_base_array_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_m_tilde_;

        Data64 inv_prod_q_mod_m_tilde_;
        std::shared_ptr<DeviceVector<Data64>> inv_m_tilde_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data64>> prod_q_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data64>> inv_prod_q_mod_Bsk_;

        Modulus64 plain_modulus_;

        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_q_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_msk_;

        std::shared_ptr<DeviceVector<Data64>> inv_punctured_prod_mod_B_array_;
        Data64 inv_prod_B_mod_m_sk_;
        std::shared_ptr<DeviceVector<Data64>> prod_B_mod_q_;

        std::shared_ptr<DeviceVector<Modulus64>> q_Bsk_merge_modulus_;
        std::shared_ptr<DeviceVector<Root64>> q_Bsk_merge_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> q_Bsk_merge_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse64>> q_Bsk_n_inverse_;

        std::shared_ptr<DeviceVector<Data64>> half_p_;
        std::shared_ptr<DeviceVector<Data64>> half_mod_;

        Data64 upper_threshold_;
        std::shared_ptr<DeviceVector<Data64>> upper_halfincrement_;

        Data64 Q_mod_t_;
        std::shared_ptr<DeviceVector<Data64>> coeeff_div_plainmod_;

        /////////

        int d;
        int d_tilda;
        int r_prime;

        std::shared_ptr<DeviceVector<Modulus64>> B_prime_;
        std::shared_ptr<DeviceVector<Root64>> B_prime_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> B_prime_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse64>> B_prime_n_inverse_;

        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_D_to_B_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_B_to_D_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_D_to_B_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_B_to_D_;
        std::shared_ptr<DeviceVector<Data64>> prod_D_to_B_;
        std::shared_ptr<DeviceVector<Data64>> prod_B_to_D_;

        // Method2
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data64>> prod_D_to_Q_tilda_;

        std::shared_ptr<DeviceVector<int>> I_j_;
        std::shared_ptr<DeviceVector<int>> I_location_;
        std::shared_ptr<DeviceVector<int>> Sk_pair_;

        /////////

        std::vector<Modulus64> prime_vector_; // in CPU

        // Temp(to avoid allocation time)

        // new method
        DeviceVector<int> new_prime_locations_;
        DeviceVector<int> new_input_locations_;
        int* new_prime_locations;
        int* new_input_locations;

        // Encode params
        int slot_count_;
        std::shared_ptr<DeviceVector<Modulus64>>
            plain_modulus_pointer_; // we already have it
        std::shared_ptr<DeviceVector<Ninverse64>> n_plain_inverse_;
        std::shared_ptr<DeviceVector<Root64>> plain_intt_tables_;
        std::shared_ptr<DeviceVector<Data64>> encoding_location_;

        Data64 plain_psi_;
        Data64 plain_inv_psi;

        // hera
        int round_ = 2;
        size_t rc_vec_size_ = 0;

        std::vector<heongpu::DeviceVector<Data64>> linear_matrix_diagonals_;
        std::vector<std::vector<int>> linear_matrix_shifts_;

        std::vector<uint64_t> icVec;
        Plaintext<Scheme::RTF> icPt_;
        Ciphertext<Scheme::RTF> icCt_;

        std::vector<std::vector<uint64_t>> rcVec;

        std::vector<Plaintext<Scheme::RTF>> rcPt;

        std::vector<Ciphertext<Scheme::RTF>> rckCt; // rc * k

        // Slot To Coeff
        std::vector<std::vector<heongpu::DeviceVector<Data64>>> s2c_matrix_diagonals_;
        std::vector<std::vector<int>> s2c_matrix_shifts_;
        std::vector<std::vector<heongpu::DeviceVector<Data64>>> s2c_groups_no_swap_;
        std::vector<std::vector<heongpu::DeviceVector<Data64>>> s2c_groups_swap_;
        std::vector<std::vector<int>> s2c_shifts_mod_r_; // {baby_shifts, giant_shifts}
        Plaintext<Scheme::RTF> s2c_correction_pt_;
        std::vector<heongpu::Plaintext<heongpu::Scheme::RTF>> s2c_correction_pts_;
        std::vector<Root64> plain_intt_natural_table_; 
        std::vector<Root64> plain_ntt_natural_table_; 
        int s2c_bsgs_r_ = 0;
        int s2c_bsgs_m_ = 0;
    }; 
} // namespace heongpu

#endif //HEONGPU_RTF_HERA_H