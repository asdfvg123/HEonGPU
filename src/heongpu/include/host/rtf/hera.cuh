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

#include "ckks/context.cuh"
#include "ckks/ciphertext.cuh"
#include "ckks/operator.cuh"

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
                        HEContext<Scheme::CKKS>& contextckks,
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

        __host__ void precompute(
            Ciphertext<Scheme::RTF>& key, 
            const ExecutionOptions& options = ExecutionOptions()
        );

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
            Ciphertext<Scheme::RTF>& input,
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ Plaintext<Scheme::RTF> vec2poly(
            const std::vector<uint64_t>& input,
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ Ciphertext<Scheme::RTF> eval_decrypt(
            Ciphertext<Scheme::RTF>& keyCt,
            Plaintext<Scheme::RTF>& SKEpt,
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ Ciphertext<Scheme::CKKS> transcipher_bfv2ckks(
            Ciphertext<Scheme::RTF>& ct_bfv,
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ void gen_FV_moddown_params(
            const ExecutionOptions& options = ExecutionOptions()
        );

        __host__ void moddown_FV_inplace(
            Ciphertext<Scheme::RTF>& input,
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

        __host__ Data64 get_psi() const {
            return plain_psi_;
        }

        __host__ int get_n_power() const {
            return n_power;
        }

        static inline __host__ __device__ uint64_t mul_mod_u128(uint64_t a, uint64_t b, uint64_t m) {
            return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) % m);
        }

        int64_t extendedGCD_internal(int64_t a, int64_t b, int64_t& x, int64_t& y)
        {
            if (a == 0)
            {
                x = 0;
                y = 1;
                return b;
            }

            int64_t x1, y1;
            int64_t gcd = extendedGCD_internal(b % a, a, x1, y1);

            // ★★★ 이 계산은 x, y가 음수가 될 수 있어 반드시 signed 타입으로 수행되어야 합니다 ★★★
            x = y1 - (b / a) * x1;
            y = x1;

            return gcd;
        }

        // 최종적으로 안전하게 사용할 수 있는 메인 함수
        Data64 modInverseHERA(Data64 a, Data64 m)
        {
            int64_t x, y;
            // 입력값을 안전하게 signed 타입으로 변환하여 계산
            int64_t a_signed = static_cast<int64_t>(a);
            int64_t m_signed = static_cast<int64_t>(m);

            int64_t gcd = extendedGCD_internal(a_signed, m_signed, x, y);

            if (gcd != 1)
            {
                // 역원이 존재하지 않음
                return 0;
            }
            else
            {
                // 결과가 음수일 경우를 대비해 양수로 변환
                // (x % m + m) % m 은 C++에서 음수 나머지를 처리하는 표준적인 방법입니다.
                int64_t result_signed = (x % m_signed + m_signed) % m_signed;
                return static_cast<Data64>(result_signed);
            }
        }


        static void debug_print_ct_coeffs_bfv(
            const heongpu::Ciphertext<heongpu::Scheme::RTF>& ct,
            const DeviceVector<Modulus64>& device_moduli, // 보통 context의 *modulus_
            cudaStream_t stream)
        {
            const int N          = ct.ring_size();
            const int poly_count = ct.size();
            const int L          = ct.coeff_modulus_count();

            // 1) 남은 모듈러스 L개를 호스트로 복사
            std::vector<Modulus64> host_modulus(L);
            if (L > 0) {
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    host_modulus.data(), device_moduli.data(),
                    L * sizeof(Modulus64),
                    cudaMemcpyDeviceToHost, stream));
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // 2) 계수 전체를 호스트 버퍼로 복사 (friend 접근으로 data 포인터 취득)
            const size_t words = static_cast<size_t>(poly_count) * (L > 0 ? L : 1) * N;
            const size_t bytes = words * sizeof(Data64);

            std::vector<Data64> host(words);
            if (ct.storage_type_ == storage_type::DEVICE) {
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    host.data(),
                    ct.device_locations_.data(),   // friend access
                    bytes,
                    cudaMemcpyDeviceToHost, stream));
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
            } else {
                // HOST에 있으면 그대로 복사
                std::memcpy(host.data(), ct.host_locations_.data(), bytes);
            }

            // 3) 너무 길어지는 출력 방지: 앞/뒤 일부만
            constexpr int MAX_PRINT = 32;              // 필요 시 조절
            const bool truncate = (N > MAX_PRINT);
            const int head = truncate ? (MAX_PRINT / 2) : N;
            const int tail = truncate ? (MAX_PRINT - head) : 0;

            const int effL = (L > 0 ? L : 1);
            for (int p = 0; p < poly_count; ++p) {
                for (int ell = 0; ell < effL; ++ell) {
                    if (L > 0) {
                        printf("=== poly %d, q[%d] = %llu ===\n",
                            p, ell, (unsigned long long)host_modulus[ell].value);
                    } else {
                        printf("=== poly %d, (L=0?) ===\n", p);
                    }
                    const Data64* base = host.data()
                        + (static_cast<size_t>(p) * effL + ell) * N;

                    if (!truncate) {
                        for (int i = 0; i < N; ++i) {
                            printf("%llu%c",
                                (unsigned long long)base[i],
                                (i + 1 == N ? '\n' : ' '));
                        }
                    } else {
                        for (int i = 0; i < head; ++i)
                            printf("%llu ", (unsigned long long)base[i]);
                        printf("... ");
                        for (int i = N - tail; i < N; ++i) {
                            printf("%llu%c", (unsigned long long)base[i],
                                (i + 1 == N ? '\n' : ' '));
                        }
                    }
                }
            }
        }
        inline void print_coeffs_block_inline(
            const char* label,
            const uint64_t* base,    // 블록 시작 포인터 (host)
            int N,
            int max_print = 32)
        {
            if (label && *label) std::printf("%s", label);

            const bool truncate = (N > max_print);
            const int head = truncate ? (max_print / 2) : N;
            const int tail = truncate ? (max_print - head) : 0;

            if (!truncate) {
                for (int i = 0; i < N; ++i) {
                    std::printf("%llu%c",
                                (unsigned long long)base[i],
                                (i + 1 == N ? '\n' : ' '));
                }
            } else {
                for (int i = 0; i < head; ++i)
                    std::printf("%llu ", (unsigned long long)base[i]);
                std::printf("... ");
                for (int i = N - tail; i < N; ++i) {
                    std::printf("%llu%c",
                                (unsigned long long)base[i],
                                (i + 1 == N ? '\n' : ' '));
                }
            }
        }

        // host 버퍼 전체에서 (poly, modulus, coeff) 순서로 출력
        inline void print_ct_coeffs_inline(
            const char* tag,                   // 예: "[BFV after moddown]" 또는 "[CKKS after copy]"
            const uint64_t* host,             // host.data()
            int poly_count,                   // 보통 2 (c0, c1)
            int L,                            // 남은 모듈러스 개수
            int N,                            // 링 크기
            const uint64_t* moduli_values,    // nullptr 허용. 있으면 q값도 출력
            int max_print = 32)
        {
            const int effL = (L > 0 ? L : 1);

            for (int p = 0; p < poly_count; ++p) {
                for (int ell = 0; ell < effL; ++ell) {
                    char header[256];
                    if (moduli_values && L > 0) {
                        std::snprintf(header, sizeof(header),
                                    "%s poly %d, q[%d]=%llu\n",
                                    (tag ? tag : ""), p, ell,
                                    (unsigned long long)moduli_values[ell]);
                    } else {
                        std::snprintf(header, sizeof(header),
                                    "%s poly %d, ell=%d\n",
                                    (tag ? tag : ""), p, ell);
                    }

                    const uint64_t* base =
                        host + (static_cast<size_t>(p) * effL + ell) * N;

                    print_coeffs_block_inline(header, base, N, max_print);
                }
            }
        }
    private:
        HEContext<Scheme::RTF>& contextbfv_;   
        HEContext<Scheme::CKKS>& contextckks_;

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
        int round_ = 5;
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
        std::vector<uint64_t> psi_pow_;
        const uint64_t generator_ = 3;
        bool is_S2C_initialized_ = false;

        //FV moddown
        std::shared_ptr<DeviceVector<Data64>> qhalf_;
        std::shared_ptr<DeviceVector<Data64>> invq_;
        

        //FV
        Data64 Delta_FV_;
        const double message_ratio_ = 1<<15;
        double messageScaling_;


    }; 
} // namespace heongpu

#endif //HEONGPU_RTF_HERA_H