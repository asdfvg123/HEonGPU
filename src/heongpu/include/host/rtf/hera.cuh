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
                         HEOperator<Scheme::RTF>& op);


        HEHERA() = default;
        HEHERA(const HEHERA& copy) = default;
        HEHERA(HEHERA&& source) = default;
        HEHERA& operator=(const HEHERA& assign) = default;
        HEHERA& operator=(HEHERA&& assign) = default;


        __host__ void gen_stream_key(
            heongpu::DeviceVector<Data64>& nonce,
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
            HEEncoder<Scheme::RTF>& encoder,
            HEContext<Scheme::RTF>& context,
            Galoiskey<Scheme::RTF>& galois_key,
            const ExecutionOptions& options = ExecutionOptions()
        );
    

    private:
        HEContext<Scheme::RTF>& context_;   
        HEEncoder<Scheme::RTF>& encoder_;   
        HEOperator<Scheme::RTF>& operator_;
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
    }; 
} // namespace heongpu

#endif //HEONGPU_RTF_HERA_H