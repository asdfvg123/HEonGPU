#include "heongpu.cuh"
#include "../example_util.h"
#include <vector>
#include <numeric>
#include <set>
#include <cmath>

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::RTF;

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    size_t poly_modulus_degree = 4096;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_default_values(1);
    int plain_modulus = 786433;
    context.set_plain_modulus(plain_modulus);
    context.generate();
    context.print_parameters();

    // Key Generation
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);
    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);
    
    std::cout << "Generating Galois keys for BSGS..." << std::endl;
    // Calculate the required BSGS parameters
    const int row_len = poly_modulus_degree >> 1;
    const size_t g2_unsigned = static_cast<size_t>(ceil(sqrt(row_len)));
    const int g2_signed = static_cast<int>(g2_unsigned);
    std::cout << "BSGS parameters: g2 = " << g2_signed << std::endl;
    
    // g2 as above
    std::set<int> required_shifts;
    required_shifts.insert(1); // for building baby steps by +1 repeatedly

    // 2. Collect all unique diagonal shifts from all matrices that will be used.
    std::vector<int> diags_M1 = {0, 2, -2};
    std::vector<int> diags_M2 = {0, 1};
    // collect all diagonals you'll use (unique)
    std::set<int> unique_diag_shifts;
    unique_diag_shifts.insert(diags_M1.begin(), diags_M1.end());
    unique_diag_shifts.insert(diags_M2.begin(), diags_M2.end());


    // for each diagonal shift s, we will rotate by i*g2 in the giant step
    for (int s : unique_diag_shifts) {
        int j = (s % g2_signed + g2_signed) % g2_signed;
        int i = (s - j) / g2_signed;
        required_shifts.insert(i * g2_signed);  // could be negative, that’s fine
    }

    // build galois keys for {1} and all distinct {i*g2}
    std::vector<int> all_required(required_shifts.begin(), required_shifts.end());
    heongpu::Galoiskey<Scheme> galois_key(context, all_required);
    keygen.generate_galois_key(galois_key, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    // Initial Vector
    std::vector<uint64_t> message(poly_modulus_degree, 0ULL);
    message[0] = 1ULL;
    message[1] = 2ULL;
    message[2] = 3ULL;
    message[3] = 4ULL;
    std::cout << "Initial vector:" << std::endl;
    display_matrix(message, 0);

    heongpu::Plaintext<Scheme> P1(context);
    encoder.encode(P1, message);
    heongpu::Ciphertext<Scheme> C1(context);
    encryptor.encrypt(C1, P1);

    /********************************************************************************
     * MATRIX DEFINITIONS
     ********************************************************************************/
    
    // --- Define Matrix 1 (M1) ---
    std::vector<uint64_t> m1_diag0(poly_modulus_degree, 3ULL);
    std::vector<uint64_t> m1_diag2(poly_modulus_degree, 1ULL);
    std::vector<uint64_t> m1_diag_minus2(poly_modulus_degree, 5ULL);

    // --- Define Matrix 2 (M2) ---
    std::vector<uint64_t> m2_diag0(poly_modulus_degree, 2ULL);
    std::vector<uint64_t> m2_diag1(poly_modulus_degree, 7ULL);


    /********************************************************************************
     * HOMOMORPHIC CALCULATION (BSGS)
     ********************************************************************************/
    std::cout << "\n--- Calculating with multiply_matrix_bsgs ---" << std::endl;
    
    // Prepare diagonals for M1
    heongpu::DeviceVector<Data64> concat_diags_M1(diags_M1.size() * poly_modulus_degree);
    {
        heongpu::Plaintext<Scheme> p_m1_d0(context), p_m1_d2(context), p_m1_dm2(context);
        encoder.encode(p_m1_d0, m1_diag0);
        encoder.encode(p_m1_d2, m1_diag2);
        encoder.encode(p_m1_dm2, m1_diag_minus2);
        cudaMemcpy(concat_diags_M1.data(), p_m1_d0.data(), poly_modulus_degree * sizeof(Data64), cudaMemcpyDeviceToDevice);
        cudaMemcpy(concat_diags_M1.data() + poly_modulus_degree, p_m1_d2.data(), poly_modulus_degree * sizeof(Data64), cudaMemcpyDeviceToDevice);
        cudaMemcpy(concat_diags_M1.data() + (2 * poly_modulus_degree), p_m1_dm2.data(), poly_modulus_degree * sizeof(Data64), cudaMemcpyDeviceToDevice);
    }

    // Prepare diagonals for M2
    heongpu::DeviceVector<Data64> concat_diags_M2(diags_M2.size() * poly_modulus_degree);
    {
        heongpu::Plaintext<Scheme> p_m2_d0(context), p_m2_d1(context);
        encoder.encode(p_m2_d0, m2_diag0);
        encoder.encode(p_m2_d1, m2_diag1);
        cudaMemcpy(concat_diags_M2.data(), p_m2_d0.data(), poly_modulus_degree * sizeof(Data64), cudaMemcpyDeviceToDevice);
        cudaMemcpy(concat_diags_M2.data() + poly_modulus_degree, p_m2_d1.data(), poly_modulus_degree * sizeof(Data64), cudaMemcpyDeviceToDevice);
    }
    
    // Structure for BSGS function
    std::vector<std::vector<heongpu::DeviceVector<Data64>>> matrix_groups_for_bsgs;
    matrix_groups_for_bsgs.push_back({std::move(concat_diags_M1)});
    matrix_groups_for_bsgs.push_back({std::move(concat_diags_M2)});

    std::vector<std::vector<int>> shifts_for_bsgs = {diags_M1, diags_M2};
    
    heongpu::Ciphertext<Scheme> C_result_bsgs = operators.multiply_matrix_bsgs(
        C1, matrix_groups_for_bsgs, shifts_for_bsgs, galois_key);

    std::vector<uint64_t> vec_result_bsgs;
    {
        heongpu::Plaintext<Scheme> P_result(context);
        decryptor.decrypt(P_result, C_result_bsgs);
        encoder.decode(vec_result_bsgs, P_result);
    }
    std::cout << "Result from BSGS function:" << std::endl;
    for(int i = 0; i < 128; ++i) {
        std::cout << vec_result_bsgs[i] << " ";
        if (i % 16 == 15) std::cout << "\n";
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}