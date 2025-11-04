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

    size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_default_values(1);
    int plain_modulus = 65537;
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
    const int N = poly_modulus_degree;
    const int g2_signed = static_cast<int>(ceil(sqrt(N/2)));
    const int N_div_2 = N / 2;


    std::cout << "BSGS parameters: g2 = " << g2_signed << std::endl;
        // --- Correct Key Generation Logic for multiply_matrix_bsgs ---
    std::cout << "Generating Galois keys for BSGS..." << std::endl;


    // Define all unique diagonal shifts from all matrices you will use
    std::vector<int> diags_M1 = {0, 2, -2};
    std::vector<int> diags_M2 = {0, 1};
    std::vector<int> diags_M3 = {0, 66};
    std::cout << "N/2 " << N/2 << std::endl;

    std::set<int> unique_diag_shifts;
    unique_diag_shifts.insert(diags_M1.begin(), diags_M1.end());
    unique_diag_shifts.insert(diags_M2.begin(), diags_M2.end());
    unique_diag_shifts.insert(diags_M3.begin(), diags_M3.end());

    std::set<int> required_row_rotations;
    bool needs_conjugation_key = false;

    // 1. Determine keys needed for baby steps
    // The function directly rotates by `j` for j in [1, g2-1].
    for (int j = 1; j < g2_signed; ++j) {
        required_row_rotations.insert(j);
    }

    // 2. Determine the set of unique giant rotation amounts that will be performed
    std::set<int> unique_giant_rotations;
    for (int s_orig : unique_diag_shifts) {
        // Mimic the exact normalization from the function's accumulation loop
        int effective_s = s_orig % N;
        if (effective_s > N_div_2) {
            effective_s -= N;
        } else if (effective_s < -N_div_2) {
            effective_s += N;
        }

        // Mimic the BSGS decomposition to find the giant step index 'i'
        int j = (effective_s % g2_signed + g2_signed) % g2_signed;
        int i = (effective_s - j) / g2_signed;
        
        int giant_rot = i * g2_signed;
        if (giant_rot != 0) {
            unique_giant_rotations.insert(giant_rot);
        }
    }

    // 3. For each unique giant rotation, determine the final key needed
    for (int rot : unique_giant_rotations) {
        // Mimic the exact normalization from the function's final combination loop
        long long effective_rot = static_cast<long long>(rot) % N;
        if (effective_rot < 0) {
            effective_rot += N;
        }

        if (N % 2 == 0 && effective_rot == N_div_2) {
            // This rotation requires rotate_columns, so we need the conjugation key
            needs_conjugation_key = true;
        } else {
            // This rotation uses rotate_rows_inplace, find its minimal representation
            if (effective_rot > N_div_2) {
                effective_rot -= N;
            }
            if (effective_rot != 0) {
                required_row_rotations.insert(static_cast<int>(effective_rot));
            }
        }
    }

    // 4. Combine all keys into the final list for the key generator
    std::vector<int> all_required_keys(required_row_rotations.begin(), required_row_rotations.end());
    if (needs_conjugation_key) {
        // Add the special value (-1 is common) to request the conjugation key
        all_required_keys.push_back(-1); 
    }
    
    heongpu::Galoiskey<Scheme> galois_key(context, all_required_keys);
    keygen.generate_galois_key(galois_key, secret_key);
    // --- END OF KEY GENERATION LOGIC ---

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    // Initial Vector
    std::vector<uint64_t> message(N, 1ULL);
    for(int u = 0; u < N; u++)
        message[u] = u % 17;

    // message[0] = 1ULL; message[1] = 2ULL; message[2] = 3ULL; message[3] = 4ULL;
    std::cout << "Initial vector:" << std::endl;
    display_matrix(message, 0);

    heongpu::Plaintext<Scheme> P1(context);
    encoder.encode(P1, message);
    heongpu::Ciphertext<Scheme> C1(context);
    encryptor.encrypt(C1, P1);

    // Matrix Definitions
    std::vector<uint64_t> m3_diag0(N, 4ULL);
    std::vector<uint64_t> m3_diag_N_div_4(N, 2ULL);
    for(int u = 0; u < N; u++)
        m3_diag_N_div_4[u] = (u-g2_signed) % 19;

    // std::vector<uint64_t> m3_diag_N_div_2(N, 6ULL);


    /********************************************************************************
     * NEW TEST CASE: M3 (diagonals 0, N/2) * V
     ********************************************************************************/
    std::cout << "\n--- New Test Case: Calculating with M3 (diagonals 0, N/2) using BSGS ---" << std::endl;

    heongpu::DeviceVector<Data64> concat_diags_M3(diags_M3.size() * N);
    {
        heongpu::Plaintext<Scheme> p_m3_d0(context), p_m3_dNdiv2(context), p_m3_dNdiv4(context);
        encoder.encode(p_m3_d0, m3_diag0);
        encoder.encode(p_m3_dNdiv4, m3_diag_N_div_4);
        // encoder.encode(p_m3_dNdiv2, m3_diag_N_div_2);

        cudaMemcpy(concat_diags_M3.data(), p_m3_d0.data(), N * sizeof(Data64), cudaMemcpyDeviceToDevice);
        cudaMemcpy(concat_diags_M3.data() + N, p_m3_dNdiv4.data(), N * sizeof(Data64), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(concat_diags_M3.data() + N*2, p_m3_dNdiv2.data(), N * sizeof(Data64), cudaMemcpyDeviceToDevice);

    }

    std::vector<std::vector<heongpu::DeviceVector<Data64>>> matrix_groups_for_bsgs_M3;
    matrix_groups_for_bsgs_M3.push_back({std::move(concat_diags_M3)});
    std::vector<std::vector<int>> shifts_for_bsgs_M3 = {diags_M3};

    heongpu::Ciphertext<Scheme> C_result_bsgs_M3 = operators.multiply_matrix_bsgs(
        C1, matrix_groups_for_bsgs_M3, shifts_for_bsgs_M3, galois_key, galois_key);

    std::vector<uint64_t> vec_result_bsgs_M3;
    {
        heongpu::Plaintext<Scheme> P_result(context);
        decryptor.decrypt(P_result, C_result_bsgs_M3);
        encoder.decode(vec_result_bsgs_M3, P_result);
    }
    std::cout << "Result from BSGS function with M3 (M3*v):" << std::endl;
    for(int i = 0; i < 128; ++i) {
        std::cout << vec_result_bsgs_M3[i] << " ";
        if (i % 16 == 15) std::cout << "\n";
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}