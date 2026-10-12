# HEonGPU Basic Examples

The basic example directory in the HEonGPU repository contains sample programs that demonstrate how to utilize various features of the HEonGPU library. These examples are designed to help users understand and experiment with the library's functionalities, from basic encryption and decryption to more advanced homomorphic operations.

## Overview

- [1_basic_bfv.cu](1_basic_bfv.cu): Demonstrates basic arithmetic operations on encrypted integers using the BFV scheme, including encryption, evaluation, and decryption.

- [2_basic_ckks.cu](2_basic_ckks.cu): Introduces the CKKS scheme for approximate arithmetic on real and complex numbers, showing encoding, encryption, evaluation, and decryption.

- [3_switchkey_methods_bfv.cu](3_switchkey_methods_bfv.cu): Explores different key switching methods in the BFV scheme, illustrating how to manage and switch between keys during encrypted computations.

- [4_switchkey_methods_ckks.cu](4_switchkey_methods_ckks.cu): Examines key switching methods within the CKKS scheme, demonstrating the process of changing keys in homomorphic operations.

- [5_default_stream_usage.cu](5_default_stream_usage.cu): Shows how to perform encrypted computations using the default CUDA stream, highlighting the integration of GPU processing in homomorphic encryption.

- [6_multi_stream_usage_way1.cu](6_multi_stream_usage_way1.cu): Uses OpenMP to assign a CUDA stream to each CPU thread, demonstrating a multi-stream approach for executing parallel encrypted operations efficiently.

- [7_multi_stream_usage_way2.cu](7_multi_stream_usage_way2.cu): Implements a basic encrypted operation using multiple CUDA streams, where each stream is invoked by a single CPU thread. This example highlights an alternative method to leverage multi-stream functionality for concurrent GPU processing.

- [8_basic_bfv_logic.cu](8_basic_bfv_logic.cu): Demonstrates basic logic operations on encrypted binarys using the BFV scheme, including encryption, evaluation, and decryption.

- [9_basic_ckks_logic.cu](9_basic_ckks_logic.cu): Demonstrates basic logic operations on encrypted binarys using the CKKS scheme, including encryption, evaluation, and decryption.

### RtF / Rubato (Scheme::RTF)

Each of these prints `[PASS]` / `[FAIL]` checks against a plaintext reference and exits non-zero on failure.

- [13_basic_rtf_matmul.cu](13_basic_rtf_matmul.cu): Diagonal matrix-vector product on FV slots with `multiply_matrix`.

- [15_rtf_fv_slot2coeff.cu](15_rtf_fv_slot2coeff.cu): SlotToCoeff_FV (`HEHERA::S2C_FV`) on a random vector; `[logN] [keyswitching method 1|2]`, default METHOD_II.

- [16_rtf_matmul_bsgs.cu](16_rtf_matmul_bsgs.cu): Baby-step giant-step product (`multiply_matrix_bsgs`) for a sparse matrix, with its calling convention (groups, pre-rotated diagonals, keys).

- [18_rtf_moddown.cu](18_rtf_moddown.cu): FV mod-down to q0 (`HEHERA::moddown_FV_inplace`), decrypted on the host with a known secret key and checked coefficient by coefficient.

- [19_rtf_rubato.cu](19_rtf_rubato.cu): The full RtF transciphering flow with Rubato Par-128L: homomorphic keystream vs host reference, SlotToCoeff_FV, HalfBoot output vs client messages, and server timing.
