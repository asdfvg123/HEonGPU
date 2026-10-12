#!/usr/bin/env python3
"""Plain Rubato (Par-128L) reference for checking the HEonGPU RtF server.

Mirrors plainRubato / SampleZqx of the authors' implementation
(KAIST-CryptLab/RtF-Transciphering), with AES-128-CTR as the XOF:
AES key = nonce (8 bytes) || counter (8 bytes, big-endian), IV = 0.
No AGN noise (that is what the server evaluates).

Usage: rubato_ref.py NBLOCKS   -> one line per block j (counter = j),
       60 keystream words for key[i] = i + 1 and an all-zero nonce.
Needs the `cryptography` package.
"""
import sys

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

N, L, ROUNDS, Q = 64, 60, 2, 0x1FC0001
Y8 = [5, 3, 4, 3, 6, 2, 1, 1]


class AesCtrXof:
    def __init__(self, seed):
        self.enc = Cipher(algorithms.AES(seed), modes.CTR(bytes(16))).encryptor()

    def read(self, n):
        return self.enc.update(bytes(n))


def sample_zqx(xof, q):
    bit_len = (q - 2).bit_length()
    byte_len = (bit_len + 7) // 8
    b = bit_len % 8 or 8
    while True:
        buf = bytearray(xof.read(byte_len))
        buf[-1] &= (1 << b) - 1
        res = int.from_bytes(buf, "little")
        if res < q:
            return res


def linear(st):
    buf = [0] * 64
    for row in range(8):
        for col in range(8):
            buf[row * 8 + col] = sum(Y8[k] * st[((row + k) % 8) * 8 + col] for k in range(8)) % Q
    for row in range(8):
        for col in range(8):
            st[row * 8 + col] = sum(Y8[k] * buf[row * 8 + (col + k) % 8] for k in range(8)) % Q


def feistel(st):
    buf = st[:]
    for i in range(1, len(st)):
        st[i] = (buf[i] + buf[i - 1] * buf[i - 1]) % Q


def keystream(key, nonce, counter):
    xof = AesCtrXof(nonce + counter.to_bytes(8, "big"))
    rk = [[sample_zqx(xof, Q) * key[i] % Q for i in range(N)] for _ in range(ROUNDS + 1)]
    st = [(i + 1 + rk[0][i]) % Q for i in range(N)]
    for r in range(1, ROUNDS):
        linear(st)
        feistel(st)
        st = [(st[i] + rk[r][i]) % Q for i in range(N)]
    linear(st)
    feistel(st)
    linear(st)
    st = [(st[i] + rk[ROUNDS][i]) % Q for i in range(N)]
    return st[:L]


if __name__ == "__main__":
    key = [(i + 1) % Q for i in range(N)]
    for j in range(int(sys.argv[1])):
        print(" ".join(map(str, keystream(key, bytes(8), j))))
