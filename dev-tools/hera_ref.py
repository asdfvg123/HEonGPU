#!/usr/bin/env python3
"""Plain HERA (Par-128a) reference for checking the HEonGPU RtF server.

Mirrors plainHera / SampleZqx of the authors' implementation
(KAIST-CryptLab/RtF-Transciphering, examples/ckks_fv/main.go), with
AES-128-CTR as the XOF instead of SHAKE256, keyed like the Rubato reference
(rubato_ref.py): AES key = nonce (8 bytes) || counter (8 bytes, big-endian),
IV = 0. State index 4*row + col; rc sampled round-major, 16 per round.

Usage: hera_ref.py NBLOCKS   -> one line per block j (counter = j),
       16 keystream words for key[i] = i + 1 and an all-zero nonce.
Needs the `cryptography` package (e.g. uv run --with cryptography).
"""
import sys

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

N, ROUNDS, Q = 16, 5, 0x1FC0001
C4 = [2, 3, 1, 1]  # circ(2, 3, 1, 1)


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
    # MixColumns: y[r][c] = sum_k C4[(k - r) % 4] x[k][c]
    buf = [sum(C4[(k - r) % 4] * st[4 * k + c] for k in range(4)) % Q
           for r in range(4) for c in range(4)]
    # MixRows: y[r][c] = sum_k C4[(k - c) % 4] x[r][k]
    st[:] = [sum(C4[(k - c) % 4] * buf[4 * r + k] for k in range(4)) % Q
             for r in range(4) for c in range(4)]


def cube(st):
    st[:] = [x * x % Q * x % Q for x in st]


def keystream(key, nonce, counter):
    xof = AesCtrXof(nonce + counter.to_bytes(8, "big"))
    rk = [[sample_zqx(xof, Q) * key[i] % Q for i in range(N)] for _ in range(ROUNDS + 1)]
    st = [(i + 1 + rk[0][i]) % Q for i in range(N)]
    for r in range(1, ROUNDS):
        linear(st)
        cube(st)
        st = [(st[i] + rk[r][i]) % Q for i in range(N)]
    linear(st)
    cube(st)
    linear(st)
    return [(st[i] + rk[ROUNDS][i]) % Q for i in range(N)]


if __name__ == "__main__":
    key = [(i + 1) % Q for i in range(N)]
    for j in range(int(sys.argv[1])):
        print(" ".join(map(str, keystream(key, bytes(8), j))))
