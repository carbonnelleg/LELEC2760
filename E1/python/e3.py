import numpy as np
from e1 import encrypt
from utils import dec2bits


def gen_data(k, rounds, n):
    """
        Generate plaintext - ciphertext pairs for different attacks
    """
    pts = np.random.randint(0, 16, (n, 8), dtype=np.uint8)
    cts = np.zeros((n, 8), dtype=np.uint8)
    for i, ct in enumerate(cts):
        cts[i] = encrypt(pts[i], k, rounds)

    return pts, cts


def experimental_bias(pts, pts_msk, cts, cts_msk):
    """
    Computes the experimental bias given n pairs of plaintexts/ciphertexts.

    Parameters
    ----------
    pts : np.ndarray[np.uint8]
        plaintexts, decimal form, shape = (n * 8)
    pts_msk : np.ndarray[np.uint8]
        input mask, decimal form, shape = (8,)
    cts : np.ndarray[np.uint8]
        ciphertext, decimal form, shape = (n * 8)
    cts_msk : np.ndarray[uint]
        output mask, decimal form, shape = (8,)

    Returns
    -------
    Experimental bias

    Additional Info
    ---------------
    Exerimental bias is the deviation from 0.5 of the result of the boolean
    equation of the linear approximation averaged over n plaintext/ciphertext pairs.
    The key is fixed, all input plaintexts encrypt to the corresponding input
    ciphertexts with the same encryption key.
    """

    n = len(pts)
    count = 0
    for pt, ct in zip(pts, cts):
        bool_eq = dec2bits([*pts_msk, *cts_msk]) & dec2bits([*pt, *ct])
        count += np.bitwise_xor.reduce(bool_eq)

    return count/n - 1/2


if __name__ == "__main__":

    pts_msk = np.array(
        [0x0, 0x0, 0x0, 0x7, 0x0, 0x0, 0x0, 0x0], dtype=np.uint8)
    cts_msk = np.array(
        [0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x2, 0x0], dtype=np.uint8)
    k = np.random.randint(0, 16, 8, dtype=np.uint8)
    n = 1000
    rounds = 2

    pts, cts = gen_data(k, 2, n)
    print(f'Experimental bias: {
        experimental_bias(pts, pts_msk, cts, cts_msk):.3f}')
    k_bits = dec2bits(k)
    print(f'Sum of K: k[10] ^ k[11] ^ k[13] ^ k[14] ^ k[15] ^ k[26]\n'
          f'Sum of K = {
              k_bits[10] ^ k_bits[11] ^ k_bits[13] ^ k_bits[14] ^ k_bits[15] ^
              k_bits[26]}')
