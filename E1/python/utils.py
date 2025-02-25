import numpy as np
import scipy.io


def dec2bits(dec: list[int], size=4) -> np.ndarray[np.uint8]:
    dec = np.array(dec, dtype=int)
    s = list(dec.shape)
    s[0] *= size
    bits: np.ndarray[np.uint8] = np.zeros(tuple(s), dtype=np.uint8)
    for i, x in enumerate(dec):
        for j in range(size):
            bits[i * size + j] = (x >> (size - 1 - j)) & 0x1
    return bits


def bits2dec(bits, size=4):
    bits = np.array(bits, dtype=np.uint8)
    dec = np.zeros(len(bits) // size, dtype=np.uint8)
    for i in range(len(bits) // size):
        for j in range(size):
            dec[i] |= bits[i * size + j] << (size-1 - j)
    return dec


if __name__ == "__main__":

    print(dec2bits([0x8, 0x4, 0x0, 0xA]))

    x = np.random.randint(0, 16, 64, dtype=np.uint8)
    assert (x == bits2dec(dec2bits(x))).all()
