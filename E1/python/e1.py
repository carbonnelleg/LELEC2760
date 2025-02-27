import numpy as np
from utils import dec2bits, bits2dec

sbox: np.ndarray[np.uint8] = np.load("sbox.npz")["sbox"]
sboxinv: np.ndarray[np.uint8] = np.load("sbox.npz")["sboxinv"]

player: np.ndarray[np.uint8] = np.load("player.npz")["player"]
playerinv: np.ndarray[np.uint8] = np.load("player.npz")["playerinv"]


def encrypt(pt, k, nrounds):
    """
    Encrypts a plaintext using the scheme presented in the exercise session. 
    Returns the corresponding ciphertext. 

    Parameters
    ----------
    pt : np.ndarray[np.uint8]
        plaintext
    k : np.ndarray[np.uint8]
        key
    nrounds : int
        number of rounds

    Returns
    -------
    ct : np.ndarray[np.uint8]
        ciphertext

    Additional Info
    ---------------
    This function does not support multiple axes object.

    .. code-block::

        # Don't use this
        pts = np.random.randint(0, 16, (n, 8), dtype=np.uint8)
        cts = encrypt(pts, k, nrounds, axis=0)

        # Instead use this
        pts = np.random.randint(0, 16, (n, 8), dtype=np.uint8)
        cts = np.zeros((n, 8), dtype=np.uint8)
        for i, ct in enumerate(cts):
            cts[i] = encrypt(pts[i], k, nrounds)
    """

    new_input: np.ndarray[np.uint8] = pt

    for r in range(nrounds):
        xored_out = new_input ^ k
        subst_out = sbox[xored_out]
        permu_out = bits2dec(dec2bits(subst_out)[player])
        new_input = permu_out

    return new_input ^ k


def decrypt(ct, k, nrounds):
    """
    Decrypts a ciphertext using the scheme presented in the exercise session. 
    Returns the corresponding plaintext.

    Parameters
    ----------
    ct : np.ndarray[np.uint8]
        ciphertext
    k : np.ndarray[np.uint8]
        key
    nrounds : int
        number of rounds

    Returns
    -------
    pt : np.ndarray[np.uint8]
        plaintext
    """

    xored_out: np.ndarray[np.uint8] = ct ^ k

    for r in range(nrounds):
        permu_out = bits2dec(dec2bits(xored_out)[playerinv])
        subst_out = sboxinv[permu_out]
        xored_out = subst_out ^ k

    return xored_out


if __name__ == "__main__":
    print("LELEC2760: TP 1 - Ex 1")

    print(" -> Run known answer tests")
    key = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
    plain = np.array([0, 7, 0, 2, 2, 0, 1, 2], dtype=np.uint8)
    ctexts = np.array([
        [12, 11, 9, 1, 2, 13, 7, 5],
        [6, 3, 9, 13, 6, 9, 7, 1],
        [14, 15, 1, 4, 11, 9, 3, 4],
    ], dtype=np.uint8)

    for r in range(1, 4):
        assert (ctexts[r - 1] == encrypt(plain, key, r)).all()
        assert (plain == decrypt(ctexts[r - 1], key, r)).all()

    print(" -> Run random tests")

    for _ in range(100):
        k = np.random.randint(0, 16, 8)
        pt = np.random.randint(0, 16, 8)
        assert (decrypt(encrypt(pt, k, 3), k, 3) == pt).all()
