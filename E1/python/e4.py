import numpy as np
from e1 import encrypt, decrypt
from utils import dec2bits, bits2dec
from e3 import gen_data, experimental_bias
from tqdm import tqdm
from scipy.special import erfinv

import os
import_path: str = os.path.dirname(os.path.realpath(__file__)) 

sboxinv: np.ndarray[np.uint8] = np.load(import_path+"/sbox.npz")["sboxinv"]
playerinv: np.ndarray[np.uint8] = np.load(import_path+"/player.npz")["playerinv"]


def attack(
    pts, pts_msk, cts, cts_msk, key_target,
    init_key_guess=np.zeros(8, dtype=np.uint8)
) -> np.ndarray[float]:
    """
    Performs an attack on the encryption scheme by computing the experimental bias
    of the linear approximation for all possible subkeys.

    Parameters
    ----------
    pts : np.ndarray[np.uint8]
        plaintexts, decimal form, shape = (n, 8)
    pts_msk : np.ndarray[np.uint8]
        input mask, decimal form, shape = (8,)
    cts : np.ndarray[np.uint8]
        ciphertext, decimal form, shape = (n, 8)
    cts_msk : np.ndarray[np.uint8]
        output mask, decimal form, shape = (8,)
    key_target : np.ndarray[np.uint8]
        targeted key bits mask, decimal form, shape = (8,)

        in binary format, M bits are 1 and 32 - M bits are 0
    init_key_guess : np.ndarray[np.uint8], optional
        initial (total) key guess. The default is np.zeros(8, dtype=np.uint8).

    Returns
    -------
    bias : np.ndarray[float]
        array of experimental bias with index corresponding to subkey, shape = (2**M,)
    """
    # key_target is converted in binary format and guess_size ( = M) is calculated
    key_bits_target = dec2bits(key_target)
    guess_size = int(np.sum(key_bits_target))
    key_bits_locs = np.where(key_bits_target == 1)[0]

    # tested_key will be changing 2**M times
    tested_key = dec2bits(init_key_guess)
    bias = np.zeros(2**guess_size)

    # subst_outs will contain the reversed last round of every ciphertext using tested_key
    subst_outs = np.zeros_like(cts, dtype=np.uint8)

    for subkey_guess in tqdm(range(2**guess_size)):
        tested_key[key_bits_locs] = dec2bits([subkey_guess], size=guess_size)
        # Backtrack the last round of the encryption scheme for every ciphertext
        for i, ct in enumerate(cts): 
            xored_out = ct ^ bits2dec(tested_key)
            permu_out = bits2dec(dec2bits(xored_out)[playerinv])
            subst_outs[i] = sboxinv[permu_out]

        # Compute the experimental bias for the tested_key (the subkey_guess), using the
        # last round of the encryption scheme
        bias[subkey_guess] = experimental_bias(
            pts, pts_msk, subst_outs, cts_msk)

    return bias


def exhaustive_search(pt, ct, nrounds, final_key_target, init_key_guess):
    """
    Performs an exhaustive search on the remaining subkey bits and returns the
    key if it is found, else None

    Parameters
    ----------
    pt : np.ndarray[np.uint8]
        plaintext
    ct : np.ndarray[np.uint8]
        ciphertext
    nrounds : int
        number of rounds
    final_key_target : np.ndarray[np.uint8]
        targeted key bits mask, decimal form, shape = (8,)
    init_key_guess : np.ndarray[np.uint8]
        initial (total) key guess

    Returns
    -------
    final_key_guess : np.ndarray[np.uint8] | None
        final key where it is verified that the key encrypts pt into ct. If no key
        is found, returns None

    """
    key_bits_target = dec2bits(final_key_target)
    guess_size = int(np.sum(key_bits_target))
    key_bits_locs = np.where(key_bits_target == 1)[0]
    k = dec2bits(init_key_guess)

    for key_guess in tqdm(range(2**guess_size)):
        k[key_bits_locs] = dec2bits([key_guess], size=guess_size)
        final_key_guess = bits2dec(k)
        # Check if the key encrypts pt into ct
        if (encrypt(pt, final_key_guess, nrounds) == ct).all():
            return final_key_guess
        else:
            pass


def calc_n(eps, M, alpha_M=0.05):
    r"""
    Calculates the amount of plaintext/ciphertext pairs N to recover the subkey
    bits given the estimated bias eps, the tolerable error probability alpha_M
    and the subkey size M.

    Parameters
    ----------
    eps : float
        (estimated) bias of the linear approximation, sign does not matter
    M : int
        guess size, amount of bits targeted for the target subkey recovery
    alpha_M : float, optional
        target error probability of true partial key not having the largest
        experimental bias amongst the 2**M tested subkey bits. Default is 0.05

    Returns
    -------
    N : int
        amount of plaintext/ciphertext pairs to be tested to recover subkey bits

    Additional Info
    ---------------
    The error probability for :math:`2^M` tries :math:`\alpha_M` relates to
    the error probability for 1 try :math:`\alpha` as follows:

    .. math::
        \alpha_M &= 1 - (1 - \alpha)^{2^M}

        \alpha &= 1 - (1-\alpha_M)^{2^{-M}}

    :math:`\Phi` denotes the cdf of a normal distribution with zero mean and
    variance equal to 1.

    Following the course notes, the number of samples required to distinguish
    these 2 distributions with error probability :math:`\alpha` is given by:

    .. math::
        n &= \frac{d}{2\varepsilon^2}

        \alpha &= \Phi\left(\frac{-\sqrt{d}}2\right)

        d &= 4\Phi^{-2}(\alpha)

        &= 8 \textrm{erf}^{-2}(2\alpha-1)

        n &= \frac{4\textrm{erf}^{-2}(2\alpha-1)}{\varepsilon^2}
    """
    # Calculate the error probability for 1 try
    alpha = 1 - (1-alpha_M)**(2**(-M))
    # Calculate the distance between the 2 distributions
    d = 8*erfinv(2*alpha-1)**2
    # Calculate the number of samples required
    n = round(d/(2*eps**2))
    return n


"""
_______________________________________________________________________________
A few plaintexts masks and ciphertexts masks and their corresponding estimated
bias and key targets are stored here below.
The main_eval function performs a partial subkey bits recovery using one of the
linear approximation (simply call main_eval(i)).
main_attack performs the full key recovery by combining multiple attacks on a
linear approximations and a final exhaustive search.
_______________________________________________________________________________
"""


pts_msks = np.array([[0x0, 0x0, 0xB, 0x0, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0xA, 0x0, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0xB, 0x0, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0xA, 0x0, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0xB, 0x0, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0xA, 0x0, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0x0, 0x7, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0x0, 0xA, 0x0, 0x0, 0x0, 0x0],
                     [0xC, 0x0, 0x0, 0xC, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0xB, 0x0, 0xB, 0x0, 0x0, 0x0, 0x0]], dtype=np.uint8)
cts_msks = np.array([[0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0x8, 0x0, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0x0, 0x8, 0x0, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0x0, 0x0, 0x8, 0x0, 0x0, 0x0],
                     [0x0, 0x0, 0x0, 0x0, 0x0, 0x8, 0x0, 0x0],
                     [0x0, 0x0, 0x2, 0x0, 0x0, 0x0, 0x2, 0x0],
                     [0x0, 0x0, 0x0, 0x8, 0x0, 0x0, 0x0, 0x8],
                     [0x8, 0x0, 0x0, 0x0, 0x8, 0x0, 0x0, 0x0],
                     [0x0, 0x2, 0x0, 0x0, 0x0, 0x2, 0x0, 0x0]], dtype=np.uint8)
key_targets = np.array([[0x8, 0x0, 0x8, 0x0, 0x8, 0x0, 0x8, 0x0],
                        [0x4, 0x0, 0x4, 0x0, 0x4, 0x0, 0x4, 0x0],
                        [0x2, 0x0, 0x2, 0x0, 0x2, 0x0, 0x2, 0x0],
                        [0x1, 0x0, 0x1, 0x0, 0x1, 0x0, 0x1, 0x0],
                        [0x0, 0x8, 0x0, 0x8, 0x0, 0x8, 0x0, 0x8],
                        [0x0, 0x4, 0x0, 0x4, 0x0, 0x4, 0x0, 0x4],
                        [0x2, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2],
                        [0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1],
                        [0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8],
                        [0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4]], dtype=np.uint8)
eps = [1/16,
       1/16,
       1/16,
       1/16,
       1/16,
       1/16,
       1/8,
       1/8,
       1/16,
       1/16]

global bias
bias = []
global biases
biases = []

global final_key_guess


def main_eval(att, alpha_M=.05):
    key_bits_target = dec2bits(key_targets[att])
    key_bits_locs = np.where(key_bits_target == 1)[0]
    guess_size = int(np.sum(key_bits_target))
    n = calc_n(eps[att], guess_size, alpha_M)

    key = np.random.randint(0, 16, 8, dtype=np.uint8)
    key_bits = dec2bits(key)
    pts, cts = gen_data(key, 3, n)

    global bias
    bias = attack(pts, pts_msks[att], cts, cts_msks[att], key_targets[att])
    key_guess = np.argmax(np.abs(bias))

    real_k = bits2dec(key_bits[key_bits_locs], size=guess_size)
    print()
    print("real key word:", real_k[0])
    print("key guess:", key_guess)
    print("location:", (2**guess_size) -
          np.where(np.argsort(np.abs(bias)) == real_k)[0][0])


def main_attack():
    partial_attacks = 5
    init_key_guess = np.zeros(8, dtype=np.uint8)
    all_pts = np.load(import_path+"/pts_cts_pairs.npz")["pts"]
    all_cts = np.load(import_path+"/pts_cts_pairs.npz")["cts"]
    # Try multiple different linear approximations to recover the key
    for att in range(partial_attacks):
        print(f"attack {att+1}/{partial_attacks}")

        # Calculate the guess size and the number of plaintext/ciphertext pairs
        key_bits_target = dec2bits(key_targets[att])
        key_bits_locs = np.where(key_bits_target == 1)[0]
        guess_size = int(np.sum(key_bits_target))
        n = calc_n(eps[att], guess_size, alpha_M=.05)

        # Randomly select n plaintext/ciphertext pairs
        rand_ind = np.random.choice(range(len(all_pts)), n, replace=False)
        rand_pairs = np.array(list(map(list, zip(all_pts, all_cts))))[rand_ind]
        pts = rand_pairs[:, 0]
        cts = rand_pairs[:, 1]

        # Perform the attack and store the expermiental bias, and the initial key guess
        global biases
        biases.append(attack(pts, pts_msks[att], cts, cts_msks[att], key_targets[att],
                      init_key_guess=init_key_guess))
        key_guess = np.argmax(np.abs(biases[att]))

        # Update the key guess (the key bits that are already found)
        k = dec2bits(init_key_guess)
        k[key_bits_locs] = dec2bits([key_guess], size=guess_size)
        init_key_guess = bits2dec(k)
        print(f"provisional key guess: {init_key_guess}")

    # Perform the final exhaustive search to recover the remaining key bits 
    global final_key_guess
    final_key_target = 0xF - \
        np.bitwise_or.reduce(key_targets[:partial_attacks])
    pt = all_pts[0]
    ct = all_cts[0]
    final_key_guess = exhaustive_search(
        pt, ct, 3, final_key_target, init_key_guess)

    print()
    print(f"final key guess: {final_key_guess}")


if __name__ == "__main__":
    main_attack()
