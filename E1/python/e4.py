import numpy as np
from e1 import encrypt, decrypt
from utils import dec2bits, bits2dec
from e3 import gen_data, experimental_bias
from tqdm import tqdm
from scipy.special import erfinv

sboxinv: np.ndarray[np.uint8] = np.load("sbox.npz")["sboxinv"]
playerinv: np.ndarray[np.uint8] = np.load("player.npz")["playerinv"]


def attack(
    pts, pts_msk, cts, cts_msk, key_target,
    init_key_guess=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
) -> None:

    key_bits_target = dec2bits(key_target)
    guess_size = int(np.sum(key_bits_target))
    key_bits_locs = np.where(key_bits_target == 1)[0]

    k = dec2bits(init_key_guess)
    subst_outs = np.zeros_like(cts, dtype=np.uint8)
    bias = np.zeros(2**guess_size)

    for key_guess in tqdm(range(2**guess_size)):
        k[key_bits_locs] = dec2bits([key_guess], size=guess_size)
        for i, ct in enumerate(cts):
            xored_out = ct ^ bits2dec(k)
            permu_out = bits2dec(dec2bits(xored_out)[playerinv])
            subst_outs[i] = sboxinv[permu_out]
        bias[key_guess] = experimental_bias(pts, pts_msk, subst_outs, cts_msk)

    return bias


def exhaustive_search(final_key_target, init_key_guess, pt, ct, nrounds):

    key_bits_target = dec2bits(final_key_target)
    guess_size = int(np.sum(key_bits_target))
    key_bits_locs = np.where(key_bits_target == 1)[0]
    k = dec2bits(init_key_guess)

    for key_guess in tqdm(range(2**guess_size)):
        k[key_bits_locs] = dec2bits([key_guess], size=guess_size)
        final_key_guess = bits2dec(k)
        if (encrypt(pt, final_key_guess, nrounds) == ct).all():
            return final_key_guess
        else:
            pass


def calc_n(eps, alpha_M, M):
    alpha = 1 - (1-alpha_M)**(1/2**M)
    d = 4*(erfinv(2*alpha-1))**2
    n = round(d/(2*eps**2))
    return n


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
    n = calc_n(eps[att], alpha_M, guess_size)

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
    init_key_guess = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    all_pts = np.load("pts_cts_pairs.npz")["pts"]
    all_cts = np.load("pts_cts_pairs.npz")["cts"]
    for att in range(partial_attacks):
        key_bits_target = dec2bits(key_targets[att])
        key_bits_locs = np.where(key_bits_target == 1)[0]
        guess_size = int(np.sum(key_bits_target))
        n = calc_n(eps[att], .05, guess_size)

        rand_ind = np.random.choice(range(len(all_pts)), n, replace=False)
        rand_pairs = np.array(list(map(list, zip(all_pts, all_cts))))[rand_ind]
        pts = rand_pairs[:, 0]
        cts = rand_pairs[:, 1]

        global biases
        biases.append(attack(pts, pts_msks[att], cts, cts_msks[att], key_targets[att],
                      init_key_guess=init_key_guess))
        key_guess = np.argmax(np.abs(biases[att]))

        k = dec2bits(init_key_guess)
        k[key_bits_locs] = dec2bits([key_guess], size=guess_size)
        init_key_guess = bits2dec(k)
        print(f"provisional key guess: {init_key_guess}")

    global final_key_guess
    final_key_target = 0xF - \
        np.bitwise_xor.reduce(key_targets[:partial_attacks])
    pt = all_pts[0]
    ct = all_cts[0]
    final_key_guess = exhaustive_search(
        final_key_target, init_key_guess, pt, ct, 3)

    print()
    print(f"final key guess: {final_key_guess}")


if __name__ == "__main__":
    main_attack()
