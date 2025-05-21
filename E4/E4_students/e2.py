from main import random_byte_pattern
from aes import AES, sbox_inv, xtime, mix_single_column, matrix2bytes 
import numpy as np

def encrypt_fault_SB9(aes_obj,pt,n,locx,locy):
    """
        Encrypt a plaintext with a random byte fault injection in the 
        Sbox layer of the 9th round.
        aes_obj: AES, object, similar to encrypt.
        pt, plaintext, as a numpy array of np.uint8.
        n: amount of faulty ciphertext generated 
        locx: [0-3], line chosen to inject the fault
        locy: [0-3], column chosen to inject the fault.

        Return a list, where the first element is the correct
        ciphertext, and where the second element is a list 
        of n faulty ciphertexts.
    """

    fp = random_byte_pattern(locx,locy)
    ctfs = n * [None]
    for i in range(n):
        ctfs[i] = aes_obj.encrypt_block_fault(
                pt,
                f_pattern=fp,
                f_loc="sbox",
                f_round=9
                )
    return ctfs

def compute_ciphertext_fpositions(locxSB9,locySB9):
    """
        Compute the positions of ciphertext faulty bytes considering 
        a fault injection in the Sbox layer of round 9.

        Return a list of 4 tuples, each storing the line and the column indexes.
    """

    # Compute which columns will be faulty after MC
    col_atck_post_MC = (locySB9+ 4 - locxSB9)%4    
    # Compute the faults positions in the ciphertext
    pos = [(i,(col_atck_post_MC+4-i)%4) for i in range(4)]
    return pos

def generate_faults_patterns_postMC(locxSB9): # TODO
    """
        Generate all the possible faults patterns after the MC operation, considering
        a fault occuring at a specific line.
        locxSB9: The line index of the fault. 

        Return a list of tuples, each of tuples being a possible error patterns. (Byte 1, Byte 2, Byte 3, Byte 4)
    """
    # All rows are the same, so we consider columns only
    # We have a position, meaning, we can generate all the possible patterns on that postion

    # Generate all Pre-MC patterns
    patterns = np.zeros((256,4),dtype=np.uint8)
    patterns[:,locxSB9] = np.arange(256,dtype=np.uint8)

    # Calculate the patterns after the MC operation
    for i in range(256):
        mix_single_column(patterns[i])

    # Format and return the patterns
    return [tuple(patterns[i]) for i in range(256)]

def compute_4tuples_subkeys(ct,ctf,fpos,fpatterns):# TODO
    """
        Generate all the 4-bytes subkeys possibilities, taking into account
        the fault positions in the ciphertext and the fault patterns.
        ct: correct ciphertext matrix.
        ctf: a faulty ciphertext matrix, resulting of the same encryption process as the correct one.
        fpos: list of position of faulty bytes in the ciphertext matrix.
        fpatterns: possible faults patterns.

        Return a list of tuples, each tuples being composed of 4 bytes values. Each tuple
        is a possible subkey of 32-bits.
    """
    # Arrive at AK before MC, through inversion, then XOR with the faulty ciphertext (to isolate the faulty bytes)
    # AK | The value we want <- SB <- SR <- AK

    candidates_non_faulty = []
    candidates_faulty = []
    for candidate in range(256):
        # Compute the value of the candidate
        candidate = sbox_inv[ctf[fpos[0][0]][fpos[0][1]] ^ candidate]
        # Compute the value of the faulty ciphertext
        faulty = sbox_inv[ct[fpos[0][0]][fpos[0][1]] ^ candidate]
        # Append the values to the list
        candidates_non_faulty.append(candidate)
        candidates_faulty.append(faulty)

    # Remove the non faulty values
    candidates_faults = []
    for i in range(len(candidates_non_faulty)):
        candidates_faults.append([candidates_non_faulty[i][j] ^ candidates_faulty[i][j] for j in range(4)])

    # Get possible subkeys
    subkeys = []
    for i in range(len(fpatterns)):
        # Compute the value of the subkey
        subkey = []
        for j in range(4):
            # Compute the value of the subkey
            subkey.append(candidates_faults[i][j] ^ fpatterns[i][j])
        # Append the value to the list
        subkeys.append(tuple(subkey))
    # Remove duplicates
    subkeys = list(set(subkeys))
    
    return subkeys

def attack_rndbyte_SB9(ct,ctfs,locxSB9,locySB9):# TODO
    """
        Attack a subkey of 32-bits.
        ct: correct ciphertext matrix.
        ctfs: a list of faulty ciphertext matrix, for the same inpus as the correct one.
        locxSB9: [0-3], line where fault(s) have been injected.
        locySB9: [0-3], column where the fault(s) have been injected.
    """
    
    # Compute the positions of the faulty bytes in the ciphertext
    fpos = compute_ciphertext_fpositions(locxSB9,locySB9)

    # Generate all the possible fault patterns
    fpatterns = generate_faults_patterns_postMC(locxSB9)

    # Compute the possible subkeys
    setk = set()
    for i in range(len(ctfs)):
        # Compute the possible subkeys
        subkeys = compute_4tuples_subkeys(ct,ctfs[i],fpos,fpatterns)
        # Add the subkeys to the set
        for j in range(4):
            setk.add(tuple(subkeys[j]))
    # Convert the set to a list
    setk = list(setk)

    return setk

if __name__ == "__main__":
    # Initialise aes object
    key = np.random.randint(0,256,16)
    aes = AES(key)

    # Generate attack cases
    locx_fault = 2
    locy_fault = 3
    n = 3

    pt = np.random.randint(0,256,16)
    ct = aes.encrypt_block(pt)
    ctfs = encrypt_fault_SB9(aes,pt,n,locx_fault,locy_fault)

    #########
    # Get the real value of the key
    fpos = compute_ciphertext_fpositions(locx_fault,locy_fault)
    last_key = aes._key_matrices[-1].T
    targeted_byte = [
            last_key[fpos[i][0]][fpos[i][1]]
            for i in range(4)
                ]
    ttarget = tuple(targeted_byte)

    # Compute the set of possibles keys remaining
    setk = attack_rndbyte_SB9(ct,ctfs,locx_fault,locy_fault)

    # Print result
    print("Using key:\n{}".format(key))
    print("Using plaintext:\n{}".format(pt))
    print("")
    print("Correct ciphertext:\n{}".format(ct))
    print("")
    print("Last round keys:\n{}".format(aes._key_matrices[-1].T))
    print("")
    print("Random key byte injected on byte [{},{}] at SB9".format(locx_fault,locy_fault))
    for i in range(n):
        print("Faulty ciphertext n°{}:\n{}\n".format(i,ctfs[i]))
    print("")
    print("Target subkey: {}".format(ttarget))
    print("Subkey found after the attack:")
    for ei,e in enumerate(setk):
        print("Skey num {} -> {}".format(ei,e))
    if len(setk)>1:
        print("CAUTION: more than 1 subbyte key isolated...")
    print("")
    if(ttarget in setk):
        print("Successfully isolated the correct subkey.")
    else:
        print("Fail to isolate the correct subkey...")
