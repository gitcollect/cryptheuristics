from .cryptools import permute
from .cryptools import inverse

import itertools as it


def ceasar_cipher(text, alphabet, key, decrypt=False):
    """
    Encrypt (or decrypt if *decrypt*) *text* using Ceasar cipher.

    :param text: text to be encrypted
    :type text: str
    :param alphabet: ordered string of all letters in alphabet
    :type alphabet: str
    :param key: single letter of alphabet determining shift of ceasar cipher
    :type key: str
    :param decrypt: flag whether to encrypt (if *False*) or decrypt
    :type decrypt: bool
    :return: encrypted or decrypted text
    :rtype: str
    """
    return vinegere_cipher(text, alphabet, [key], decrypt=decrypt)


def vinegere_cipher(text, alphabet, key, decrypt=False):
    """
    Encrypt (or decrypt if *decrypt*) *text* using Vinegere cipher.

    :param text: text to be encrypted
    :type text: str
    :param alphabet: ordered string of all letters in alphabet
    :type alphabet: str
    :param key: list of alphabet letters determining shifts of Vinegere cipher
    :type key: list
    :param decrypt: flag whether to encrypt (if *False*) or decrypt
    :type decrypt: bool
    :return: encrypted or decrypted text
    :rtype: str
    """
    letter_to_index = dict(zip(alphabet, range(len(alphabet))))
    key_shifts = (letter_to_index[s] for s in key)
    if decrypt:
        key_shifts = [len(alphabet) - x for x in key_shifts]
    key_shifts = it.cycle(key_shifts)
    ciphertext = ''.join(
        (
            alphabet[(letter_to_index[s] + next(key_shifts)) % len(alphabet)]
            for s in text
        ))

    return ciphertext


def transposition_block_cipher(text, key, decrypt=False):
    """
    Encrypt (or decrypt if *decrypt*) *text* using transposition block cipher.

    :param text: text to be encrypted
    :type text: str
    :param key: permutation represented as list of unique integers from {0,n-1}
    :type key: list
    :param decrypt: flag whether to encrypt (if *False*) or decrypt
    :type decrypt: bool
    :return: encrypted or decrypted text
    :rtype: str
    """
    return permute(text, inverse(key) if decrypt else key)
