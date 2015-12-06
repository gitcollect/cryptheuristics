"""Module contains basic functions useful in cryptography/cryptanalysis. Most
of them are used in other modules in **Cryptheuristics** package.

.. moduleauthor:: Marcin Jenczmyk <m.jenczmyk@knm.katowice.pl>
"""

import numpy as np
import operator as op
import collections as coll
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from math import fsum, log


def alpha(source, spaces=True):
    """
    Return string with removed nonalphabetic characters and whitespaces removed
    (*spaces* = *False*) or replaced by spaces (*spaces* = *True*). If *source*
    is path to text file than file is parsed to string, else *source* should be
    string.

    :param source: string or path to text file
    :type source: str
    :param spaces: if *True* replaces all whitespaces with spaces, else removes
        all whitespaces
    :type spaces: bool
    :return: string without nonalphabetic and whitespaces removed or replaced
        with spaces
    :rtype: str

    >>> alpha('data/articles/en/article0.txt', True)
    'egyptian police have used tear gas to disperse protesters angry that c'
    >>> alpha('data/articles/en/article0.txt', False)
    'egyptianpolicehaveusedteargastodisperseprotestersangrythatchargesagain'
    """
    try:
        with open(source, 'r') as input_file:
            source = input_file.read()
    except FileNotFoundError:
        pass

    sep = ' ' if spaces else ''
    source = sep.join(source.split()).lower()

    return ''.join(s for s in source if s.isalpha() or s.isspace())


def ngram_distribution(text, order):
    """
    Return *N*-gram distribution of text.

    :param text: text to be analysed
    :type text: str
    :param order: order of computed ngram distribution
    :type order: int
    :return: dictionary {str: float} with *N*-gram keys, and their frequencies
        as values
    :rtype: dict

    .. code-block:: python

        text = alpha("Hello, I'm the Doctor")
        digram_distribution(text) == ngram_distribution(text, 2)
        # True
    """
    length = len(text) + 1 - order

    ngrams = [
        ''.join(tup) for tup in zip(*[text[i:length+i] for i in range(order)])
    ]

    return {c: val/length for c, val in coll.Counter(ngrams).items()}


def monogram_distribution(text):
    """
    Return monogram distribution of text.

    :param text: text to be analysed
    :type text: str
    :return: dictionary {str: float} with monogram keys, and their frequencies
        as values
    :rtype: dict

    >>> monogram_distribution('hello')
    {'h': 0.2, 'l': 0.4, 'o': 0.2, 'e': 0.2}
    >>> monogram_distribution(alpha("Hello, I'm the Doctor"))
    {' ': 0.15789473684210525, 'l': 0.10526315789473684,
    'i': 0.05263157894736842, 'o': 0.15789473684210525,
    'm': 0.05263157894736842, 'r': 0.05263157894736842,
    'd': 0.05263157894736842, 'c': 0.05263157894736842,
    't': 0.10526315789473684, 'h': 0.10526315789473684,
    'e': 0.10526315789473684}
    """
    return ngram_distribution(text, 1)


def digram_distribution(text):
    """
    Return digram distribution of text.

    :param text: text to be analysed
    :type text: str
    :return: dictionary {str: float} with digram keys, and their frequencies as
        values
    :rtype: dict

    >>> digram_distribution(alpha("Hello, I'm the Doctor")
    {'to': 0.05555555555555555, 'm ': 0.05555555555555555,
    ' d': 0.05555555555555555, 'lo': 0.05555555555555555,
    'oc': 0.05555555555555555, 'he': 0.1111111111111111,
    'im': 0.05555555555555555, 'e ': 0.05555555555555555,
    'do': 0.05555555555555555, 'o ': 0.05555555555555555,
    'ct': 0.05555555555555555, 'll': 0.05555555555555555,
    'or': 0.05555555555555555, 'th': 0.05555555555555555,
    ' i': 0.05555555555555555, ' t': 0.05555555555555555,
    'el': 0.05555555555555555}
    """
    return ngram_distribution(text, 2)


def plot_monogram_distribution(data):
    """
    Plot monogram distribution of text/text with given frequencies on bar plot.

    :param data: text or dictionary {str: float}
    :type data: str or dict
    :return: None

    .. code-block:: python

        text = alpha('data/articles/en/article0.txt', False)
        plot_monogram_distribution(text)

    .. figure:: img/frequencies_en_without_00.png
       :scale: 50
       :align: center
       :alt: Monogram distribution plot

       Monogram frequency plot for english language without spaces.
    """
    try:
        # Check if that works for dictionary
        monograms, freqs = zip(*sorted(data.items(), key=lambda x: x[0]))
    except AttributeError:
        # If doesn't work check if works for string
        data = monogram_distribution(data)
        monograms, freqs = zip(*sorted(data.items(), key=lambda x: x[0]))

    n = len(monograms)

    plt.bar(range(n), freqs, align='center', color='b')

    plt.xticks(range(n), [c.upper() for c in monograms], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([-1, n])
    plt.tight_layout()
    plt.grid(True)

    plt.show()


def plot_digram_distribution(data):
    """
    Plot digram distribution of text/text with given digram frequencies
    on 2D plot with colorbar.

    :param data: text or dictionary {str: float}
    :type data: str or dict
    :return: None

    >>> plot_digram_distribution(alpha('data/articles/en/article0.txt', False))

    .. figure:: img/difrequencies_en_without_00.png
       :scale: 50
       :align: center
       :alt: Digram distribution plot

       Digram frequency plot for english language without spaces.
    """
    try:
        # Check if that works for dictionary
        letters = sorted(list(set(''.join(data))))
        grid = [[data.get(c+s, 0) for s in letters] for c in letters]
    except AttributeError:
        # If doesn't work check if works for string
        distribution = digram_distribution(data)
        letters = sorted(list(set(data)))
        grid = [[distribution.get(c+s, 0) for s in letters] for c in letters]

    plt.imshow(grid, interpolation='none')

    pylab.colorbar().ax.tick_params(labelsize=16)
    plt.xticks(range(len(letters)), [c.upper() for c in letters], fontsize=11)
    plt.yticks(range(len(letters)), [c.upper() for c in letters], fontsize=11)
    plt.tight_layout()

    plt.show()


def entropy(data, unit=2.0):
    """
    Return entropy of text/text with given frequencies. You can select
    unit of entropy by passing logarithm base as optional argument (default
    unit is bit).

    :param data: text or dictionary {str: float}
    :type data: str or dict
    :param unit: logarithm base (default: 2.0)
    :type unit: float
    :return: entropy of text
    :rtype: float

    >>> entropy(alpha('data/articles/en/article0.txt', True))
    4.113249527931756
    """
    try:
        # Check if that works for dictionary
        return -fsum((val * log(val, unit) for val in data.values()))
    except AttributeError:
        # If doesn't work check if works for string
        data = monogram_distribution(data)
        return -fsum((val * log(val, unit) for val in data.values()))


def ic(data):
    """
    Return index of coincidence of text/text with given frequencies.

    :param data: text or dictionary {str: float}
    :type data: str or dict
    :return: index of coincidence of text
    :rtype: float

    >>> ic(alpha('data/articles/en/article0.txt', True))
    0.07550422876863537
    """
    try:
        # Check if that works for dictionary
        return fsum((val**2 for val in data.values()))
    except AttributeError:
        # If doesn't work check if works for string
        data = monogram_distribution(data)
        return fsum((val**2 for val in data.values()))


def expected_ic(data, period=1, q=None):
    """
    Return expected value of index of coincidence of text/text with given
    frequencies encrypted with polyalphabetic substitution cipher with key
    length equal to *period*.

    It's possible to pass number of letters in language that text was written
    (default is number of unique characters in text or keys in dictionary).

    :param data: text or dictionary {str: float}
    :type data: str or dict
    :param period: cipher key length (default: 1)
    :type period: int
    :param q: number of letters in language
    :type q: int
    :return: expected value of index of coincidence
    :rtype: float

    >>> expected_ic(alpha('data/articles/en/article0.txt', True))
    0.07550422876863537
    >>> expected_ic(alpha('data/articles/en/article0.txt', True), 5)
    0.04472944955738299
    >>> expected_ic(alpha('data/articles/en/article0.txt', True), 100)
    0.03742043949471056
    """
    # Sufficiently large number.
    length = 30000.

    try:
        q = len(list(data.keys())) if q is None else q
    except AttributeError:
        q = len(set(data))

    val = (length - period) * ic(data) + ((period - 1) * length / q)
    return val / (period * (length - 1))


def kasiski_examination(text, limit=30):
    """
    Use Kasiski examination to recover period of substitution cipher which was
    used to encrypt text.

    Function compares text with its cyclic shifts by values from 1 to *limit*
    (default *limit*\ =30) and evaluates probability that key period is divisor
    of shift.

    :param text: text being subjected to Kasiski examination
    :type text: str
    :param limit: maximal value of shift
    :type limit: int
    :return: list (*shift*, *val*) where *val* is 'probability' that key length
        is *shift* divisor
    :rtype: list of (int, int) tuples
    """
    return [
        (shift, sum(c == s for c, s in zip(text, text[shift:] + text[:shift])))
        for shift in range(1, limit)
    ]


def permute_block(text_block, permutation, inv=False):
    """
    Permute text block with length equal to permutation length with regard to
    that permutation. This is a helper function for :func:`permute` function.
    """
    q, n = len(text_block), len(permutation)
    assert(q <= n), "Text block too long!"

    inverse_permutation = permutation if inv else inverse(permutation)

    if q == n:
        return ''.join(text_block[k] for k in inverse_permutation)
    else:
        permutation = inverse(permutation) if inv else permutation
        text_block = text_block.ljust(n, 'x')
        padded = permute_block(text_block, inverse_permutation, True)
        indexes = sorted(permutation[:q])
        return ''.join(op.itemgetter(*indexes)(padded))


def permute(text, permutation):
    """
    Encrypt text using transposition block cipher with key given as
    `permutation`.

    :param text: text to be encrypted
    :type text: str
    :param permutation: key (permutation)
    :type permutation: list of int
    :return: encrypted text
    :rtype: str

    >>> permute('abcde', [2,3,1,0,4])
    'dcabe'
    >>> permute('Run!', [2,3,1,0,4])
    '!nRu'
    >>> permute("Hello, I'm The Doctor!", [2,3,1,0,4])
    "llHeo'I, meh T tcDoor!"
    """
    inverse_permutation = inverse(permutation)
    n = len(permutation)
    out_text = ''

    while len(text) > 0:
        out_text += permute_block(text[:n], inverse_permutation, inv=True)
        text = text[n:]

    return out_text


def inverse(permutation):
    """
    Return inverse permutation of *permutation* represented as list of unique
    integer values from 0 (including) to *len(permutation)* (excluding).

    :param permutation: list of unique integers
    :type permutation: list
    :return: inverse permutation of *permutation*
    :rtype: list

    Examples:

    >>> inverse([1,3,2,0])
    [3, 0, 2, 1]
    """
    inv = np.zeros(len(permutation), dtype=int)
    np.put(inv, permutation, np.arange(len(permutation)))

    return list(inv)


def consistency(permutation_a, permutation_b):
    """
    Compute consistency index of two permutations defined as amount of indexes
    which "neighbours" are the same in both permutations.

    :param permutation_a: list of unique integers
    :param permutation_b: list of unique integers
    :return: consistency index
    :rtype: int
    """
    size = len(permutation_a)

    count = 0
    for i in range(size):
        ix = permutation_b.index(permutation_a[i])
        count += permutation_a[(i-1) % size] == permutation_b[(ix-1) % size] \
            and permutation_a[(i+1) % size] == permutation_b[(ix+1) % size]

    return count
