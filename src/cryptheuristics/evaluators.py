from .cryptools import monogram_distribution
from .cryptools import digram_distribution
from .cryptools import ngram_distribution

import itertools as it


class SimpleEvaluate:
    """
    First order (callable) class. Can be used to compare some text and typical
    english text using precomputed digram and trigram statistics by assigning
    every digram and trigram score with different weights.
    """

    def __init__(self, beta=0.7, gamma=0.3):

        self.beta = beta
        self.gamma = gamma

        self.digram_suitability = \
            {
                'e ': 2, ' t': 1, 'he': 1, 'th': 1, ' a': 1, 's ': 1, '  ': -6
            }

        self.trigram_suitability = \
            {
                'eee': -5, '   ': -10, 'ing': 5, ' th': 5, 'the': 5, 'he ': 5,
                'and': 5
            }

    def __call__(self, text):

        digram_score = sum(
            (
                self.digram_suitability.get(a+b, 0)
                for a, b in zip(text[:-1], text[1:])
            ))

        trigram_score = sum(
            (
                self.trigram_suitability.get(a+b+c, 0)
                for a, b, c in zip(text[:-2], text[1:-1], text[2:])
            ))

        return self.beta * digram_score + self.gamma * trigram_score


class DiffEvaluate:
    """
    First order (callable) class. Can be used to compare some text and
    language's typical text using monogram, digram and trigram statistics with
    different weights.

    .. code-block:: python

        evaluator = DiffEvaluate(ref_text)
        resemblance_score = evaluator(some_text)
    """

    def __init__(self, ref_text, **kwargs):
        """
        :param ref_text: reference text to get language statistics
        :type ref_text: str
        """
        self.alpha = kwargs.get('alpha', 0)
        self.beta = kwargs.get('beta', 0)
        self.gamma = kwargs.get('gamma', 0)
        self.alphabet = kwargs.get('alphabet', ''.join(set(ref_text)))

        self.expected_monograms = monogram_distribution(ref_text) \
            if self.alpha else None
        self.expected_digrams = digram_distribution(ref_text) \
            if self.beta else None
        self.expected_trigrams = ngram_distribution(ref_text, 3) \
            if self.gamma else None

        if not self.alpha + self.beta + self.gamma > 0:
            raise ValueError("At least one coefficient should be nonzero!")

    def __call__(self, text):

        if self.alpha:
            text_monograms = monogram_distribution(text)
            monogram_score = self.alpha * sum((abs(
                self.expected_monograms.get(key, 0) - text_monograms.get(key, 0)
            ) for key in self.alphabet))
        else:
            monogram_score = 0

        if self.beta:
            text_digrams = digram_distribution(text)
            keys = (''.join(tup) for tup in it.product(self.alphabet, repeat=2))
            digram_score = self.beta * sum((abs(
                self.expected_digrams.get(key, 0) - text_digrams.get(key, 0)
            ) for key in keys))
        else:
            digram_score = 0

        if self.gamma:
            text_trigrams = ngram_distribution(text, 3)
            keys = (''.join(tup) for tup in it.product(self.alphabet, repeat=3))
            trigram_score = self.gamma * sum(abs(
                self.expected_trigrams.get(key, 0) - text_trigrams.get(key, 0)
            ) for key in keys)
        else:
            trigram_score = 0

        score = monogram_score + digram_score + trigram_score

        return (1 - score / (2 * (self.alpha + self.beta + self.gamma)))**8


class WordEvaluate:
    """
    First order (callable) class. Can be used to compare some text consistency
    with language's typical text using counting words in text being proper
    words in considered language.

    .. code-block:: python

        evaluator = WordEvaluate(ref_text)
        resemblance_score = evaluator(some_text)
    """

    def __init__(self, dictionary):
        """
        :param dictionary: list of proper words in given languages
        """
        self.dictionary = set(dictionary[:])

    def score(self, word):
        """
        Compute word's score given by its length if it's a proper word in
        given language and its length is between 2 and 11 or 0 else.

        :param word: word to compute its score
        :type word: str
        :return: word's score
        :rtype: int
        """
        if (2 < len(word) < 11) and word in self.dictionary:
            return len(word)
        return 0

    def __call__(self, text):

        return sum((self.score(word) for word in text.split())) + 1


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
