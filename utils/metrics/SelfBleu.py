# --------------------------------------------------------------------------------
# Note that different from its original code https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py,
# we do not use "is_first" and "is_fast", because an issue exists otherwise for evaluating self-BLEU over training: Only
# in the first time of evaluation that the reference and hypothesis come from the same “test data” (i.e. the whole set
# of generated sentences). After that, the hypothesis keeps updated but the reference remains unchanged (due to
# “self.is_first=False”), which means hypothesis and reference are not from the same “test data” any more, and thus the
# scores obtained under that implementation is not self-BLEU scores.
# --------------------------------------------------------------------------------

import sys, math, nltk
import numpy as np
# from nltk.translate.bleu_score import SmoothingFunction

from nltk.util import ngrams
from utils.metrics.Metrics import Metrics
from nltk.util import ngrams
import os
from multiprocessing import Pool

def flatten_lol(lol):
    flat_l = []
    for l in lol:
        flat_l += l
    return flat_l


class SelfBleu(Metrics):
    def __init__(self, test_text='', gram=3, name='SelfBleu', portion=1):
        super().__init__()
        self.name = name
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 200  # SelfBLEU scores remain nearly unchanged for self.sample_size >= 200
        self.portion = portion  # how many posrtions to use in the evaluation, default to use the whole test dataset
        self.weights = tuple((1. / self.gram for _ in range(self.gram)))

    def get_name(self):
        return self.name

    def get_score(self, is_fast=False, ignore=False):
        if ignore:
            return 0

        return self.get_bleu()

    def get_reference(self):
        reference = list()
        with open(self.test_data) as real_data:
            for text in real_data:
                text = nltk.word_tokenize(text)
                reference.append(text)
        len_ref = len(reference)
        reference = reference[:int(self.portion * len_ref)]
        # refs_n_grams = set(flatten_lol(self.get_ngrams_all(reference, self.gram)))
        self.refs_n_grams_1 = set(flatten_lol(self.get_ngrams_all(reference, 1)))
        self.refs_n_grams_2 = set(flatten_lol(self.get_ngrams_all(reference, 2)))
        refs_n_grams_3 = None
        if self.gram >= 3:
            self.refs_n_grams_3 = set(flatten_lol(self.get_ngrams_all(reference, 3)))
        # refs_n_grams_4 = None
        if self.gram >= 4:
            self.refs_n_grams_4 = set(flatten_lol(self.get_ngrams_all(reference, 4)))
        if self.gram >= 5:
            self.refs_n_grams_5 = set(flatten_lol(self.get_ngrams_all(reference, 5)))

        self.ref_lengths = np.sort(np.array(list(set([len(reference) for reference in reference]))))
        # return refs_n_grams_1, refs_n_grams_2, refs_n_grams_3, refs_n_grams_4

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        self.get_reference()
        # weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            i = 0
            for hypothesis in test_data:
                if i >= self.sample_size:
                    break
                hypothesis = nltk.word_tokenize(hypothesis)
                # bleu.append(self.calc_bleu(reference, hypothesis, weight))
                # bleu.append(self.get_all_bleus_new([hypothesis], refs_grams, weights)[0])
                bleu.append(self.get_all_sentence_bleus_fast([hypothesis])[0])
                i += 1

        return sum(bleu) / len(bleu)

    # def calc_bleu(self, reference, hypothesis, weight):
    #     return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
    #                                                    smoothing_function=SmoothingFunction().method1)

    def get_n_grams(self, seq, n):
        grams = ngrams(seq, n)
        grams_list = [g for g in grams]
        return grams_list

    # Flatten a list of lists

    def get_ngrams_all(self, seqs_arr, n):
        n_grams = [self.get_n_grams(seq, n) for seq in seqs_arr]
        return n_grams

    def get_intersected_ngrams(self, model_n_grams, ref_n_grams):
        intersected_n_grams = list()
        for gram in model_n_grams:
            if gram in ref_n_grams:
                intersected_n_grams.append(gram)
        return len(intersected_n_grams)

    def doSmoothing(self, numerators, denumerators, method=0):
        if method == 0:
            return [sys.float_info.min if n_i == 0 else n_i/den_i for n_i, den_i in zip(numerators, denumerators)]
        elif method == 1:
            return [(n_i + 0.1) / den_i if n_i == 0 else n_i/den_i for n_i, den_i in zip(numerators, denumerators)]

    def get_all_sentence_bleus_fast(self, gen_indices_cleaned, smoothing_method=0):
        # model_n_grams_list = get_ngrams_all(gen_indices_cleaned, ngram)
        # for i, s in enumerate(gen_indices_cleaned):
        #     bleus_list.append(get_intersected_ngrams(model_n_grams_list[i], refs_n_grams) / len(model_n_grams_list[i]))
        # return np.array(bleus_list)

        bleus_list = list()

        for s in gen_indices_cleaned:
            numerators = list()
            denumerators = list()
            for i_gram in range(1, self.gram+1):
                if i_gram == 1:
                    ref_n_grams = self.refs_n_grams_1
                elif i_gram == 2:
                    ref_n_grams = self.refs_n_grams_2
                elif i_gram == 3:
                    ref_n_grams = self.refs_n_grams_3
                elif i_gram == 4:
                    ref_n_grams = self.refs_n_grams_4
                elif i_gram == 5:
                    ref_n_grams = self.refs_n_grams_5

                model_n_grams_list = self.get_n_grams(s, i_gram)
                numerators.append(self.get_intersected_ngrams(model_n_grams_list, ref_n_grams))
                denumerators.append(len(model_n_grams_list))

            # s_bleu = (w_i * math.log(float(nu_i)/float(denu_i) if nu_i > 0 else sys.float_info.min) for w_i, nu_i, denu_i in zip(weights, numerators, denumerators))
            grams_fractions = self.doSmoothing(numerators, denumerators, method=smoothing_method)
            fr = list()
            for w_i, fr_i in zip(self.weights, grams_fractions):
                fr.append(w_i * math.log(fr_i))

            # brevity penalty (for correct impelenetation of different hypth and ref lengths, see corpus_bleu function in nltk
            # Iterate through each hypothesis and their corresponding references.


            # Calculate the hypothesis length and the closest reference length.
            # Adds them to the corpus-level hypothesis and reference counts.
            """
            NOTE: this BLEU function is a sentence level one, so I only compute one hypothesis length.
            for the whole corpus case (which we don't need so far in text generation), the total hypo. lengths
            of all corpus sentences should be summed """
            hyp_lengths, ref_lengths = 0, 0

            hyp_len = len(s)
            hyp_lengths += hyp_len
            """
            Finds the reference that is the closest length to the
            hypothesis. The closest reference length is referred to as *r* variable
            from the brevity penalty formula in Papineni et. al. (2002)"""

            # closest_ref_len = min(self.ref_lengths, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
            i_closest_ref_len = np.argmin(abs(self.ref_lengths - hyp_len))
            closest_ref_len = self.ref_lengths[i_closest_ref_len]
            ref_lengths += closest_ref_len

            # Calculate brevity penalty.
            bp = self.brevity_penalty(ref_lengths, hyp_lengths)

            s_bleu = bp * math.exp(math.fsum(fr))
            bleus_list.append(s_bleu)

        return bleus_list

    def brevity_penalty(self, closest_ref_len, hyp_len):
        """
        Calculate brevity penalty.

        As the modified n-gram precision still has the problem from the short
        length sentence, brevity penalty is used to modify the overall BLEU
        score according to length.

        An example from the paper. There are three references with length 12, 15
        and 17. And a concise hypothesis of the length 12. The brevity penalty is 1.

            # >>> reference1 = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12
            # >>> reference2 = list('aaaaaaaaaaaaaaa')   # i.e. ['a'] * 15
            # >>> reference3 = list('aaaaaaaaaaaaaaaaa') # i.e. ['a'] * 17
            # >>> hypothesis = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12
            # >>> references = [reference1, reference2, reference3]
            # >>> hyp_len = len(hypothesis)
            # >>> closest_ref_len =  closest_ref_length(references, hyp_len)
            # >>> brevity_penalty(closest_ref_len, hyp_len)
            1.0

        In case a hypothesis translation is shorter than the references, penalty is
        applied.

            # >>> references = [['a'] * 28, ['a'] * 28]
            # >>> hypothesis = ['a'] * 12
            # >>> hyp_len = len(hypothesis)
            # >>> closest_ref_len =  closest_ref_length(references, hyp_len)
            # >>> brevity_penalty(closest_ref_len, hyp_len)
            0.2635971381157267

        The length of the closest reference is used to compute the penalty. If the
        length of a hypothesis is 12, and the reference lengths are 13 and 2, the
        penalty is applied because the hypothesis length (12) is less then the
        closest reference length (13).

            # >>> references = [['a'] * 13, ['a'] * 2]
            # >>> hypothesis = ['a'] * 12
            # >>> hyp_len = len(hypothesis)
            # >>> closest_ref_len =  closest_ref_length(references, hyp_len)
            # >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS
            0.9200...

        The brevity penalty doesn't depend on reference order. More importantly,
        when two reference sentences are at the same distance, the shortest
        reference sentence length is used.

            # >>> references = [['a'] * 13, ['a'] * 11]
            # >>> hypothesis = ['a'] * 12
            # >>> hyp_len = len(hypothesis)
            # >>> closest_ref_len =  closest_ref_length(references, hyp_len)
            # >>> bp1 = brevity_penalty(closest_ref_len, hyp_len)
            # >>> hyp_len = len(hypothesis)
            # >>> closest_ref_len =  closest_ref_length(reversed(references), hyp_len)
            # >>> bp2 = brevity_penalty(closest_ref_len, hyp_len)
            # >>> bp1 == bp2 == 1
            True

        A test example from mteval-v13a.pl (starting from the line 705):

            # >>> references = [['a'] * 11, ['a'] * 8]
            # >>> hypothesis = ['a'] * 7
            # >>> hyp_len = len(hypothesis)
            # >>> closest_ref_len =  closest_ref_length(references, hyp_len)
            # >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS
            # 0.8668...
            #
            # >>> references = [['a'] * 11, ['a'] * 8, ['a'] * 6, ['a'] * 7]
            # >>> hypothesis = ['a'] * 7
            # >>> hyp_len = len(hypothesis)
            # >>> closest_ref_len =  closest_ref_length(references, hyp_len)
            # >>> brevity_penalty(closest_ref_len, hyp_len)
            1.0

        :param hyp_len: The length of the hypothesis for a single sentence OR the
        sum of all the hypotheses' lengths for a corpus
        :type hyp_len: int
        :param closest_ref_len: The length of the closest reference for a single
        hypothesis OR the sum of all the closest references for every hypotheses.
        :type closest_ref_len: int
        :return: BLEU's brevity penalty.
        :rtype: float
        """
        if hyp_len > closest_ref_len:
            return 1
        # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
        elif hyp_len == 0:
            return 0
        else:
            return math.exp(1 - closest_ref_len / hyp_len)

