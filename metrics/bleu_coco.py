#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>
import sys
sys.path.append('../MWPG-DMR')
from metrics.bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, hypo, ref, verbose=1):


        bleu_scorer = BleuScorer(n=self._n)

        bleu_scorer += (hypo[0], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='average', verbose=verbose)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"
