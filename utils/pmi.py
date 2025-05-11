# utils/pmi.py
import math
from collections import defaultdict

class CorpusStats:
    def __init__(self):
        self.doc_count = 0
        self.term_doc_freq = defaultdict(int)
        self.term_pair_count = defaultdict(int)

    def add_document(self, text):
        self.doc_count += 1
        terms = set(text)
        for t1 in terms:
            self.term_doc_freq[t1] += 1
            for t2 in terms:
                if t1 != t2:
                    self.term_pair_count[(t1, t2)] += 1

    def pmi(self, t1, t2):
        if (t1, t2) not in self.term_pair_count:
            return 0
        p_xy = self.term_pair_count[(t1, t2)] / self.doc_count
        p_x = self.term_doc_freq[t1] / self.doc_count
        p_y = self.term_doc_freq[t2] / self.doc_count
        return math.log(p_xy / (p_x * p_y) + 1e-8)
