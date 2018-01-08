__author__ = 'martin'

from rdflib import ConjunctiveGraph, Literal, OWL
import numpy as np


"""
Module for parsing and storing the KB
"""


def datatype_property_filter(s, p, o):
    return type(o) == Literal


def OWL_annotations_filter(s, p, o):
    return p == OWL.imports


class KBParser(object):
    def __init__(self):
        pass

    def parse_kb(self, path):
        pass

    def get_facts(self):
        return self.non_zero_indices

    def get_train(self):
        pass

    def get_test(self):
        pass


class RDFKBParser(KBParser):
    def __init__(self):
        self.ent_dict = dict()
        self.rel_dict = dict()
        self.num_relations = 0
        self.num_entities = 0
        self.facts = []
        self.train = []
        self.test = []
        # expanded indices with first dimension for batch tensor
        self.non_zero_indices = []
        self.triples = []
        super(KBParser, self).__init__()

    def parse_kb(self, path, batch_size):
        g = ConjunctiveGraph()
        g.load(path)
        # exclude datatype properties
        self._remove_triples(g, datatype_property_filter)
        self._remove_triples(g, OWL_annotations_filter)
        for (s,p,o) in g.triples((None, None, None)):
            if s not in self.ent_dict:
                self.ent_dict.setdefault(s, len(self.ent_dict))
            if o not in self.ent_dict:
                self.ent_dict.setdefault(o, len(self.ent_dict))
            if p not in self.rel_dict:
                self.rel_dict.setdefault(p, len(self.rel_dict))
            lhs = self.ent_dict[s]
            rel = self.rel_dict[p]
            rhs = self.ent_dict[o]
            self.triples.append((lhs, rel, rhs))
        self.num_entities = len(self.ent_dict)
        self.num_relations = len(self.rel_dict)
        # self.non_zero_entities = ((len(self.rel_dict), len(self.ent_dict), len(self.ent_dict)), dtype=np.float32)
        for t in self.triples:
            for b in range(batch_size):
                self.non_zero_indices.append([b, t[1], t[2], t[0]])
        print('Num triples: ', len(self.triples))
        print('Num ents: ', self.num_entities)
        print('Num rels: ', self.num_relations)

    def _remove_triples(self, g, filter):
        remove_triples = []
        for (s, p, o) in g.triples((None, None, None)):
            if filter(s,p,o):
                remove_triples.append((s, p, o))
        for (s, p, o) in remove_triples:
            g.remove((s, p, o))


class TripleBatchGenerator(object):
    def __init__(self, triples, entity_dictionary, relation_dictionary, num_neg_samples, rnd, sample_negative=False,
                 bern_probs=None):
        self.all_triples = []
        self.batch_index = 0
        self.num_neg_samples = num_neg_samples
        self.rnd = rnd
        self.sample_negative = sample_negative
        self.bern_probs = bern_probs

        for (s, p, o) in sorted(triples):
            self.all_triples.append((s, p, o))

    def next(self, batch_size):
        # return lists of entity and reltaion indices
        inpr = []
        inpl = []
        inpo = []

        inprn = []
        inpln = []
        inpon = []
        if self.sample_negative:
            batch_size_tmp = batch_size // self.num_neg_samples
        else:
            batch_size_tmp = batch_size

        for b in range(batch_size_tmp):
            if self.batch_index >= len(self.all_triples):
                self.batch_index = 0
            current_triple = self.all_triples[self.batch_index]
            # Append current triple with *num_neg_samples* triples
            inpl.append(current_triple[0])
            inpr.append(current_triple[2])
            inpo.append(current_triple[1])
            self.batch_index += 1
        return np.array([inpr, inpl, inpo]), np.array([inprn, inpln, inpon])