__author__ = 'martin'

"""
Module for parsing and storing the KB
"""

from rdflib import ConjunctiveGraph


class KBParser(object):
    def __init__(self):
        self.num_relations = 0
        self.num_entities = 0
        self.facts = []
        self.train = []
        self.test = []

    def parse_kb(self, path):
        pass

    def get_facts(self):
        pass

    def get_train(self):
        pass

    def get_test(self):
        pass


class RDFKBParser(KBParser):
    def __init__(self):
        super(KBParser, self).__init__()

    def parse_kb(self, path):
        pass


class KB(object):
    def __init__(self):
        self.operator_tensor = None
        self.num_entities = 0

    def parse_kb(self, path):
        pass