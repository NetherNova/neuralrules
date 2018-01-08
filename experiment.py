import numpy as np
import tensorflow as tf
from core.database import RDFKBParser, TripleBatchGenerator
from core.model import DifferentiableQueryRules
from core.helpers import print_rules

"""
Main script for the execution of rule learning models
"""

path = './kg/PPR_individuals.rdf'
max_rule_len = 2
batch_size = 16
rnd = np.random

parser = RDFKBParser()
parser.parse_kb(path, batch_size)
num_entities = parser.num_entities
num_relations = parser.num_relations

model = DifferentiableQueryRules(max_rule_len, parser.get_facts(), num_entities, num_relations, batch_size)
tg = TripleBatchGenerator(parser.triples, parser.ent_dict, parser.rel_dict, 0, rnd)

inverse_rel_dict = dict(zip(parser.rel_dict.values(), parser.rel_dict.keys()))

with tf.Session() as sess:
    model.create_graph()
    tf.global_variables_initializer().run()
    for i in range(20):
        batch, _ = tg.next(batch_size)
        feed_dict = {
            model.x: batch[1, :],
            model.y: batch[0, :],
            model.query: batch[2, :]
        }
        _, loss = sess.run(model.optimize(), feed_dict=feed_dict)
        print("Loss: " ,loss)
    a = sess.run(model.a)
    print_rules(a, inverse_rel_dict)