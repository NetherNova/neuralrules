import unittest
import numpy as np
import tensorflow as tf

from core.model import DifferentiableQueryRules


class ModelTestCase(unittest.TestCase):
    def setUp(self):
        self.num_entities = 4
        self.num_relations = 3
        self.max_rule_len = 2
        self.batch_size = 5
        self.operator_tensor = np.zeros((self.num_relations, self.num_entities, self.num_entities), np.float32)

        non_zero_indices = [[2, 2, 1], [1, 1, 0]]

        non_zero_indices_batch = []
        for b in range(self.batch_size):
            for ind in non_zero_indices:
                non_zero_indices_batch.append([b] + ind)

        self.sq = DifferentiableQueryRules(self.max_rule_len, non_zero_indices_batch, self.num_entities,
                                           self.num_relations, self.batch_size)

    def test_batch(self):
        x_batch = [0, 0, 0, 0, 0]
        y_batch = [2, 2, 2, 2, 2]
        query_batch = [0, 0, 0, 0, 0]
        with tf.Session() as sess:
            self.sq.create_graph()
            tf.global_variables_initializer().run()
            x_, y_, a_matrix, query, a_vector, u, u_sum, w_op = \
                sess.run([self.sq.x_oh, self.sq.y_oh,
                          self.sq.a_matrix, self.sq.query,
                          self.sq.a_vector, self.sq.u,
                          self.sq.u_sum, self.sq.weighted_operator_tensor],
                         feed_dict={self.sq.x: x_batch,
                                    self.sq.y: y_batch,
                                    self.sq.query: query_batch})
            # Test Inputs
            self.assertTrue(np.allclose(x_, np.tile(np.eye(self.num_entities, self.num_entities)[0],
                                                    (self.batch_size, 1))))
            self.assertTrue(np.allclose(y_, np.tile(np.eye(self.num_entities, self.num_entities)[2],
                                                    (self.batch_size, 1))))
            # Test Operators
            self.assertEqual(a_matrix.shape, (self.max_rule_len, self.batch_size, self.num_relations) )
            self.assertTrue(np.allclose(a_vector, a_matrix[0, :]))
            # Test Outcomes

            # first relation weighted, first entity should be the outcome of every batch
            u_result = np.tile(np.array([np.zeros(self.num_entities), np.eye(self.num_entities,self.num_entities)[1],
                                         np.zeros(self.num_entities)]), (self.batch_size, 1, 1))
            self.assertEqual(u.shape, (self.batch_size, self.num_relations, self.num_entities))
            self.assertEqual(u_sum.shape, (self.batch_size, self.num_entities))


if __name__ == '__main__':
    unittest.main()
