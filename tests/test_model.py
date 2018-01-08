import unittest
import numpy as np
import tensorflow as tf

from core.model import DifferentiableQueryRules


class ModelTestCase(unittest.TestCase):
    def setUp(self):
        self.num_entities = 4
        self.num_relations = 3
        self.max_rule_len = 2
        self.operator_tensor = np.zeros((self.num_relations, self.num_entities, self.num_entities), np.float32)

        self.operator_tensor[1][1][0] = 1
        self.operator_tensor[2][2][1] = 1

        self.sq = DifferentiableQueryRules(self.max_rule_len, self.operator_tensor, self.num_entities, self.num_relations)

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
            self.assertTrue(np.allclose(x_, np.tile(np.eye(self.num_entities, self.num_entities)[0], (5, 1))))
            self.assertTrue(np.allclose(y_, np.tile(np.eye(self.num_entities, self.num_entities)[2], (5, 1))))
            # Test Operators
            self.assertEqual(a_matrix.shape, (self.max_rule_len, 5, self.num_relations) )
            self.assertTrue(np.allclose(a_vector, a_matrix[0, :]))
            # Test Outcomes

            # first relation weighted, first entity should be the outcome of every batch
            u_result = np.tile(np.array([np.zeros(4), np.eye(4,4)[1], np.zeros(4)]), (5, 1, 1))
            self.assertEqual(u.shape, (5, 3, 4))
            self.assertEqual(u_sum.shape, (5, 4))


if __name__ == '__main__':
    unittest.main()
