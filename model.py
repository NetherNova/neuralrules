import tensorflow as tf
import numpy as np


class SimpleQuery(object):
    def __init__(self, max_rule_len, operator_tensor, num_entities, num_relations):
        self.max_rule_len = max_rule_len
        self.operator_tensor = operator_tensor
        self.num_entities = num_entities
        self.num_relations = num_relations

    def create_graph(self):

        def body(x, previous):
            # TODO: consider batch!
            a_vector = tf.nn.embedding_lookup(self.a_matrix, x)

            # [1, k] [k * E, E]
            u_sum = tf.reduce_sum(tf.reshape(a_vector, [self.num_relations, 1, 1]) * self.operator_tensor, 0)
            # [3,1,1] * [3, 4, 4]

            return [x+1, tf.reduce_sum(u_sum * previous, 1)] # is input for next round

        def condition(x, previous):
            return tf.greater(self.max_rule_len, x)

        # placeholders for batch training
        self.x = tf.placeholder(tf.int32, [1])
        self.y = tf.placeholder(tf.int32, [1])
        self.query = tf.placeholder(tf.int32, [1])
        self.ops = tf.constant(self.operator_tensor, dtype=tf.float32)

        self.x_oh = tf.one_hot(self.x, depth=self.num_entities, dtype=tf.float32)
        self.y_oh = tf.one_hot(self.y, depth=self.num_entities, dtype=tf.float32)

        # variables [query, rel, T]
        self.a = tf.Variable(tf.random_uniform([self.num_relations, self.max_rule_len, self.num_relations],
                                               minval=0, maxval=0.01))
        self.b = tf.Variable(tf.random_normal([self.num_relations, self.max_rule_len, self.num_relations]))

        # a matrix for each query in batch
        self.a_matrix = tf.squeeze(tf.nn.embedding_lookup(self.a, self.query), 0) # for these queries

        self.operator_tensor_reshape = tf.reshape(self.operator_tensor, [-1, self.num_entities])

        a_vector = tf.nn.embedding_lookup(self.a_matrix, 0)
        self.u = tf.reduce_sum(tf.multiply(self.operator_tensor_reshape, self.x_oh), axis=1)

        self.u_reshape = tf.multiply(tf.reshape(a_vector, [self.num_relations, 1]), tf.reshape(self.u, [self.num_relations, self.num_entities]))

        # sum up outcomes of all weighted relation operators
        self.u_sum = tf.reduce_sum(self.u_reshape, axis=0)

        # # TODO: within loop (batch query lookup selection inside the loop)
        #
        # self.a_vector = tf.nn.embedding_lookup(self.a_matrix, 0) # first time step for this query
        #
        # # operator_tensor to matrix

        #
        # # first operator multiply with input x over all relation operators
        # self.u = tf.reduce_sum(tf.multiply(self.operator_tensor_reshape, self.x_oh), axis=1)
        # # reshape back to all relation matrices and weight with respective a
        # u_reshape = tf.multiply(self.a_vector, tf.reshape(self.u, [3, 5]))
        # # sum up outcomes of all weighted relation operators
        # self.u_sum = tf.reduce_sum(u_reshape, axis=0)

        # can pass multiple arguments loop_vars
        self.result = tf.while_loop(condition, body, [1, self.u_sum])[1]

        # TODO: end loop

        self.loss = -tf.reduce_sum(tf.multiply(self.y_oh, self.result))
        self.optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

    def loss(self):
        return [self.optimizer, self.loss]


if __name__ == '__main__':
    num_entities = 4
    num_relations = 3
    max_rule_len = 2
    operator_tensor = np.zeros((num_relations, num_entities, num_entities), np.float32)
    #operator_tensor[0][0][0] = 1
    #operator_tensor[0][1][4] = 1
    #operator_tensor[0][2][0] = 1

    #operator_tensor[1][0][0] = 1
    # operator_tensor[2][0][0] = 1
    operator_tensor[2][2][1] = 1

    #operator_tensor[1][0][0] = 1
    operator_tensor[1][1][0] = 1
    #operator_tensor[1][0][1] = 1
    #operator_tensor[2][2][0] = 1
    #operator_tensor[2][0][2] = 1

    sq = SimpleQuery(max_rule_len, operator_tensor, num_entities, num_relations)
    with tf.Session() as sess:
        sq.create_graph()
        tf.global_variables_initializer().run()
        coeffs_tmp = None
        for x in range(10):
            _, loss, op_reshape = sess.run([sq.optimizer, sq.loss, sq.u_sum], feed_dict={sq.x : [0], sq.y : [2], sq.query: [0]})
            print "Loss", loss, op_reshape

            coeffs = sess.run(sq.a)
            if coeffs_tmp is None:
                coeffs_tmp = coeffs
                coeffs_diff = 0
            else:
                coeffs_diff =  coeffs - coeffs_tmp
                coeffs_tmp = coeffs
            print coeffs_diff
        print sess.run(sq.a)