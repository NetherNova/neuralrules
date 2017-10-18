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
            # TODO: consider batch! a_matrix : 2, 5, 3 ==> a_vector : 5, 3
            a_vector = tf.nn.embedding_lookup(self.a_matrix, x)

            # u_sum = tf.reduce_sum(tf.reshape(a_vector, [self.num_relations, 1, 1]) * tf.expand_dims(self.operator_tensor, 0), 0)
            # reduce either axis 2, or 3
            u_sum = tf.reduce_sum(tf.multiply(tf.reshape(a_vector, [-1, self.num_relations, 1, 1]), tf.expand_dims(self.operator_tensor, 0)), 3)
            # [5, 3, 1, 1] * [1, 3, 4, 4] => [5, 3, 4]

            # [5,3,4] * [5,4]
            return [x+1, tf.matmul(u_sum, tf.expand_dims(previous, 1))]
            # return [x+1, tf.reduce_sum(u_sum * previous, 1)] # is input for next round

        def condition(x, previous):
            return tf.greater(self.max_rule_len, x)

        # placeholders for batch training
        self.x = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None])
        self.query = tf.placeholder(tf.int32, [None])
        self.ops = tf.constant(self.operator_tensor, dtype=tf.float32)

        self.x_oh = tf.one_hot(self.x, depth=self.num_entities, dtype=tf.float32)
        self.y_oh = tf.one_hot(self.y, depth=self.num_entities, dtype=tf.float32)

        # variables [query, rel, T]
        self.a = tf.Variable(tf.random_uniform([self.num_relations, self.max_rule_len, self.num_relations],
                                               minval=0, maxval=0.01))
        self.b = tf.Variable(tf.random_normal([self.num_relations, self.max_rule_len, self.num_relations]))

        # a matrix for each query in batch
        # self.a_matrix = tf.squeeze(tf.nn.embedding_lookup(self.a, self.query), 0) # for these queries
        # switch batch to second axis, to be able to select from *time* on first axis
        self.a_matrix = tf.transpose(tf.nn.embedding_lookup(self.a, self.query), [1, 0, 2])

        self.operator_tensor_reshape = tf.reshape(self.operator_tensor, [-1, self.num_entities])
        self.a_vector = tf.nn.embedding_lookup(self.a_matrix, 0)

        # via matmul it is already summed
        self.u = tf.matmul(self.operator_tensor_reshape, tf.transpose(self.x_oh))
        self.u_reshape = tf.transpose(tf.reshape(self.u, [num_entities, num_relations, -1]), [2, 0, 1])

        self.u_sum = tf.squeeze(tf.matmul(self.u_reshape, tf.expand_dims(self.a_vector, 2)), 2)
        # self.u = tf.reduce_sum(tf.multiply(tf.transpose(self.operator_tensor_reshape), self.x_oh), axis=1)
        # self.u_reshape = tf.multiply(tf.reshape(self.a_vector, [self.num_relations, 1]),
        #                              tf.reshape(self.u, [self.num_relations, self.num_entities]))

        # sum up outcomes of all weighted relation operators
        # self.u_sum = tf.reduce_sum(self.u_reshape, axis=0)
        # self.a_vector = tf.nn.embedding_lookup(self.a_matrix, 0) # first time step for this query

        # # first operator multiply with input x over all relation operators
        # self.u = tf.reduce_sum(tf.multiply(self.operator_tensor_reshape, self.x_oh), axis=1)
        # # reshape back to all relation matrices and weight with respective a
        # u_reshape = tf.multiply(self.a_vector, tf.reshape(self.u, [3, 5]))
        # # sum up outcomes of all weighted relation operators
        # self.u_sum = tf.reduce_sum(u_reshape, axis=0)

        # can pass multiple arguments loop_vars
        #self.result = tf.while_loop(condition, body, [1, self.u_sum])[1]

        # TODO: end loop

        #self.loss = -tf.reduce_sum(tf.multiply(self.y_oh, self.result))
        #self.optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

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
    x_batch = [0, 0, 0, 0, 0]
    y_batch = [2, 2, 2, 2, 2]
    query_batch = [0, 0, 0, 0, 0]

    with tf.Session() as sess:
        sq.create_graph()
        tf.global_variables_initializer().run()
        coeffs_tmp = None
        for x in range(2):
            x_, y_, a_matrix, query, a_vector, u, u_reshape, u_sum = sess.run([sq.x_oh, sq.y_oh, sq.a_matrix, sq.query,
                                                             sq.a_vector, sq.u, sq.u_reshape, sq.u_sum],
                                               feed_dict= { sq.x : x_batch, sq.y : y_batch, sq.query: query_batch} )
            print("x:", x_.shape)
            print("y:", y_.shape)
            print("a_matrix:", a_matrix.shape)
            print("query", query.shape)
            print("a_vector", a_vector.shape)
            print("u", u.shape)
            print("u_reshape", u_reshape.shape)
            print("u_sum", u_sum.shape)
            #_, loss, op_reshape = sess.run([sq.optimizer, sq.loss, sq.u_sum], feed_dict={sq.x : [0], sq.y : [2], sq.query: [0]})
            #print "Loss", loss, op_reshape.shape

            coeffs = sess.run(sq.a)
            if coeffs_tmp is None:
                coeffs_tmp = coeffs
                coeffs_diff = 0
            else:
                coeffs_diff =  coeffs - coeffs_tmp
                coeffs_tmp = coeffs
            print coeffs_diff
        print sess.run(sq.a)