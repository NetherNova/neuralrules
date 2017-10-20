import tensorflow as tf
import numpy as np


class DifferentiableQueryRules(object):
    """
    Model to learn relation weights of logical rules for each *query*
    e.g. w1 and w2 in the following example

    query(X,Y) <- w1 * r1(X, Z) , w2 * r2(Z, Y)

    In this simple model, it is assumed that all rules have the same length *max_rule_len*
    cf. [Fan Yang, NIPS 2017]
    """
    def __init__(self, max_rule_len, operator_tensor, num_entities, num_relations):
        self.max_rule_len = max_rule_len
        self.operator_tensor = operator_tensor
        self.num_entities = num_entities
        self.num_relations = num_relations

    def create_graph(self):

        def body(x, previous):
            # batch [max_rule_len, batch_size, num_relations] ==> a_vector : [batch_size, num_relations, 1, 1]
            a_vector = tf.nn.embedding_lookup(self.a_matrix, x)
            a_vector_exp = tf.expand_dims(tf.expand_dims(a_vector, 2), 2)

            # [batch_size, num_relations, num_entities, num_entities]
            weighted_operator_tensor = tf.multiply(a_vector_exp,
                                                        tf.expand_dims(self.operator_tensor, 0))
            # score with previous input
            u = tf.reduce_sum(tf.multiply(weighted_operator_tensor,
                                               tf.expand_dims(tf.expand_dims(previous, 1), 1)), 3)
            # sum up relation weighted vectors
            u_sum = tf.reduce_sum(u, axis=1)
            # [batch_size,num_relations,num_entities,num_entities] * [batch_size,1,num_entities]
            return [x+1, u_sum]

        def condition(x, previous):
            return tf.greater(self.max_rule_len, x)

        # placeholders for batch training
        self.x = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None])
        self.query = tf.placeholder(tf.int32, [None])
        self.ops = tf.constant(self.operator_tensor, dtype=tf.float32)

        self.x_oh = tf.one_hot(self.x, depth=self.num_entities, dtype=tf.float32)
        self.y_oh = tf.one_hot(self.y, depth=self.num_entities, dtype=tf.float32)

        # variables [query, max_rule_len, num_relations]
        self.a = tf.Variable(tf.random_uniform([self.num_relations, self.max_rule_len, self.num_relations],
                                               minval=0, maxval=1))
        # self.b = tf.Variable(tf.random_normal([self.num_relations, self.max_rule_len, self.num_relations]))

        # a matrix for each query in batch
        # switch batch to second axis, to be able to select from *time* on first axis
        self.a_matrix = tf.transpose(tf.nn.embedding_lookup(self.a, self.query), [1, 0, 2])

        # [batch_size, num_relations]
        self.a_vector = tf.nn.embedding_lookup(self.a_matrix, 0)

        # [batch_size, num_relations, 1, 1] * [1, num_relations, num_entities, num_entities]
        # ==> [batch_size,num_relations,num_entities,num_entities]
        self.weighted_operator_tensor = tf.multiply(tf.expand_dims(tf.expand_dims(self.a_vector, 2), 2),
                                                    tf.expand_dims(self.operator_tensor, 0))

        # [batch_size, num_relations, num_entities, num_entities] * [batch_size, 1, 1, num_entities]
        # ==> [batch_size, num_relations, num_entities]
        self.u = tf.reduce_sum(tf.multiply(self.weighted_operator_tensor,
                                           tf.expand_dims(tf.expand_dims(self.x_oh, 1), 1)), 3)

        # sum up outcomes of all weighted relation operators ==> [batch_size,num_entities]
        self.u_sum = tf.reduce_sum(self.u, axis=1)

        self.result = tf.while_loop(condition, body, [1, self.u_sum])[1]

        self.loss = -tf.log(tf.reduce_sum(tf.multiply(self.y_oh, self.result)))
        self.optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

    def loss(self):
        return [self.optimizer, self.loss]


if __name__ == '__main__':
    """
    Simple sanity checking example
    """
    num_entities = 4
    num_relations = 3
    max_rule_len = 2
    operator_tensor = np.zeros((num_relations, num_entities, num_entities), np.float32)

    # two links in the knowledge graph
    operator_tensor[2][2][1] = 1
    operator_tensor[1][1][0] = 1

    sq = DifferentiableQueryRules(max_rule_len, operator_tensor, num_entities, num_relations)
    x_batch = [0, 0, 0, 0, 0]
    y_batch = [2, 2, 2, 2, 2]
    query_batch = [0, 0, 0, 0, 0]

    with tf.Session() as sess:
        sq.create_graph()
        tf.global_variables_initializer().run()
        coeffs_tmp = None
        for x in range(10):
            _, loss, op_reshape = sess.run([sq.optimizer, sq.loss, sq.u_sum],
                                           feed_dict={sq.x : x_batch, sq.y : y_batch, sq.query: query_batch})
            print(x, ": loss:", loss)
            coeffs = sess.run(sq.a)
            if coeffs_tmp is None:
                coeffs_tmp = coeffs
                coeffs_diff = 0
            else:
                coeffs_diff =  coeffs - coeffs_tmp
                coeffs_tmp = coeffs
            print("Rule parameter updates: ", coeffs_diff)
        print("Final rule parameters: ", coeffs)