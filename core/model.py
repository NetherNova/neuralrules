import tensorflow as tf
import numpy as np

"""
Models for rule learning

Limitations: Need to replicate sparse operator tensor to match *batch_size* of inputs
"""


def tensor_mul_batch_vector(vector, tensor):
    """
    broadbast batch vector and apply it to the matching dimension in a static tensor
    :param vector:
    :param tensor:
    :return:
    """
    # vector [batch_size, d1, 1, 1]
    vector_exp = tf.expand_dims(tf.expand_dims(vector, 2), 2)
    # non-sparse broadcasting:
    # tensor [1, d1, d2, d3]
    # tensor_exp = tf.expand_dims(tensor, 0)
    # [batch_size, d1, d2, d3]
    return tensor.__mul__(vector_exp)


class DifferentiableQueryRules(object):
    """
    Model to learn relation weights of logical rules for each *query*
    e.g. w1 and w2 in the following example

    query(X,Y) <- w1 * r1(X, Z) , w2 * r2(Z, Y)

    In this simple model, it is assumed that all rules have the same length *max_rule_len*
    cf. [Fan Yang, NIPS 2017]
    """
    def __init__(self, max_rule_len, non_zero_indices, num_entities, num_relations, batch_size):
        self.max_rule_len = max_rule_len
        self.non_zero_indices = non_zero_indices
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.batch_size = batch_size

    def create_graph(self):

        def body(x, previous):
            # batch [max_rule_len, batch_size, num_relations]
            # ==> a_vector : [batch_size, num_relations, 1, 1]
            a_vector = tf.nn.embedding_lookup(self.a_matrix, x)
            weighted_operator_tensor = tensor_mul_batch_vector(a_vector, self.operator_tensor)
            # a_vector_exp = tf.expand_dims(tf.expand_dims(a_vector, 2), 2)

            # [batch_size, num_relations, num_entities, num_entities]
            # weighted_operator_tensor = tf.multiply(a_vector_exp,
            #                                            tf.expand_dims(self.operator_tensor, 0))
            # score with previous input
            u = tf.sparse_reduce_sum(weighted_operator_tensor.__mul__(
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
        self.operator_tensor = tf.SparseTensor(indices=self.non_zero_indices,
                                               values=np.ones(len(self.non_zero_indices), dtype=np.float32),
                                               dense_shape=[self.batch_size, self.num_relations, self.num_entities,
                                                            self.num_entities])
        self.operator_tensor = tf.sparse_reorder(self.operator_tensor)

        self.x_oh = tf.one_hot(self.x, depth=self.num_entities, dtype=tf.float32)
        self.y_oh = tf.one_hot(self.y, depth=self.num_entities, dtype=tf.float32)

        # variables [query, max_rule_len, num_relations]
        self.a = tf.Variable(tf.random_uniform([self.num_relations, self.max_rule_len, self.num_relations],
                                               minval=0, maxval=0.1))
        # self.b = tf.Variable(tf.random_normal([self.num_relations, self.max_rule_len, self.num_relations]))

        # a matrix for each query in batch
        # switch batch to second axis, to be able to select from *time* on first axis
        self.a_matrix = tf.transpose(tf.nn.embedding_lookup(self.a, self.query), [1, 0, 2])

        # [batch_size, num_relations]
        self.a_vector = tf.nn.embedding_lookup(self.a_matrix, 0)

        # broadcast relation weight to each respective "1" element in the operator tensor

        # [batch_size, num_relations, 1, 1] * [batch_size, num_relations, num_entities, num_entities]
        # ==> [batch_size,num_relations,num_entities,num_entities]
        self.weighted_operator_tensor = tensor_mul_batch_vector(self.a_vector, self.operator_tensor)

        # [batch_size, num_relations, num_entities, num_entities] * [batch_size, 1, 1, num_entities]
        # ==> [batch_size, num_relations, num_entities]
        self.weighted_operator_tensor_query = self.weighted_operator_tensor.__mul__(
            tf.expand_dims(tf.expand_dims(self.x_oh, 1), 1))

        # sparse sum returns dense
        self.u = tf.sparse_reduce_sum(self.weighted_operator_tensor_query, 3)

        # sum up outcomes of all weighted relation operators
        # ==> [batch_size,num_entities]
        self.u_sum = tf.reduce_sum(self.u, axis=1)

        self.result = tf.while_loop(condition, body, [1, self.u_sum])[1]

        self.loss = -tf.log(tf.reduce_sum(tf.multiply(self.y_oh, self.result)))
        self.optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)

    def optimize(self):
        return [self.optimizer, self.loss]


class AttentionQueryModel(object):
    def __init__(self, max_rule_len, operator_tensor, num_entities, num_relations, hidden_size):
        self.max_rule_len = max_rule_len
        self.operator_tensor = operator_tensor
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_size = hidden_size

    def create_graph(self):
        # placeholders for batch training
        self.x = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None])
        self.query = tf.placeholder(tf.int32, [None])
        self.ops = tf.constant(self.operator_tensor, dtype=tf.float32)

        self.x_oh = tf.one_hot(self.x, depth=self.num_entities, dtype=tf.float32)
        self.y_oh = tf.one_hot(self.y, depth=self.num_entities, dtype=tf.float32)

        self.a = tf.Variable(tf.random_uniform([self.num_relations, self.max_rule_len, self.num_relations]))
        self.u = tf.Variable(tf.random_uniform([self.num_relations, self.max_rule_len + 1, self.num_relations]))
        self.W = tf.Variable(tf.random_uniform([self.hidden_size, self.max_rule_len]))
        self.bias = tf.Variable(tf.ones(self.max_rule_len))

        # LSTM get hidden update
        # LSTM(self.x_oh)

    def loss(self):
        return []



if __name__ == '__main__':
    """
    Simple sanity checking example
    """
    num_entities = 4
    num_relations = 3
    max_rule_len = 2
    batch_size = 5
    # two links in the knowledge graph
    # [relation, object, subject]
    non_zero_indices = [[2, 2, 1], [1, 1, 0]]

    non_zero_indices_batch = []
    for b in range(batch_size):
        for ind in non_zero_indices:
            non_zero_indices_batch.append([b] + ind)

    sq = DifferentiableQueryRules(max_rule_len, non_zero_indices_batch, num_entities, num_relations, batch_size)
    # can be seen like: how to get from entity 0 to entity 2 to explain their relation 2
    x_batch = [0, 0, 0, 0, 0]
    y_batch = [2, 2, 2, 2, 2]
    query_batch = [0, 0, 0, 0, 0]

    # now, we expect:
    # a) the weight of relation 1 to increase for rule len 1
    # b) the weight of relation 2 to increase for rule len 2

    with tf.Session() as sess:
        sq.create_graph()
        tf.global_variables_initializer().run()
        coeffs_tmp = None
        for i in range(10):
            _, loss, op_reshape = sess.run([sq.optimizer, sq.loss, sq.u_sum],
                                           feed_dict={sq.x : x_batch, sq.y : y_batch, sq.query: query_batch})
            print(i, ": loss:", loss)
            coeffs = sess.run(sq.a)
            if coeffs_tmp is None:
                coeffs_tmp = coeffs
                coeffs_diff = 0
            else:
                coeffs_diff =  coeffs - coeffs_tmp
                coeffs_tmp = coeffs
            print("Rule parameter updates: ", coeffs_diff)
        print("Final rule parameters: ", coeffs)