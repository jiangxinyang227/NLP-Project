"""
采用text-cnn模型进行多意图分类
"""

import tensorflow as tf


class SiameseModel(object):
    def __init__(self, config, vocab_size, word_vectors):
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        self.query = tf.placeholder(tf.int32, [None, None], name="query")
        self.answer = tf.placeholder(tf.int32, [None, None], name="answer")
        self.labels = tf.placeholder(tf.float32, [None], name="labels")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout

        self.l2_loss = tf.constant(0.0)  # l2 regulation

        # construct graph
        self.model_structure()
        # init model saver
        self.init_saver()

    def model_structure(self):
        # embedding layer
        with tf.name_scope("embedding"):
            # # Initialization of Word Embedding Matrix Using Pre-trained Word Vectors
            if self.word_vectors is not None:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config["embedding_size"]],
                                              initializer=tf.contrib.layers.xavier_initializer())

            query_embedded = tf.nn.embedding_lookup(embedding_w, self.query, name="query_embedded")
            answer_embedded = tf.nn.embedding_lookup(embedding_w, self.answer, name="answer_embedded")

        with tf.name_scope("Bi-LSTM"):
            for idx, hidden_size in enumerate(self.config["hidden_sizes"]):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # frontward lstm cell
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, initializer=tf.orthogonal_initializer(),
                                                state_is_tuple=True),
                        output_keep_prob=self.keep_prob)
                    # backward lstm cell
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, initializer=tf.orthogonal_initializer(),
                                                state_is_tuple=True),
                        output_keep_prob=self.keep_prob)

                    # bi-lstm dynamic decode
                    query_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                      query_embedded,
                                                                      dtype=tf.float32,
                                                                      scope="bi-lstm_1" + str(idx))

                    answer_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                       answer_embedded,
                                                                       dtype=tf.float32,
                                                                       scope="bi-lstm_2" + str(idx))

                    # concat frontward and backward output of lstm
                    query_embedded = tf.concat(query_output, -1)
                    answer_embedded = tf.concat(answer_output, -1)

        with tf.name_scope("final_step_output"):
            query_final_output = query_embedded[:, -1, :]
            answer_final_output = answer_embedded[:, -1, :]

# -------------------------------------------------------------------------------------
# cosine similarity + contrastive loss function
# -------------------------------------------------------------------------------------
#         with tf.name_scope("cosine_similarity"):
#
#             # tensor: [batch_size]
#             query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_final_output), axis=-1))
#
#             # tensor: [batch_size]
#             answer_norm = tf.sqrt(tf.reduce_sum(tf.square(answer_final_output), axis=-1))
#
#             # tensor: [batch_size]
#             dot = tf.reduce_sum(tf.multiply(query_final_output, answer_final_output), axis=-1)
#
#             # tensor: [batch_size]
#             norm = query_norm * answer_norm
#
#             # tensor: [batch_size]
#             self.similarity = tf.div(dot, norm, name="predictions")
#
#         with tf.name_scope("loss"):
#             # Probability calculation with positive prediction
#             pred_pos_prob = tf.div(tf.square((1 - self.similarity)), 4, name="pos_pred")
#             cond = (self.similarity > self.config["neg_threshold"])
#             zeros = tf.zeros_like(self.similarity, dtype=tf.float32)
#             pred_neg_prob = tf.where(cond, tf.square(self.similarity), zeros)
#             losses = self.labels * pred_pos_prob + (1 - self.labels) * pred_neg_prob
#             self.loss = tf.reduce_mean(losses, name="loss")

# -------------------------------------------------------------------------------------
# manhattan distance + binary cross entropy loss function
# -------------------------------------------------------------------------------------
        with tf.name_scope("manhattan_distance"):
            man_distance = tf.reduce_sum(tf.abs(query_final_output - answer_final_output), -1)
            self.sim_score = tf.exp(-man_distance, name="sim")
            self.predictions = tf.cast(tf.greater_equal(self.sim_score, 0.5), tf.int32, name="predictions")

        with tf.name_scope("loss"):
            losses = self.labels * tf.log(self.sim_score) + \
                     (1 - self.labels) * tf.log(1 - self.sim_score)
            self.loss = tf.reduce_mean(-losses)

# ---------------------------------------------------------------------------------------
# euclidean distance + contrastive loss function
# ---------------------------------------------------------------------------------------
#         with tf.name_scope("euclidean distance"):
#             euc_distance = tf.sqrt(tf.reduce_sum(tf.square(query_final_output - answer_final_output), axis=-1))
#             denominator = tf.add(tf.sqrt(tf.reduce_sum(tf.square(query_final_output), axis=-1)),
#                                  tf.sqrt(tf.reduce_sum(tf.square(answer_final_output), axis=-1)))
#             self.similarity = tf.div(euc_distance, denominator, name="similarity")
#             self.predictions = tf.subtract(tf.ones_like(self.similarity), tf.rint(self.similarity), name="predictions")
#
#         with tf.name_scope("loss"):
#             losses = self.labels * tf.square(self.similarity) + \
#                      (1 - self.labels) * tf.square(tf.maximum(1 - self.similarity), 0)
#             self.loss = tf.reduce_mean(losses, name="loss")

        with tf.name_scope("train_op"):
            # define optimizer
            optimizer = self.get_optimizer()

            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            # gradient clip
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    def get_optimizer(self):
        """
        define optimizer
        :return:
        """
        optimizer = None
        if self.config["optimization"] == "adam":
            optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.config["learning_rate"])
        return optimizer

    def init_saver(self):
        """
        init model saver object
        :return:
        """
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, dropout_prob):
        """
        train model method
        :param sess: session object of tensorflow
        :param batch: train batch data
        :param dropout_prob: dropout keep prob
        :return: loss, predict result
        """
        feed_dict = {self.query: batch["first"],
                     self.answer: batch["second"],
                     self.labels: batch["labels"],
                     self.keep_prob: dropout_prob}

        # 训练模型
        _, loss, predictions = sess.run([self.train_op, self.loss, self.predictions],
                                        feed_dict=feed_dict)
        return loss, predictions

    def eval(self, sess, batch):
        """
        eval model method
        :param sess: session object of tensorflow
        :param batch: eval batch data
        :return: loss, predict result
        """
        feed_dict = {self.query: batch["first"],
                     self.answer: batch["second"],
                     self.labels: batch["labels"],
                     self.keep_prob: 1.0}

        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def infer(self, sess, batch):
        """
        infer model method
        :param sess: session object of tensorflow
        :param batch: eval batch data
        :return: similarity result
        """
        feed_dict = {self.query: batch["first"],
                     self.answer: batch["second"],
                     self.keep_prob: 1.0}

        predict = sess.run(self.sim_score, feed_dict=feed_dict)

        return predict

