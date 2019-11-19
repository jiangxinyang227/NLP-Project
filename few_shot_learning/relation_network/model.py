"""
prototypical network model for few shot learning
"""

import tensorflow as tf


class RelationModel(object):
    def __init__(self, config, vocab_size, word_vectors):
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        # [num_classes, num_support, sequence_length]
        self.support = tf.placeholder(tf.int32, [None, None, None], name="support")
        # [num_classes * num_queries, sequence_length]
        self.queries = tf.placeholder(tf.int32, [None, None], name="queries")
        # [num_classes * num_queries]
        self.labels = tf.placeholder(tf.int32, [None], name="labels")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout

        self.l2_loss = tf.constant(0.0)  # l2 regulation

        # construct graph
        self.model_structure()
        # Initialize the object that saves the model
        self.init_saver()

    def model_structure(self):
        # embedding layer
        with tf.name_scope("embedding"):
            # Initialization of Word Embedding Matrix Using Pre-trained Word Vectors
            if self.word_vectors is not None:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config["embedding_size"]],
                                              initializer=tf.random_normal_initializer())

            # support embedding. dimension: [num_classes, num_support, sequence_length, embedding_size]
            support_embedded = tf.nn.embedding_lookup(embedding_w, self.support, name="support_embedded")
            # query embedding. dimension: [num_classes * num_queries, sequence_length, embedding_size]
            queries_embedded = tf.nn.embedding_lookup(embedding_w, self.queries, name="queries_embedded")

            # reshape support set to 3 dimensions
            support_embedded_reshape = tf.reshape(support_embedded,
                                                  [-1, self.config["sequence_length"], self.config["embedding_size"]])

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
                    support_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                        support_embedded_reshape,
                                                                        dtype=tf.float32,
                                                                        scope="bi-lstm_1" + str(idx))

                    queries_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                        queries_embedded,
                                                                        dtype=tf.float32,
                                                                        scope="bi-lstm_2" + str(idx))

                    # concat frontward and backward output of lstm
                    support_embedded_reshape = tf.concat(support_output, -1)
                    queries_embedded = tf.concat(queries_output, -1)

        with tf.name_scope("concat_support_query"):
            # [num_classes * num_support, hidden_size * 2]
            support_final_output = self._attention(support_embedded_reshape, scope_name="support")
            # [num_classes * num_queries, hidden_size * 2]
            queries_final_output = self._attention(queries_embedded, scope_name="queries")
            # [num_classes, num_support, hidden_size * 2]
            support_final_output = tf.reshape(support_final_output,
                                              [self.config["num_classes"],
                                               self.config["num_support"],
                                               self.config["hidden_sizes"][-1] * 2])

            # computing class vector means. dimension: [num_classes, hidden_size * 2]
            support_class_final_output = tf.reduce_mean(support_final_output, axis=1)

        # define relation module
        with tf.name_scope("relation_layer"):
            scores = self.neural_tensor_layer(support_class_final_output, queries_final_output)
            self.predictions = tf.argmax(scores, axis=-1, name="predictions")

        with tf.name_scope("loss"):
            labels_one_hot = tf.one_hot(self.labels, self.config["num_classes"], dtype=tf.float32)
            self.loss = tf.losses.mean_squared_error(labels=labels_one_hot, predictions=scores)

        with tf.name_scope("train_op"):
            # define optimizer
            optimizer = self.get_optimizer()

            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            # gradient clip
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    def neural_tensor_layer(self, class_vector, query_encoder):
        """
        calculate relation scores
        :param class_vector: class vectors
        :param query_encoder: query set encoding matrix. [num_classes * num_queries, encode_size]
        :return:
        """
        num_classes = self.config["num_classes"]
        encode_size = self.config["hidden_sizes"][-1] * 2
        layer_size = self.config["layer_size"]

        M = tf.get_variable("M", [encode_size, encode_size, layer_size], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=(2 / encode_size) ** 0.5))

        # 该层可以理解为从不同的视角去计算类向量和query向量的分数
        # [[class1, class2, ..], [class1, class2, ..], ... layer_size]
        all_mid = []
        for i in range(layer_size):
            # [num_classes, num_classes * num_queries]
            slice_mid = tf.matmul(tf.matmul(class_vector, M[:, :, i]), query_encoder, transpose_b=True)
            all_mid.append(tf.split(slice_mid, [1] * num_classes, axis=0))

        # [[1, 2, .. layer_size], ... class_n], 将同一个类经tensor layer计算出来的分数放在一起
        all_mid = [[mid[j] for mid in all_mid] for j in range(len(all_mid[0]))]

        # [layer_size, num_classes * num_queries]
        all_mid_concat = [tf.concat(mid, axis=0) for mid in all_mid]

        # [num_classes * num_queries, layer_size]
        all_mid_transpose = [tf.nn.relu(tf.transpose(mid)) for mid in all_mid_concat]

        relation_w = tf.get_variable("relation_w", [layer_size, 1], dtype=tf.float32,
                                     initializer=tf.glorot_normal_initializer())
        relation_b = tf.get_variable("relation_b", [1], dtype=tf.float32,
                                     initializer=tf.glorot_normal_initializer())

        scores = []
        for mid in all_mid_transpose:
            score = tf.nn.sigmoid(tf.matmul(mid, relation_w) + relation_b)
            scores.append(score)

        # [num_classes * num_queries, num_classes]
        scores = tf.concat(scores, axis=-1)

        return scores

    def _attention(self, H, scope_name):
        """
        attention for the final output of Lstm
        :param H: [batch_size, sequence_length, hidden_size * 2]
        """
        with tf.variable_scope(scope_name):
            hidden_size = self.config["hidden_sizes"][-1] * 2
            attention_size = self.config["attention_size"]
            w_1 = tf.get_variable("w_1", shape=[hidden_size, attention_size],
                                  initializer=tf.glorot_normal_initializer)

            w_2 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

            # Nonlinear conversion for LSTM output, [batch_size * sequence_length, attention_size]
            M = tf.tanh(tf.matmul(tf.reshape(H, [-1, hidden_size]), w_1))

            # calculate weights, [batch_size, sequence_length]
            weights = tf.reshape(tf.matmul(M, tf.reshape(w_2, [-1, 1])),
                                 [-1, self.config["sequence_length"]])

            # softmax normalization, [batch_size, sequence_length]
            alpha = tf.nn.softmax(weights, axis=-1)

            # calculate weighted sum
            # r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, self.config["sequence_length"], 1]))
            # print(r)
            output = tf.reduce_sum(H * tf.reshape(alpha, [-1, self.config["sequence_length"], 1]), axis=1)

            return output

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
        feed_dict = {self.support: batch["support"],
                     self.queries: batch["queries"],
                     self.labels: batch["labels"],
                     self.keep_prob: dropout_prob}

        # 训练模型
        _, loss, predictions = sess.run([self.train_op, self.loss, self.predictions],
                                        feed_dict=feed_dict)
        return loss, predictions

    def eval(self, sess, batch):
        """
        evaluate model method
        :param sess: session object of tensorflow
        :param batch: eval batch data
        :return: loss, predict result
        """
        feed_dict = {self.support: batch["support"],
                     self.queries: batch["queries"],
                     self.labels: batch["labels"],
                     self.keep_prob: 1.0}

        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def infer(self, sess, batch):
        """
        infer model method
        :param sess:
        :param batch:
        :return: predict result
        """
        feed_dict = {self.support: batch["support"],
                     self.queries: batch["queries"],
                     self.keep_prob: 1.0}

        predict = sess.run(self.predictions, feed_dict=feed_dict)

        return predict
