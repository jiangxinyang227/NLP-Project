"""
prototypical network model for few shot learning
"""

import tensorflow as tf


class PrototypicalModel(object):
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
                                              initializer=tf.contrib.layers.xavier_initializer())

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

        with tf.name_scope("final_step_output"):
            # [num_classes, num_support, embedding_size]
            support_final_output = tf.reshape(support_embedded_reshape[:, -1, :],
                                              [self.config["num_classes"],
                                               self.config["num_support"],
                                               -1])

            # computing class vector means. dimension: [num_classes, embedding_size]
            support_class_output = tf.reduce_mean(support_final_output, axis=1)
            # [num_classes * num_queries, embedding_size]
            queries_final_output = queries_embedded[:, -1, :]

        # ---------------------------------------------------------------------------------------
        # squared Euclidean distance + cross entropy loss function
        # ---------------------------------------------------------------------------------------
        with tf.name_scope("euclidean_distance"):
            # expand dimension. [nun_classes * num_queries, num_classes, embedding_size]
            support_class_output_expand = tf.tile(tf.expand_dims(support_class_output, axis=0),
                                                  (self.config["num_classes"] * self.config["num_queries"], 1, 1))
            # expand dimension. dimension same as above
            queries_output_expand = tf.tile(tf.expand_dims(queries_final_output, axis=1),
                                            (1, self.config["num_classes"], 1))

            # distance between queries set and class in support set. [num_classes * num_queries, num_classes]
            distance = -tf.reduce_mean(tf.square(support_class_output_expand - queries_output_expand),
                                       axis=2, name="distance")
            self.predictions = tf.argmax(distance, axis=-1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=distance, labels=self.labels)
            self.loss = tf.reduce_mean(losses, name="loss")

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
        feed_dict = {self.support: batch["support"],
                     self.queries: batch["queries"],
                     self.labels: batch["labels"],
                     self.keep_prob: dropout_prob}

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
