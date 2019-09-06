"""
采用text-cnn模型进行多意图分类
"""

import tensorflow as tf


class LstmDssmModel(object):
    def __init__(self, config, vocab_size, word_vectors):
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors
        self.batch_size = config["batch_size"]
        self.samples = config["neg_samples"] + 1

        self.query = tf.placeholder(tf.int32, [self.batch_size, None], name="query")
        self.sim_query = tf.placeholder(tf.int32, [self.batch_size * self.samples, None], name="sim_query")
        self.query_length = tf.placeholder(tf.int32, [self.batch_size], name="query_length")
        self.sim_length = tf.placeholder(tf.int32, [self.batch_size * self.samples], name="sim_length")

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout

        # 创建计算图
        self.model_structure()
        # 初始化保存状态
        self.init_saver()

    def model_structure(self):

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            if self.word_vectors is not None:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config["embedding_size"]],
                                              initializer=tf.contrib.layers.xavier_initializer())
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            embedded_query = tf.nn.embedding_lookup(embedding_w, self.query)
            embedded_sim_query = tf.nn.embedding_lookup(embedding_w, self.sim_query)

            # 定义两层双向LSTM的模型结构
            with tf.name_scope("Bi-LSTM"):
                for idx, hidden_size in enumerate(self.config["hidden_sizes"]):
                    with tf.name_scope("Bi-LSTM" + str(idx)):
                        # 定义前向LSTM结构
                        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                            tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                            output_keep_prob=self.keep_prob)
                        # 定义反向LSTM结构
                        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                            tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                            output_keep_prob=self.keep_prob)

                        # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                        # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                        # fw和bw的hidden_size一样
                        # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                        query_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                          embedded_query, dtype=tf.float32,
                                                                          scope="query" + str(idx))
                        sim_query_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                              embedded_sim_query, dtype=tf.float32,
                                                                              scope="sim_query" + str(idx))

                        # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                        embedded_query = tf.concat(query_output, 2)
                        embedded_sim_query = tf.concat(sim_query_output, 2)

        with tf.name_scope("final_step_output"):
            query_final_output = self.get_final_step_output(self.query_length,
                                                            self.batch_size,
                                                            embedded_query)
            sim_query_final_output = self.get_final_step_output(self.sim_length,
                                                                self.batch_size * self.samples,
                                                                embedded_sim_query)

        with tf.name_scope("reshape_sim"):
            # batch_size 个tensor：[neg_samples + 1, hidden_size * 2]
            split_sim_query_final_output = tf.split(sim_query_final_output,
                                                    [self.samples] * self.batch_size,
                                                    axis=0)
            # batch_size 个tensor: [1, neg_samples + 1, hidden-size * 2]
            expand_concat_sim_query = [tf.expand_dims(tensor, 0) for tensor in split_sim_query_final_output]
            # [batch_size, neg_samples + 1, hidden_size * 2]
            reshape_sim_query_final_output = tf.concat(expand_concat_sim_query, axis=0)

        with tf.name_scope("cosine_similarity"):
            # [batch_size, 1, hidden_size * 2]
            expand_query_final_output = tf.expand_dims(query_final_output, 1)
            # [batch_size, 1]
            query_norm = tf.sqrt(tf.reduce_sum(tf.square(expand_query_final_output), -1))
            # [batch_size, n_samples + 1]
            sim_query_norm = tf.sqrt(tf.reduce_sum(tf.square(reshape_sim_query_final_output), -1))
            # [batch_size, n_samples + 1]
            dot = tf.reduce_sum(tf.multiply(expand_query_final_output, reshape_sim_query_final_output), axis=-1)

            # [batch_size, n_samples + 1]
            norm = query_norm * sim_query_norm

            self.similarity = tf.div(dot, norm, name="similarity")
            self.predictions = tf.argmax(self.similarity, -1, name="predictions")

        with tf.name_scope("loss"):
            if self.config["neg_samples"] == 1:
                pos_similarity = tf.reshape(tf.slice(self.similarity, [0, 0], [self.batch_size, 1]),
                                            [self.batch_size])
                neg_similarity = tf.reshape(tf.slice(self.similarity,
                                                     [0, 1],
                                                     [self.batch_size, self.config["neg_samples"]]),
                                            [self.batch_size])
                distance = self.config["margin"] - pos_similarity + neg_similarity
                zeros = tf.zeros_like(distance, dtype=tf.float32)
                cond = (distance >= zeros)
                losses = tf.where(cond, distance, zeros)
                self.loss = tf.reduce_mean(losses, name="loss")
            else:
                pos_similarity = tf.exp(tf.reshape(tf.slice(self.similarity, [0, 0], [self.batch_size, 1]),
                                            [self.batch_size]))
                neg_similarity = tf.exp(
                    tf.slice(self.similarity, [0, 1], [self.batch_size, self.config["neg_samples"]])
                )
                norm_seg_similarity = tf.reduce_sum(neg_similarity, axis=-1)
                pos_prob = tf.div(pos_similarity, norm_seg_similarity)
                self.loss = tf.reduce_mean(-tf.log(pos_prob), name="loss")

        with tf.name_scope("train_op"):
            # 定义优化器
            optimizer = self.get_optimizer()

            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            # 对梯度进行梯度截断
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    @staticmethod
    def get_final_step_output(sequence_length, batch_size, output):
        """

        :param sequence_length:
        :param batch_size:
        :param output:
        :return:
        """
        # 取出每一个样本最后时间步对应的索引位置
        col = sequence_length - 1
        # 给出batch中每个样本的索引编号
        row = tf.range(batch_size)
        # 给出batch样本的[[样本编号，样本的最后时间步对应的索引位置], .....]
        index = tf.unstack(tf.stack([row, col], axis=0), axis=1)
        final_output = tf.gather_nd(output, index)
        return final_output

    def get_optimizer(self):
        """
        获得优化器
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
        初始化saver对象
        :return:
        """
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, dropout_prob):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :param dropout_prob: dropout比例
        :return: 损失和预测结果
        """
        feed_dict = {self.query: batch["query"],
                     self.sim_query: batch["sim"],
                     self.query_length: batch["query_length"],
                     self.sim_length: batch["sim_length"],
                     self.keep_prob: dropout_prob}

        # 训练模型
        _, loss, predictions = sess.run([self.train_op, self.loss, self.predictions],
                                        feed_dict=feed_dict)
        return loss, predictions

    def eval(self, sess, batch):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        feed_dict = {self.query: batch["query"],
                     self.sim_query: batch["sim"],
                     self.query_length: batch["query_length"],
                     self.sim_length: batch["sim_length"],
                     self.keep_prob: 1.0}

        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def infer(self, sess, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: 待预测的数据
        :return: 预测结果
        """
        feed_dict = {self.query: batch["query"],
                     self.sim_query: batch["sim"],
                     self.query_length: batch["query_length"],
                     self.sim_length: batch["sim_length"],
                     self.keep_prob: 1.0}

        predict = sess.run(self.predictions, feed_dict=feed_dict)

        return predict
