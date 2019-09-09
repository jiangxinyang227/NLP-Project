"""
采用text-cnn模型进行多意图分类
"""

import tensorflow as tf


class LstmClassifier(object):
    def __init__(self, config, vocab_size, word_vectors):
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors
        self.contents = tf.placeholder(tf.int32, [None, config["sequence_length"]], name="inputs")  # 数据输入
        self.labels = tf.placeholder(tf.int32, [None, config["num_aspects"]], name="labels")  # 标签
        self.aspect = tf.placeholder(tf.int32, [config["num_aspects"], None], name="aspect")
        self.aspect_length = tf.placeholder(tf.int32, [config["num_aspects"]], name="aspect_length")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout

        self.l2_loss = tf.constant(0.0)  # 定义l2损失

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
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.contents)
            embedded_aspect = tf.nn.embedding_lookup(embedding_w, self.aspect)

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
                        outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                                 embedded_words, dtype=tf.float32,
                                                                                 scope="bi-lstm" + str(idx))

                        # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                        embedded_words = tf.concat(outputs, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(embedded_words, 2, -1)

        with tf.name_scope("aspect"):
            aspect_representation = tf.reduce_mean(embedded_aspect, axis=1)
            aspect_w = tf.get_variable(
                name='aspect_w',
                shape=[self.config["embedding_size"], self.config["hidden_sizes"][-1] * 2],
                initializer=tf.random_uniform_initializer(-0.1, 0.1)
            )
            aspect_b = tf.get_variable(
                name='aspect_b',
                shape=[self.config["hidden_sizes"][-1] * 2],
                initializer=tf.zeros_initializer()
            )
            trans_aspect = tf.matmul(aspect_representation, aspect_w) + aspect_b

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            content_output = tf.concat(outputs, axis=-1)

            # 得到Attention的输出, [batch_size, num_aspects, hidden_size * 2]
            output = self._attention(content_output, trans_aspect)
            output_size = self.config["hidden_sizes"][-1] * 2

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            logits = tf.nn.xw_plus_b(tf.reshape(output, [-1, output_size]), output_w, output_b, name="logits")
            reshape_logits = tf.reshape(logits, [-1, self.config["num_aspects"], self.config["num_classes"]])
            self.predictions = tf.argmax(reshape_logits, axis=-1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=tf.reshape(self.labels, [-1]))
            self.loss = tf.reduce_mean(losses) + self.config["l2_reg_lambda"] * self.l2_loss

        with tf.name_scope("train_op"):
            # 定义优化器
            optimizer = self.get_optimizer()

            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            # 对梯度进行梯度截断
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    def _attention(self, content_output, embedded_aspect):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        encoder_size = self.config["hidden_sizes"][-1] * 2

        # 初始化一个权重向量，是可训练的参数, [attention_size, num_aspects]
        targets = tf.transpose(embedded_aspect, [1, 0])
        # attention_w = tf.get_variable("attention_w", shape=[encoder_size, attention_size],
        #                               initializer=tf.glorot_normal_initializer())

        # 对Bi-LSTM的输出用激活函数做非线性转换, [batch_size*sequence_length, attention_size]
        content_trans = tf.tanh(tf.reshape(content_output, [-1, encoder_size]))

        # [batch_size, sequence_length, num_aspects]
        weights = tf.reshape(tf.matmul(content_trans, targets),
                             [-1, self.config["sequence_length"], self.config["num_aspects"]])

        # 用softmax做归一化处理[batch_size, sequence_length, num_aspects]
        self.alpha = tf.nn.softmax(weights, axis=1)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作, [batch_size, encoder_size, num_aspects]
        r = tf.matmul(tf.transpose(content_output, [0, 2, 1]), self.alpha)

        # [batch_size, num_aspects, encoder_size]
        transpose_r = tf.transpose(r, [0, 2, 1])

        return transpose_r

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

    def train(self, sess, batch, aspect, aspect_length, dropout_prob):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :param aspect:
        :param aspect_length:
        :param dropout_prob: dropout比例
        :return: 损失和预测结果
        """
        feed_dict = {self.contents: batch["contents"],
                     self.labels: batch["labels"],
                     self.aspect: aspect,
                     self.aspect_length: aspect_length,
                     self.keep_prob: dropout_prob}

        # 训练模型
        _, loss, predictions = sess.run([self.train_op, self.loss, self.predictions],
                                        feed_dict=feed_dict)
        return loss, predictions

    def eval(self, sess, batch, aspect, aspect_length):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :param aspect:
        :param aspect_length:
        :return: 损失和预测结果
        """
        feed_dict = {self.contents: batch["contents"],
                     self.labels: batch["labels"],
                     self.aspect: aspect,
                     self.aspect_length: aspect_length,
                     self.keep_prob: 1.0}

        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def infer(self, sess, content):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param content: 待预测的数据
        :return: 预测结果
        """
        feed_dict = {self.contents: content,
                     self.keep_prob: 1.0}

        predict = sess.run(self.predictions, feed_dict=feed_dict)

        return predict
