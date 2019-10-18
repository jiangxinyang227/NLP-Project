import tensorflow as tf


class CharRNNModel(object):

    def __init__(self, config, vocab_size=None, word_vectors=None, is_training=True):
        """
        文本分类的基类，提供了各种属性和训练，验证，测试的方法
        :param config: 模型的配置参数
        :param vocab_size: 当不提供词向量的时候需要vocab_size来初始化词向量
        :param word_vectors：预训练的词向量，word_vectors 和 vocab_size必须有一个不为None
        """
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors
        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")  # 数据输入
        self.labels = tf.placeholder(tf.int32, [None, None], name="labels")  # 标签
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout

        self.batch_size = config["batch_size"]
        self.sequence_length = config["sequence_length"]

        if is_training is False:
            self.batch_size, self.sequence_length = 1, 1

        self.initial_state = None
        self.final_state = None

        self.build_model()
        self.init_saver()

    def build_model(self):
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
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs)

        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.config["hidden_size"], self.keep_prob) for _ in range(self.config["num_layers"])]
            )
            self.initial_state = cell.zero_state(self.batch_size, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            outputs, self.final_state = tf.nn.dynamic_rnn(cell, embedded_words, initial_state=self.initial_state)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(outputs, 1)
            x = tf.reshape(seq_output, [-1, self.config["hidden_size"]])

        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[self.config["hidden_size"], self.vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[self.vocab_size]), name="output_b")
            self.logits = tf.matmul(x, output_w) + output_b
            self.predictions = self.get_predictions()

        self.loss = self.cal_loss()
        self.train_op, self.summary_op = self.get_train_op()

    def cal_loss(self):
        """
        计算损失，支持二分类和多分类
        :return:
        """
        labels = tf.reshape(self.labels, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)
        loss = tf.reduce_mean(losses)
        return loss

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
        if self.config["optimization"] == "momentum":
            optimizer = tf.train.MomentumOptimizer(self.config["learning_rate"], 0.9)
        if self.config["optimization"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.config["learning_rate"])
        return optimizer

    def get_train_op(self):
        """
        获得训练的入口
        :return:
        """
        # 定义优化器
        optimizer = self.get_optimizer()

        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 对梯度进行梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
        tf.summary.scalar("loss", self.loss)
        summary_op = tf.summary.merge_all()
        return train_op, summary_op

    def get_predictions(self):
        """
        得到预测结果，因为在预测的时候会做补全，因此在预测的时候做mask处理
        :return:
        """
        predictions = tf.nn.softmax(self.logits, name="predictions")
        return predictions

    def init_saver(self):
        """
        初始化saver对象
        :return:
        """
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, state, dropout_prob):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :param state: 更新后的lstm的初始状态
        :param dropout_prob: dropout比例
        :return: 损失和预测结果
        """

        feed_dict = {self.inputs: batch["inputs"],
                     self.labels: batch["labels"],
                     self.initial_state: state,
                     self.keep_prob: dropout_prob}

        # 训练模型
        _, summary_op, loss, final_state = sess.run([self.train_op, self.summary_op, self.loss, self.final_state], feed_dict=feed_dict)
        return summary_op, loss, final_state

    def eval(self, sess, batch, state):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :param state: 上一状态
        :return: 损失和预测结果
        """
        feed_dict = {self.inputs: batch["inputs"],
                     self.labels: batch["labels"],
                     self.initial_state: state,
                     self.keep_prob: 1.0}

        summary_op, loss, final_state = sess.run([self.summary_op, self.loss, self.final_state], feed_dict=feed_dict)
        return summary_op, loss, final_state

    def sample(self, sess, start, state):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param start: 启动词
        :param state:
        :return: 预测结果
        """
        feed_dict = {self.inputs: start,
                     self.initial_state: state,
                     self.keep_prob: 1.0}

        predict, state = sess.run([self.predictions, self.final_state], feed_dict=feed_dict)

        return predict, state



