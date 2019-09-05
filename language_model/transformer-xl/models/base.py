import tensorflow as tf


class BaseModel(object):
    def __init__(self, config, vocab_size=None, word_vectors=None):
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

        self.initial_state = None
        self.final_state = None

        self.l2_loss = tf.constant(0.0)  # 定义l2损失
        self.loss = 0.0  # 损失
        self.train_op = None  # 训练入口
        self.summary_op = None
        self.logits = None  # 模型最后一层的输出
        self.predictions = None  # 预测结果
        self.saver = None  # 保存为ckpt模型的对象

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
        得到预测结果
        :return:
        """
        predictions = tf.nn.softmax(self.logits, name="predictions")
        return predictions

    def build_model(self):
        """
        创建模型
        :return:
        """
        raise NotImplementedError

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
        _, summary_op, loss, final_state = sess.run([self.train_op, self.summary_op, self.loss, self.final_state],
                                                    feed_dict=feed_dict)
        return summary_op, loss, final_state

    def eval(self, sess, batch, state):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :param state: 更新后的lstm的初始状态
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


