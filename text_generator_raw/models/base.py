import tensorflow as tf


class BaseModel(object):
    def __init__(self, config):
        self.batch_size = config["batch_size"]

        # placeholder 值
        self.encoder_inputs = tf.placeholder(tf.int32, [self.batch_size, None], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(tf.int32, [self.batch_size, None], name="decoder_inputs")
        self.decoder_outputs = tf.placeholder(tf.int32, [self.batch_size, None], name="decoder_outputs")
        self.encoder_max_len = tf.placeholder(tf.int32, None, name="encoder_max_len")
        self.decoder_max_len = tf.placeholder(tf.int32, None, name="decoder_max_len")

        self.loss = 0.0  # 损失
        self.train_op = None  # 训练入口
        self.summary_op = None
        self.logits = None  # 模型最后一层的输出
        self.predictions = None  # 预测结果
        self.saver = None  # 保存为ckpt模型的对象

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

    def train(self, sess, batch, dropout_prob):
        # 对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据

        raise NotImplementedError

    def eval(self, sess, batch):
        raise NotImplementedError

    def infer(self, sess, batch):
        raise NotImplementedError


