import tensorflow as tf


class BaseModel(object):
    def __init__(self, config):
        self.config = config

        # 定义模型的placeholder, 也就是喂给feed_dict的参数
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 传入一个带维度的标量的正确方式，用空列表代入
        self.sample_prob = tf.placeholder(tf.float32, [], name="sample_prob")

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        # 根据目标序列长度，选出其中最大值，然后使用该值构建序列长度的mask标志。用一个sequence_mask的例子来说明起作用
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.target_mask = tf.sequence_mask(self.decoder_targets_length,
                                            self.max_target_sequence_length,
                                            dtype=tf.float32,
                                            name='target_masks')

        self.loss = 0.0  # 损失
        self.train_op = None  # 训练入口
        self.summary_op = None
        self.logits = None  # 模型最后一层的输出
        self.predictions = None  # 预测结果
        self.saver = None  # 保存为ckpt模型的对象
        self.learning_rate = self.config["learning_rate"]

    def cal_loss(self):
        """
        计算损失
        :return:
        """
        loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                targets=self.decoder_targets,
                                                weights=self.target_mask)
        if self.config["use_antilm"]:
            print("use_antilm")
            logits_softmax = tf.nn.softmax(self.logits, axis=-1)
            logits_argmax_prob = tf.reduce_max(logits_softmax, axis=-1)
            log_logits_argmax = tf.log1p(logits_argmax_prob)
            lm_loss = tf.reduce_mean(tf.reduce_sum(log_logits_argmax, axis=-1))
            loss += 10 * lm_loss
        return loss

    def get_optimizer(self):
        """
        获得优化器
        :return:
        """
        optimizer = None
        if self.config["optimization"] == "adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        if self.config["optimization"] == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        if self.config["optimization"] == "momentum":
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        if self.config["optimization"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer

    def get_train_op(self):
        """
        获得训练的入口
        :return:
        """
        # 定义优化器
        optimizer = self.get_optimizer()

        trainable_params = tf.trainable_variables()
        # for param in trainable_params:
        #     tf.summary.histogram(param.name, param)

        gradients = tf.gradients(self.loss, trainable_params)
        # for gradient in gradients:
        #     tf.summary.histogram(gradient.name, gradient)

        # 对梯度进行梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
        # for clip_grad in clip_gradients:
        #     tf.summary.histogram(clip_grad.name, clip_grad)

        tf.summary.scalar("loss", self.loss)
        summary_op = tf.summary.merge_all()

        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
        return train_op, summary_op

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

    def train(self, sess, batch, dropout_prob, sample_prob=0.0):
        # 对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据

        feed_dict = {self.encoder_inputs: batch["questions"],
                     self.decoder_targets: batch["responses"],
                     self.decoder_targets_length: batch["response_length"],
                     self.encoder_inputs_length: batch["question_length"],
                     self.keep_prob: dropout_prob,
                     self.sample_prob: sample_prob
                     }

        # 训练模型
        _, loss, predictions, summary = sess.run([self.train_op, self.loss, self.predictions, self.summary_op],
                                                 feed_dict=feed_dict)
        return loss, predictions, summary

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: batch["questions"],
                     self.encoder_inputs_length: batch["question_length"],
                     self.decoder_targets: batch["responses"],
                     self.decoder_targets_length: batch["response_length"],
                     self.keep_prob: 1.0,
                     self.sample_prob: 1.0
                     }
        loss, predictions, summary = sess.run([self.loss, self.predictions, self.summary_op], feed_dict=feed_dict)
        return loss, predictions, summary

    def infer(self, sess, batch):
        # infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.encoder_inputs: batch["questions"],
                     self.encoder_inputs_length: batch["question_length"],
                     self.keep_prob: 1.0,
                     }
        predictions = sess.run(self.predictions, feed_dict=feed_dict)

        return predictions