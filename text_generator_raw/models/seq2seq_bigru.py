import tensorflow as tf
from .base import BaseModel


class Seq2SeqGru(BaseModel):
    def __init__(self, config, vocab_size, word_vectors=None):
        super(Seq2SeqGru, self).__init__(config)
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors  # 词向量得到的词嵌入矩阵
        self.embedding_size = config["embedding_size"]
        self.encoder_hidden_sizes = config["encoder_hidden_sizes"]
        self.decoder_hidden_sizes = config["decoder_hidden_sizes"]
        self.keep_prob = config["keep_prob"]
        self.learning_rate = config["learning_rate"]  # 学习速率
        self.smooth_rate = config["smooth_rate"]  # smooth label的比例
        self.warmup_step = config["warmup_step"]  # 学习速率预热的步数
        self.decode_step = config["decode_step"]  # 解码的最大长度

        # 得到encoder decoder batch 的最大长度
        self.encoder_max_len = tf.reduce_max(self.encoder_length, name="encoder_max_len")
        self.decoder_max_len = tf.reduce_max(self.decoder_length, name="decoder_max_len")

        self.pad_token = 0

        # 编码和解码共享embedding矩阵，若是不同语言的，如机器翻译，就各定义一个embedding矩阵
        self.embedding_matrix = self._get_embedding_matrix()
        self.built_model()
        self.init_saver()

    def _get_embedding_matrix(self, zero_pad=True):
        """
        词嵌入层
        :param zero_pad:
        :return:
        """
        with tf.variable_scope("embedding"):
            embeddings = tf.get_variable('embedding_w',
                                         dtype=tf.float32,
                                         shape=(self.vocab_size, self.embedding_size),
                                         initializer=tf.contrib.layers.xavier_initializer())
            if zero_pad:
                embeddings = tf.concat((tf.zeros(shape=[1, self.embedding_size]),
                                        embeddings[1:, :]), 0)

        return embeddings

    def encoder(self, encoder_inputs, encoder_length, encoder_max_len):
        """
        定义encoder层
        :param encoder_inputs:
        :param encoder_length:
        :param encoder_max_len:
        :return:
        """
        with tf.name_scope("encoder"):
            # 词嵌入层，并加上位置向量
            embedded_word = tf.nn.embedding_lookup(self.embedding_matrix, encoder_inputs)
            embedded_word = tf.nn.dropout(embedded_word, self.keep_prob)
            # 得到encoder mask矩阵
            encoder_mask = tf.sequence_mask(encoder_length, encoder_max_len, dtype=tf.float32, name="encoder_mask")
            input_size = self.embedding_size
            for idx, hidden_size in enumerate(self.encoder_hidden_sizes):
                with tf.name_scope("gru_{}".format(idx)):
                    initial_state = tf.zeros([self.batch_size, hidden_size], dtype=tf.float32, name="initial_state")
                    embedded_word, state = self.encoder_layer(input_size, hidden_size,
                                                              initial_state, embedded_word, encoder_mask)
                input_size = hidden_size

        return embedded_word, encoder_mask, state

    @staticmethod
    def encoder_layer(input_size, hidden_size, initial_state, inputs, input_mask):
        """
        单向gru结构层
        :param input_size: 输入的大小
        :param hidden_size: 隐层的大小
        :param initial_state: 初始状态
        :param inputs: 输入
        :param input_mask: 输入的mask
        :return: 输出，最终的状态
        """
        # 创建一个gru cell对象
        cell = GRUCell(input_size, hidden_size)
        outputs = []
        state = initial_state
        """
        按照序列从前到后遍历，将每一时间的输出添加到outputs中，tf.unstack是解包函数，可以按照序列维度解包成一个个的
        [batch_size, input_size]的tensor，input_mask是为了对padding部分做mask处理
        """
        for time, (embedded, mask) in enumerate(zip(tf.unstack(tf.transpose(inputs, [1, 0, 2])),
                                                    tf.unstack(tf.transpose(input_mask, [1, 0, 2])))):
            output, state = cell(embedded, state)
            output = tf.expand_dims(mask, 1) * output
            outputs.append(output)
        outputs = tf.transpose(tf.convert_to_tensor(outputs, dtype=tf.float32), [1, 0, 2])
        return outputs, state

    @staticmethod
    def bidirectional_encoder_layer(input_size, hidden_size, initial_fw_state, initial_bw_state, fw_inputs, bw_inputs,
                                    fw_input_mask, bw_input_mask):
        """
        双向gru结构层
        :param input_size: 输入的大小
        :param hidden_size: 隐层的大小
        :param initial_fw_state: 前向的初始状态
        :param initial_bw_state: 后向的初始状态
        :param fw_inputs: 前向输入
        :param bw_inputs: 反向输入
        :param fw_input_mask: 前向输入的mask
        :param bw_input_mask: 反响输入的mask
        :return:
        """
        cell = GRUCell(input_size, hidden_size)
        fw_outputs = []
        bw_outputs = []
        fw_state = initial_fw_state
        bw_state = initial_bw_state
        for fw_time, (fw_embedded, fw_mask) in enumerate(zip(tf.unstack(tf.transpose(fw_inputs, [1, 0, 2])),
                                                             tf.unstack(tf.transpose(fw_input_mask, [1, 0, 2])))):
            fw_output, fw_state = cell(fw_embedded, fw_state)
            fw_output = tf.expand_dims(fw_mask, 1) * fw_output
            fw_outputs.append(fw_output)

        for bw_time, (bw_embedded, bw_mask) in enumerate(zip(tf.unstack(tf.transpose(bw_inputs, [1, 0, 2])),
                                                             tf.unstack(tf.transpose(bw_input_mask, [1, 0, 2])))):
            bw_output, bw_state = cell(bw_embedded, bw_state)
            bw_output = tf.expand_dims(bw_mask, 1) * bw_output
            bw_outputs.append(bw_output)

        fw_outputs = tf.transpose(tf.convert_to_tensor(fw_outputs, dtype=tf.float32), [1, 0, 2])
        bw_outputs = tf.transpose(tf.convert_to_tensor(bw_outputs, dtype=tf.float32), [1, 0, 2])

        return (fw_outputs, bw_outputs), (fw_state, bw_state)

    def decoder(self, encoder_outputs, encoder_mask, decoder_inputs, decoder_length, decoder_max_len,
                encoder_final_state=None, encoder_length=None, use_attention=True):
        """
        定义decoder
        :param encoder_outputs:
        :param encoder_mask:
        :param decoder_inputs:
        :param decoder_length:
        :param decoder_max_len:
        :param encoder_final_state:
        :param encoder_length:
        :param use_attention:
        :return:
        """
        with tf.name_scope("decoder"):
            # 词嵌入层，并加上位置向量
            embedded_word = tf.nn.embedding_lookup(self.embedding_matrix, decoder_inputs)
            embedded_word = tf.nn.dropout(embedded_word, self.keep_prob)
            # 得到encoder mask矩阵
            decoder_mask = tf.sequence_mask(decoder_length, decoder_max_len, dtype=tf.float32, name="encoder_mask")
            input_size = self.embedding_size
            for idx, hidden_size in enumerate(self.decoder_hidden_sizes):
                with tf.name_scope("gru_{}".format(idx)):
                    if encoder_final_state:
                        initial_state = encoder_final_state
                    else:
                        initial_state = tf.zeros([self.batch_size, hidden_size], dtype=tf.float32, name="initial_state")
                    embedded_word, state = self.decoder_layer(input_size, hidden_size, initial_state, encoder_outputs,
                                                              encoder_mask, decoder_inputs, decoder_mask,
                                                              encoder_length, use_attention)
                input_size = hidden_size

        return embedded_word

    def decoder_layer(self, input_size, hidden_size, initial_state, encoder_outputs, encoder_mask,
                      decoder_inputs, decoder_mask, encoder_length=None, use_attention=True):
        """
        定义decoder层
        :param input_size: 输入大小
        :param hidden_size: 隐层大小
        :param initial_state: 初始状态
        :param encoder_outputs: encoder的输出
        :param encoder_mask: encoder的mask
        :param decoder_inputs: decoder的输入
        :param decoder_mask: decoder的mask
        :param encoder_length: encoder的实际长度
        :param use_attention: 是否用attention
        :return: 输出和结束状态
        """

        if not use_attention and not encoder_length:
            raise ("if not use attention, please be sure encoder_length is not None")

        # 创建一个gru cell对象
        cell = GRUCell(input_size, hidden_size, use_attention=use_attention)
        outputs = []
        state = initial_state

        # 因为encoder有部分是padding的，因此再取最后时间步的输出作为整个encoder的编码时，
        # 我们需要取非padding部分的最后时间步的输出
        col = encoder_length - 1
        row = tf.range(self.batch_size)
        index = tf.unstack(tf.stack([row, col], axis=0), axis=1)
        encoder_final_output = tf.gather_nd(encoder_outputs, index)

        for time, (embedded, mask) in enumerate(zip(tf.unstack(tf.transpose(decoder_inputs, [1, 0, 2])),
                                                    tf.unstack(tf.transpose(decoder_mask, [1, 0, 2])))):
            if not use_attention:
                c = encoder_final_output
            else:
                c = self._attention(embedded, encoder_outputs, encoder_mask)
            output, state = cell(embedded, state, c)
            output = tf.expand_dims(mask, 1) * output
            outputs.append(output)
        outputs = tf.transpose(tf.convert_to_tensor(outputs, dtype=tf.float32), [1, 0, 2])
        return outputs, state

    @staticmethod
    def _attention(embedded, encoder_outputs, encoder_mask):
        """
        encoder decoder之间的attention机制
        :param embedded:
        :param encoder_outputs:
        :param encoder_mask:
        :return:
        """
        embedded = tf.expand_dims(embedded, -1)
        similarity = tf.matmul(encoder_outputs, embedded)
        weight = tf.nn.softmax(similarity, axis=-1)
        weight *= encoder_mask
        # 注，在这里的weight必须放在前面，表示对encoder_outputs中做行相加，即沿着序列做加权和
        c = tf.matmul(weight, encoder_outputs)
        return c

    def label_smoothing(self, inputs):
        """
        标签平滑，将原本的one-hot真实标签向量变成一个不含0的标签向量
        :param inputs:
        :return:
        """
        V = inputs.get_shape().as_list()[-1]
        return ((1 - self.smooth_rate) * inputs) + (self.smooth_rate / V)

    @staticmethod
    def noam_scheme(init_lr, global_step, warmup_steps=4000.):
        """
        采用预热学习速率的方法来训练模型
        :param init_lr:
        :param global_step:
        :param warmup_steps:
        :return:
        """
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    def train_method(self):
        self.predictions = tf.to_int32(tf.argmax(self.logits, axis=-1))
        # train scheme
        # 对真实的标签做平滑处理
        y_ = self.label_smoothing(tf.one_hot(self.decoder_outputs, depth=self.vocab_size))
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_)
        # 取出非0部分，即非padding部分
        non_padding = tf.to_float(tf.not_equal(self.decoder_outputs, self.pad_token))
        self.loss = tf.reduce_sum(losses * non_padding) / (tf.reduce_sum(non_padding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        # 动态的修改初始的学习速率
        lr = self.noam_scheme(self.learning_rate, global_step, self.warmup_step)
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

    def built_model(self):
        encoder_outputs, encoder_mask, final_state = self.encoder(self.encoder_inputs,
                                                                  self.encoder_length,
                                                                  self.encoder_max_len)
        self.logits = self.decoder(encoder_outputs, encoder_mask, self.decoder_inputs, self.decoder_length,
                                   self.decoder_max_len, final_state, use_attention=True)

        self.train_method()

    def train(self, sess, batch, keep_prob):
        """
        对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        :param sess:
        :param batch:
        :param keep_prob:
        :return:
        """

        feed_dict = {self.encoder_inputs: batch["encoder_inputs"],
                     self.decoder_inputs: batch["decoder_inputs"],
                     self.decoder_outputs: batch["decoder_outputs"],
                     self.encoder_length: batch["encoder_length"],
                     self.decoder_length: batch["decoder_length"],
                     self.keep_prob: keep_prob
                     }

        # 训练模型
        _, summary, loss, predictions = sess.run([self.train_op, self.summary_op, self.loss, self.predictions],
                                                 feed_dict=feed_dict)
        return summary, loss, predictions

    def eval(self, sess, batch):
        """
        对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        :param sess:
        :param batch:
        :return:
        """
        feed_dict = {self.encoder_inputs: batch["encoder_inputs"],
                     self.decoder_inputs: batch["decoder_inputs"],
                     self.decoder_outputs: batch["decoder_outputs"],
                     self.encoder_length: batch["encoder_length"],
                     self.decoder_length: batch["decoder_length"],
                     self.keep_prob: 1.0
                     }
        summary, loss, predictions = sess.run([self.summary_op, self.loss, self.predictions], feed_dict=feed_dict)
        return summary, loss, predictions

class GRUCell(object):
    """Gated Recurrent Unit cell, batch incorporated.

    Based on [3].
    GRU's output is equal to state.
    The arg wt_attention is associated with arg context.
    """

    def __init__(self, input_size, hidden_size, use_attention=False, activation=tf.nn.tanh):
        # Initialize parameters
        self._activation = activation  # 激活函数
        self._input_size = input_size  # 输入的大小
        self._hidden_size = hidden_size  # 隐层的大小
        self._use_attention = use_attention  # 是否选择attention，decoder的时候设为True
        # 定义更新门的权重系数
        self._W_r_x = tf.Variable(
            tf.random_uniform([self._input_size, self._hidden_size], -0.1, 0.1, tf.float32), name="w_r_x")
        self._W_r_h = tf.Variable(
            tf.random_uniform([self._hidden_size, self._hidden_size], -0.1, 0.1, tf.float32), name="w_r_h")
        # 定义重置门的权重系数
        self._W_z_x = tf.Variable(
            tf.random_uniform([self._input_size, self._hidden_size], -0.1, 0.1, tf.float32), name="w_z_x")
        self._W_z_h = tf.Variable(
            tf.random_uniform([self._hidden_size, self._hidden_size], -0.1, 0.1, tf.float32), name="w_z_h")
        # 定义w_h的权重系数
        self._W_h_x = tf.Variable(
            tf.random_uniform([self._input_size, self._hidden_size], -0.1, 0.1, tf.float32), name="w_h_x")
        self._W_h_h = tf.Variable(
            tf.random_uniform([self._hidden_size, self._hidden_size], -0.1, 0.1, tf.float32), name="w_h_h")

        # 定义更新门的偏置
        self._B_r = tf.Variable(tf.ones([self._hidden_size], tf.float32), name="b_r")
        # 定义重置门的偏置
        self._B_z = tf.Variable(tf.ones([self._hidden_size], tf.float32), name="b_z")
        # 定义b_h的偏置
        self._B_h = tf.Variable(tf.ones([self._hidden_size], tf.float32), name="b_h")

        if self._use_attention:
            # 更新门中对attention的结果的权重系数
            self._W_r_c = tf.Variable(
                tf.random_uniform([self._hidden_size, self._hidden_size], -0.1, 0.1, tf.float32), name="w_r_c")
            # 重置门中对attention的结果的权重系数
            self._W_z_c = tf.Variable(
                tf.random_uniform([self._hidden_size, self._hidden_size], -0.1, 0.1, tf.float32), name="w_z_c")
            # h中对attention的结果的权重系数
            self._W_h_c = tf.Variable(
                tf.random_uniform([self._hidden_size, self._hidden_size], -0.1, 0.1, tf.float32), name="w_h_c")

    def __call__(self, inputs, state, context=None):
        """
        传入输入，隐层状态，attention的结果，输出下一状态
        :param inputs:
        :param state:
        :param context:
        :return:
        """
        if not self._use_attention:
            # 得到更新门
            r = tf.sigmoid(tf.matmul(tf.concat([inputs, state], axis=1),
                                     tf.concat([self._W_r_x, self._W_r_h], axis=0)) + self._B_r, name="r")
            # 得到重置门
            z = tf.sigmoid(tf.matmul(tf.concat([inputs, state], axis=1),
                                     tf.concat([self._W_z_x, self._W_z_h], axis=0)) + self._B_z, name="z")
        else:
            if context is None:
                raise ValueError("Attention mechanism used, while context vector is not received.")
            # 得到更新门
            r = tf.sigmoid(tf.matmul(tf.concat([inputs, state, context], axis=1),
                                     tf.concat([self._W_r_x, self._W_r_h, self._W_r_c], axis=0)) + self._B_r, name="r")
            # 得到重置门
            z = tf.sigmoid(tf.matmul(tf.concat([inputs, state, context], axis=1),
                                     tf.concat([self._W_z_x, self._W_z_h, self._W_z_c], axis=0)) + self._B_z, name="z")

        if not self._use_attention:
            h_hat = self._activation(tf.matmul(tf.concat([inputs, r * state], axis=1),
                                               tf.concat([self._W_h_x, self._W_h_h], axis=0)) + self._B_h, name="h_hat")
        else:
            h_hat = self._activation(
                tf.matmul(tf.concat(1, [inputs, r * state, context]),
                          tf.concat(0, [self._W_h_x, self._W_h_h, self._W_h_c])) + self._B_h, name="h_hat")
        h = z * state + (1 - z) * h_hat
        return h, h
