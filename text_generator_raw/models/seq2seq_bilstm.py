import tensorflow as tf


class Seq2SeqBiLstm(object):
    def __init__(self, config, vocab_size, word_vectors=None):
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors  # 词向量得到的词嵌入矩阵
        self.embedding_size = config["embedding_size"]
        self.encoder_hidden_sizes = config["encoder_hidden_sizes"]
        self.decoder_hidden_sizes = config["decoder_hidden_sizes"]
        self.num_blocks = config["num_blocks"]  # transformer block的数量
        self.batch_size = config["batch_size"]
        self.keep_prob = config["keep_prob"]
        self.hidden_size = config["hidden_size"]  # feed forward层的隐层大小
        self.learning_rate = config["learning_rate"]  # 学习速率
        self.epsilon = config["lr_epsilon"]  # layer normalization 中除数中的极小值
        self.smooth_rate = config["smooth_rate"]  # smooth label的比例
        self.warmup_step = config["warmup_step"]  # 学习速率预热的步数
        self.decode_step = config["decode_step"]  # 解码的最大长度

        # 编码和解码共享embedding矩阵，若是不同语言的，如机器翻译，就各定义一个embedding矩阵
        self.embedding_matrix = self._get_embedding_matrix()

        self.pad_token = 0
        self.start_token = 2

    def _multi_rnn_cell(self, hidden_sizes):
        """
        创建多层cell
        :return:
        """

        def get_lstm_cell(hidden_size, keep_prob):
            """
            创建单个cell ，并添加dropout
            :param hidden_size:
            :param keep_prob:
            :return:
            """
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True,
                                                initializer=tf.orthogonal_initializer())

            drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)

            return drop_cell

        # 创建多层cell
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(hidden_size, self.keep_prob)
                                                  for hidden_size in hidden_sizes])

        return multi_cell

    def encode(self, encoder_inputs, training=True):
        """
        定义decode部分
        :param encoder_inputs:
        :param training:
        :return:
        """
        with tf.name_scope("encoder"):

            # embedding 层
            embeddings = self._get_embedding_matrix()
            embedded_words = tf.nn.embedding_lookup(embeddings, encoder_inputs)

            states = []
            with tf.name_scope("Bi-LSTM"):
                for idx, hidden_size in enumerate(self.encoder_hidden_sizes):
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
                        # current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                        outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                                 embedded_words, dtype=tf.float32,
                                                                                 scope="bi-lstm" + str(idx))
                        # 对双向输出的状态进行拼接合并
                        fw_state, bw_state = current_state
                        fw_state_c, fw_state_h = fw_state
                        bw_state_c, bw_state_h = bw_state
                        state_c = tf.concat([fw_state_c, bw_state_c], -1)
                        state_h = tf.concat([bw_state_c, bw_state_h], -1)
                        state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
                        states.append(state)
                        # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                        embedded_words = tf.concat(outputs, 2)

        # 对双向输出的状态进行拼接合并

        tuple_states = tuple(states)
        return embedded_words, tuple_states, embeddings

    def decode(self, encoder_inputs, decoder_inputs, encoder_outputs, training=True):
        """
        decode部分
        :param encoder_inputs:
        :param decoder_inputs:
        :param encoder_outputs:
        :param training:
        :return:
        """
        with tf.name_scope("decoder"):
            # embedding
            embedded_word = tf.nn.embedding_lookup(self.embedding_matrix, decoder_inputs)
            embedded_word = tf.nn.dropout(embedded_word, self.keep_prob)

            multi_cells = []
            for idx, hidden_size in enumerate(self.decoder_hidden_sizes):
                with tf.name_scope("LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)
                    multi_cells.append(lstm_cell)

            cell = tf.nn.rnn_cell.MultiRNNCell(multi_cells)

            initial_state = cell.zero_state(self.batch_size, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            outputs, final_state = tf.nn.dynamic_rnn(cell, embedded_word, initial_state=initial_state)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(outputs, 1)
            x = tf.reshape(seq_output, [-1, self.config["hidden_size"]])

    def attention(self):
        pass

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

    def train(self, encoder_inputs, decoder_inputs, y_true, training=True):
        """
        预测模型
        :param encoder_inputs:
        :param decoder_inputs:
        :param y_true:
        :param training:
        :return:
        """
        # forward
        encoder_outputs = self.encode(encoder_inputs, training=training)
        logits, y_pred = self.decode(encoder_inputs, decoder_inputs, encoder_outputs, training=training)

        # train scheme
        # 对真实的标签做平滑处理
        y_ = self.label_smoothing(tf.one_hot(y_true, depth=self.vocab_size))
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        # 取出非0部分，即非padding部分
        non_padding = tf.to_float(tf.not_equal(y_true, self.pad_token))
        loss = tf.reduce_sum(losses * non_padding) / (tf.reduce_sum(non_padding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        # 动态的修改初始的学习速率
        lr = self.noam_scheme(self.learning_rate, global_step, self.warmup_step)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, encoder_inputs, decoder_inputs, y_true, training=False):
        """
        验证模型
        :param encoder_inputs:
        :param decoder_inputs:
        :param y_true:
        :param training:
        :return:
        """
        # forward
        encoder_outputs = self.encode(encoder_inputs, training=training)
        logits, y_pred = self.decode(encoder_inputs, decoder_inputs, encoder_outputs, training=training)

        # train scheme
        # 对真实的标签做平滑处理
        y_ = self.label_smoothing(tf.one_hot(y_true, depth=self.vocab_size))
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        # 取出非0部分，即非padding部分
        non_padding = tf.to_float(tf.not_equal(y_true, self.pad_token))
        loss = tf.reduce_sum(losses * non_padding) / (tf.reduce_sum(non_padding) + 1e-7)

        tf.summary.scalar("loss", loss)

        summaries = tf.summary.merge_all()

        return loss, summaries

    def infer(self, encoder_inputs):
        """
        预测部分
        :param encoder_inputs:
        :return:
        """

        # 在验证时，没有真实的decoder_inputs，此时需要构造一个初始的输入，初始的输入用初始符
        decoder_inputs = tf.ones((tf.shape(encoder_inputs)[0], 1), tf.int32) * self.start_token

        encoder_outputs = self.encode(encoder_inputs)

        for _ in range(self.decode_step):
            logits, y_pred = self.decode(encoder_inputs, decoder_inputs, encoder_outputs)
            if tf.reduce_sum(y_pred, 1) == self.pad_token: break

            decoder_inputs = tf.concat((decoder_inputs, y_pred), 1)

        return y_pred


