# encoding=utf-8
import tensorflow as tf


class BiLSTMCRF(object):
    def __init__(self, embedded_chars, hidden_sizes, layers, dropout_rate, num_labels,
                 max_len, labels, sequence_lens, is_training):
        """
        构建Bi-LSTM + CRF结构
        :param embedded_chars:
        :param hidden_sizes:
        :param dropout_rate:
        :param num_labels:
        :param max_len:
        :param labels:
        :param sequence_lens:
        :param is_training:
        """
        self.hidden_sizes = hidden_sizes
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.embedded_chars = embedded_chars
        self.max_len = max_len
        self.num_labels = num_labels
        self.labels = labels
        self.sequence_lens = sequence_lens
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training

        self.l2_loss = tf.constant(0.0)

    def bi_lstms(self):
        """
        定义Bi-LSTM层，支持实现多层
        :return:
        """
        with tf.name_scope("embedding"):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, keep_prob=self.dropout_rate)

        with tf.name_scope("Bi-LSTM"):

            for idx, hidden_size in enumerate(self.hidden_sizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.dropout_rate)
                    # 定义反向LSTM结构
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.dropout_rate)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                    # fw和bw的hidden_size一样
                    # current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                             self.embedded_chars, dtype=tf.float32,
                                                                             scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                    self.embedded_chars = tf.concat(outputs, 2)

        output_size = self.hidden_sizes[-1] * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
        output = tf.reshape(self.embedded_chars, [-1, output_size])  # reshape成全连接层的输入维度

        return output, output_size

    def output_layer(self, output, output_size):
        """
        定义全连接输出层
        :param output:
        :param output_size:
        :return:
        """
        with tf.name_scope("output_layers"):
            for idx, layer in enumerate(self.layers):
                with tf.variable_scope("output_layer" + str(idx)):
                    fc_w = tf.get_variable("fc_w", shape=[output_size, layer],
                                           initializer=tf.contrib.layers.xavier_initializer())
                    fc_b = tf.get_variable("fc_b", shape=[layer], initializer=tf.zeros_initializer())
                    output = tf.nn.dropout(tf.tanh(tf.nn.xw_plus_b(output, fc_w, fc_b)),
                                           keep_prob=self.dropout_rate,
                                           name="output" + str(idx))
                    output_size = layer

        with tf.variable_scope("final_output_layer"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.get_variable("output_b", shape=[self.num_labels], dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            new_logits = tf.reshape(logits, [-1, self.max_len, self.num_labels])

            return new_logits

    def cal_loss(self, new_logits):
        """
        计算损失值
        :param mask:
        :param new_logits:
        :param true_y:
        :return:
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=new_logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.sequence_lens)
            return tf.reduce_mean(-log_likelihood), trans

    def get_pred(self, new_logits, trans_params=None):
        """
        得到预测值
        :param logits:
        :param new_logits:
        :param trans_params:
        :return:
        """
        with tf.name_scope("maskedOutput"):
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(new_logits, trans_params,
                                                                        self.sequence_lens)
            return viterbi_sequence

    def construct_graph(self):
        """
        构建计算图
        :return:
        """
        # 根据传入的序列长度取出实际的序列，mask的维度[batchSize, maxLen]，mask中的值为布尔值

        output, output_size = self.bi_lstms()
        new_logits = self.output_layer(output, output_size)
        loss, trans_params = self.cal_loss(new_logits)
        pred_y = self.get_pred(new_logits, trans_params)

        return (loss, new_logits, trans_params, pred_y)






