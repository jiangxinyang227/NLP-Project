import numpy as np
import tensorflow as tf

from .base import BaseModel


class CharRNNModel(BaseModel):
    def __init__(self, config, vocab_size, word_vectors, is_training=True):
        super(CharRNNModel, self).__init__(config, vocab_size, word_vectors)
        self.batch_size = config["batch_size"]
        self.hidden_sizes = config["hidden_sizes"]
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        if is_training is False:
            self.batch_size = 1

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

        with tf.name_scope("LSTM"):
            multi_cells = []
            for idx, hidden_size in enumerate(self.hidden_sizes):
                with tf.name_scope("LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)
                    multi_cells.append(lstm_cell)

            cell = tf.nn.rnn_cell.MultiRNNCell(multi_cells)

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
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.matmul(x, output_w) + output_b
            self.predictions = self.get_predictions()

        self.loss = self.cal_loss()
        self.train_op, self.summary_op = self.get_train_op()



