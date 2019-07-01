import tensorflow as tf
from .base import BaseModel


class Seq2SeqConv(BaseModel):
    """
    定义卷积到卷积的seq2seq网络结构
    """

    def __init__(self, config, vocab_size, word_vectors=None, training=True):
        super(Seq2SeqConv, self).__init__(config)

        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors
        self.num_layers = config["num_layers"]
        self.kernel_size = config["kernel_size"]
        self.num_filters = config["num_filters"]
        self.training = training

        # 根据目标序列长度，选出其中最大值，然后使用该值构建序列长度的mask标志。用一个sequence_mask的例子来说明起作用
        self.encoder_max_len = tf.reduce_max(self.encoder_length, name="encoder_max_len")
        self.decoder_max_len = tf.reduce_max(self.decoder_length, name="decoder_max_len")

        # 编码和解码共享embedding矩阵，若是不同语言的，如机器翻译，就各定义一个embedding矩阵
        self.embedding_matrix = self._get_embedding_matrix()

        # 实例化对象时构建网络结构
        self.build_network()
        self.init_saver()

    def encoder_layer(self, inputs, input_len, training):
        """
        单层encoder层的实现
        :param inputs: encoder的输入 [batch_size, seq_len, hidden_size]
        :param input_len: encoder输入的实际长度
        :param training:
        :return:
        """

        seq_len = tf.shape(inputs)[1]

        # 计算序列在卷积时要补的长度
        num_pad = self.kernel_size - 1

        # 在序列的左右各补长一半
        x_pad = tf.pad(inputs, paddings=[[0, 0], [int(num_pad / 2), int(num_pad / 2)], [0, 0]])

        filters = tf.constant(1, tf.float32, (self.kernel_size, self.hidden_size, 2 * self.hidden_size))
        # 一维卷积操作，针对3维的输入的卷积 [batch_size, seq_len, 2 * hidden_size]
        x_conv = tf.nn.conv1d(x_pad, filters, stride=1, padding="VALID")

        # glu 门控函数 [batch_size, seq_len, hidden_size]
        x_glu = self.glu(x_conv)

        # mask padding [batch_size, seq_len]
        mask = tf.sequence_mask(lengths=input_len, maxlen=seq_len, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)  # [batch_size, seq_len, 1]

        # element-wise的乘积, [batch_size, seq_len, hidden_size]
        x_mask = tf.multiply(mask, x_glu)

        # drouput 正则化 [batch_size, seq_len, hidden_size]
        x_drop = tf.nn.dropout(x_mask, keep_prob=self.keep_prob, noise_shape=[self.batch_size, 1, self.hidden_size])

        # 残差连接 [batch_size, seq_len, hidden_size]
        x_drop = x_drop + inputs

        # bn处理，只对最后一个维度做bn处理 [batch_size, seq_len, hidden_size]
        x_bn = tf.layers.BatchNormalization(axis=2)(x_drop, training=training)

        return x_bn

    def encoder(self, encoder_inputs, encoder_length, encoder_max_len, training):
        """
        多层encoder层
        :param encoder_inputs: encoder的原始输入
        :param encoder_length: encoder 输入的真实长度
        :param encoder_max_len: encoder 输入的最大长度
        :param training:
        :return:
        """
        embedded_word = tf.nn.embedding_lookup(self.embedding_matrix, encoder_inputs)

        embedded_word += self._position_embedding(embedded_word, encoder_max_len, "encoder")
        embedded_word_drop = tf.nn.dropout(embedded_word, self.keep_prob)

        with tf.name_scope("encoder_start_linear_map"):
            w_start = tf.get_variable("w_start", shape=[self.embedding_size, self.hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_start = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name="b_start")

            inputs_start = tf.reshape(tf.nn.xw_plus_b(tf.reshape(embedded_word_drop,
                                                                 [-1, self.embedding_size]),
                                                      w_start, b_start),
                                      [self.batch_size, -1, self.hidden_size])  # [batch_size, seq_len, hidden_size]

            inputs_start = tf.nn.dropout(inputs_start, keep_prob=self.keep_prob)

        # 维度不变
        for layer_id in range(self.num_layers):
            with tf.name_scope("encoder_layer_" + str(layer_id)):
                inputs_start = self.encoder_layer(inputs=inputs_start,
                                                  input_len=encoder_length,
                                                  training=training)

        with tf.name_scope("encoder_final_linear_map"):
            w_final = tf.get_variable("w_final", shape=[self.hidden_size, self.embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_final = tf.Variable(tf.constant(0.1, shape=[self.embedding_size]), name="b_final")

            # [batch_size, seq_len, embedding_size]
            inputs_final = tf.reshape(tf.nn.xw_plus_b(tf.reshape(inputs_start,
                                                                 [-1, self.hidden_size]),
                                                      w_final, b_final),
                                      [self.batch_size, -1, self.embedding_size])

        return embedded_word, inputs_final

    def decoder_layer(self, raw_inputs, new_inputs, input_len, encoder_embedded, encoder_output,
                      encoder_length, training):
        """
        单层decoder层的实现
        :param raw_inputs: 原始的decoder输入
        :param new_inputs: 上一层decoder的输出
        :param input_len: decoder的输入的真实长度
        :param encoder_embedded: encoder的原始输入
        :param encoder_output: encoder的输出
        :param encoder_length: encoder的输入的真实长度
        :param training:
        :return:
        """
        seq_len = tf.shape(new_inputs)[1]

        num_pad = self.kernel_size - 1

        # 在序列的左边进行补长，
        x_pad = tf.pad(new_inputs, paddings=[[0, 0], [num_pad, 0], [0, 0]])

        filters = tf.constant(1, tf.float32, (self.kernel_size, self.hidden_size, 2 * self.hidden_size))
        # 一维卷积操作，针对3维的输入的卷积，[batch_size, seq_len, 2 * hidden_size]
        x_conv = tf.nn.conv1d(x_pad, filters, stride=1, padding="VALID")

        # glu 门控函数 [batch_size, seq_len, hidden_size]
        x_glu = self.glu(x_conv)

        with tf.variable_scope("decoder_middle_linear_map"):
            w_middle = tf.get_variable("w_middle", shape=[self.hidden_size, self.embedding_size],
                                       initializer=tf.contrib.layers.xavier_initializer())
            b_middle = tf.Variable(tf.constant(0.1, shape=[self.embedding_size]), name="b_middle")

            # [batch_size, seq_len, embedding_size]
            x_middle = tf.add(tf.reshape(tf.nn.xw_plus_b(tf.reshape(x_glu, [-1, self.hidden_size]),
                                                         w_middle, b_middle),
                                         [self.batch_size, -1, self.embedding_size]), raw_inputs)

        # attention [batch_size, seq_len, embedding_size]
        x_atten = self.attention(query=x_middle,
                                 encoder_embedded=encoder_embedded,
                                 key=encoder_output,
                                 value=encoder_output,
                                 query_len=input_len,
                                 key_len=encoder_length)

        with tf.variable_scope("decoder_middle_linear_map_1_"):
            w_middle_1 = tf.get_variable("w_middle_1",
                                         shape=[self.embedding_size, self.hidden_size],
                                         initializer=tf.contrib.layers.xavier_initializer())
            b_middle_1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name="b_middle_1")

            # [batch_size, seq_len, hidden_size]
            x_middle_1 = tf.reshape(tf.nn.xw_plus_b(tf.reshape(x_atten,
                                                               [-1, self.embedding_size]),
                                                    w_middle_1, b_middle_1),
                                    [self.batch_size, -1, self.hidden_size])

            # [batch_size, seq_len, hidden_size]
            x_final = x_glu + x_middle_1 + new_inputs

        # mask padding [batch_size, seq_len]
        mask = tf.sequence_mask(lengths=input_len, maxlen=seq_len, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)  # [batch_size, seq_len, 1]

        # element-wise的乘积, [batch_size, seq_len, hidden_size]
        x_mask = tf.multiply(mask, x_final)

        # drouput 正则化
        x_drop = tf.nn.dropout(x_mask, keep_prob=self.keep_prob,
                               noise_shape=[self.batch_size, 1, self.hidden_size])

        # 残差连接
        x_drop = x_drop + new_inputs

        x_bn = tf.layers.BatchNormalization(axis=2)(x_drop, training=training)

        return x_bn

    def decoder(self, decoder_inputs, decoder_length, decoder_max_len,
                encoder_embedded, encoder_output, encoder_length, training):
        """
        decoder部分
        :param decoder_inputs: decoder的输入
        :param decoder_length: decoder的输入的真实长度
        :param decoder_max_len: decoder的输入的最大长度
        :param encoder_embedded: encoder的输入
        :param encoder_output: encoder的输出
        :param encoder_length: encoder的输入的真实长度
        :param training:
        :return: 卷积的seq2seq在解码时是独立的对每个时间步进行多分类 [batch_size, de_seq_len, vocab_size]
        """
        embedded_word = tf.nn.embedding_lookup(self.embedding_matrix, decoder_inputs)

        embedded_word += self._position_embedding(embedded_word, decoder_max_len, "decoder")
        embedded_word = tf.nn.dropout(embedded_word, self.keep_prob)
        print(embedded_word)

        with tf.variable_scope("decoder_start_linear_map"):
            w_start = tf.get_variable("w_start", shape=[self.embedding_size, self.hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_start = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name="b_start")

            inputs_start = tf.reshape(tf.nn.xw_plus_b(tf.reshape(embedded_word,
                                                                 [-1, self.embedding_size]),
                                                      w_start, b_start),
                                      [self.batch_size, -1, self.hidden_size])  # [batch_size, seq_len, hidden_size]

            inputs_start = tf.nn.dropout(inputs_start, keep_prob=self.keep_prob)

        for layer_id in range(self.num_layers):
            with tf.variable_scope("decoder_layer_" + str(layer_id)):
                inputs_start = self.decoder_layer(raw_inputs=embedded_word,
                                                  new_inputs=inputs_start,
                                                  input_len=decoder_length,
                                                  encoder_embedded=encoder_embedded,
                                                  encoder_output=encoder_output,
                                                  encoder_length=encoder_length,
                                                  training=training)

        with tf.variable_scope("decoder_final_linear_map"):
            w_final = tf.get_variable("w_final", shape=[self.hidden_size, self.embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_final = tf.Variable(tf.constant(0.1, shape=[self.embedding_size]), name="b_final")

            # [batch_size, seq_len, embedding_size]
            inputs_final = tf.reshape(tf.nn.xw_plus_b(tf.reshape(inputs_start,
                                                                 [-1, self.hidden_size]),
                                                      w_final, b_final),
                                      [self.batch_size, -1, self.embedding_size])

            inputs_final = tf.nn.dropout(inputs_final, keep_prob=self.keep_prob)

        with tf.name_scope("output"):
            w_output = tf.get_variable("w_output", shape=[self.embedding_size, self.vocab_size],
                                       initializer=tf.contrib.layers.xavier_initializer())
            b_output = tf.Variable(tf.constant(0.1, shape=[self.vocab_size]), name="b_output")

            output = tf.reshape(tf.nn.xw_plus_b(tf.reshape(inputs_final,
                                                           [-1, self.embedding_size]),
                                                w_output, b_output),
                                [self.batch_size, -1, self.vocab_size])

        return output

    @staticmethod
    def padding_and_softmax(logits, query_len, key_len):
        """
        对attention权重归一化处理
        :param logits: 未归一化的attention权重 [batch_size, de_seq_len, en_seq_len]
        :param query_len: decoder的输入的真实长度 [batch_size]
        :param key_len: encoder的输入的真实长度 [batch_size]
        :return:
        """
        with tf.name_scope("padding_aware_softmax"):
            # 获得序列最大长度值
            de_seq_len = tf.shape(logits)[1]
            en_seq_len = tf.shape(logits)[2]

            # masks
            # [batch_size, de_seq_len]
            query_mask = tf.sequence_mask(lengths=query_len, maxlen=de_seq_len, dtype=tf.int32)
            # [batch_size, en_seq_len]
            key_mask = tf.sequence_mask(lengths=key_len, maxlen=en_seq_len, dtype=tf.int32)

            # 扩展一维
            query_mask = tf.expand_dims(query_mask, axis=2)  # [batch_size, de_seq_len, 1]
            key_mask = tf.expand_dims(key_mask, axis=1)  # [batch_size, 1, en_seq_len]

            # 将query和key的mask相结合 [batch_size, de_seq_len, en_seq_len]
            joint_mask = tf.cast(tf.matmul(query_mask, key_mask), tf.float32, name="joint_mask")

            # Padding should not influence maximum (replace with minimum)
            logits_min = tf.reduce_min(logits, axis=2, keepdims=True, name="logits_min")  # [batch_size, de_seq_len, 1]
            logits_min = tf.tile(logits_min, multiples=[1, 1, en_seq_len])  # [batch_size, de_seq_len, en_seq_len]
            logits = tf.where(condition=joint_mask > .5,
                              x=logits,
                              y=logits_min)

            # 获得最大值
            logits_max = tf.reduce_max(logits, axis=2, keepdims=True, name="logits_max")  # [batch_size, de_seq_len, 1]
            # 所有的元素都减去最大值  [batch_size, de_seq_len, en_seq_len]
            logits_shifted = tf.subtract(logits, logits_max, name="logits_shifted")

            # 导出未缩放的值
            weights_unscaled = tf.exp(logits_shifted, name="weights_unscaled")

            # mask 部分权重  [batch_size, de_seq_len, en_seq_len]
            weights_unscaled = tf.multiply(joint_mask, weights_unscaled, name="weights_unscaled_masked")

            # 得到每个时间步的总值 [batch_size, de_seq_len, 1]
            weights_total_mass = tf.reduce_sum(weights_unscaled, axis=2,
                                               keepdims=True, name="weights_total_mass")

            # 避免除数为0
            weights_total_mass = tf.where(condition=tf.equal(query_mask, 1),
                                          x=weights_total_mass,
                                          y=tf.ones_like(weights_total_mass))

            # 对权重进行正规化  [batch_size, de_seq_len, en_seq_len]
            weights = tf.divide(weights_unscaled, weights_total_mass, name="normalize_attention_weights")

            return weights

    def attention(self, query, encoder_embedded, key, value, query_len, key_len):
        """
        计算encoder decoder之间的attention
        :param query: decoder 的输入 [batch_size, de_seq_len, embedding_size]
        :param encoder_embedded:  encoder的嵌入输入 [batch_size, en_seq_len, embedding_size]
        :param key: encoder的输出 [batch_size, en_seq_len, embedding_size]
        :param value: encoder的输出 [batch_size, en_seq_len, embedding_size]
        :param query_len: decoder的输入的真实长度 [batch_size]
        :param key_len: encoder的输入的真实长度 [batch_size]
        :return:
        """
        with tf.name_scope("attention"):
            # 通过点积的方法计算权重, 得到[batch_size, de_seq_len, en_seq_len]
            attention_scores = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))

            # 对权重进行归一化处理
            attention_scores = self.padding_and_softmax(logits=attention_scores,
                                                        query_len=query_len,
                                                        key_len=key_len)
            # 对source output进行加权平均 [batch_size, de_seq_len, embedding_size]
            weighted_output = tf.matmul(attention_scores, tf.add(value, encoder_embedded))

            return weighted_output

    @staticmethod
    def glu(x):
        """
        glu门函数, 将后半段计算门系数，前半段作为输入值，element-wise的乘积
        :param x: 卷积操作后的Tensor [batch_size, seq_len, hidden_size * 2]
        :return: [batch_size, seq_len, hidden_size]
        """
        a, b = tf.split(x, num_or_size_splits=2, axis=2)
        return tf.multiply(tf.nn.sigmoid(b), a)

    def _position_embedding(self, inputs, max_len, mode):
        """
        对映射后的词向量加上位置向量，位置向量和transformer中的位置向量一样
        :param inputs: [batch_size, seq_len, embedding_size]
        :return: [batch_size, seq_len, embedding_size]
        """
        d_model = self.embedding_size

        # [embedding_size, seq_len]
        pos = tf.cast(tf.tile(tf.expand_dims(tf.range(max_len), axis=0), multiples=[d_model, 1]), tf.float32)
        # [embedding_size, seq_len]
        i = tf.cast(tf.tile(tf.expand_dims(tf.range(d_model), axis=1), multiples=[1, max_len]), tf.float32)

        # 定义正弦和余弦函数
        sine = tf.sin(tf.divide(pos, tf.pow(float(10 ** 4), tf.divide(i, d_model))))  # [E, T]
        cosine = tf.cos(tf.divide(pos, tf.pow(float(10 ** 4), tf.divide(i, d_model))))  # [E, T]
        cosine = tf.manip.roll(cosine, shift=1, axis=0)

        # 生成正弦和余弦的分段函数
        even_mask = tf.equal(tf.mod(tf.range(d_model), 2), 0)  # [embedding_size]
        joint_pos = tf.where(condition=even_mask, x=sine, y=cosine)  # [embedding_size, seq_len]
        joint_pos = tf.transpose(joint_pos)  # [seq_len, embedding_size]

        # 对位置向量进行缩放, 标量
        gamma = tf.get_variable(name="gamma_" + mode,
                                shape=[],
                                initializer=tf.initializers.ones,
                                trainable=True,
                                dtype=tf.float32)

        # 添加位置向量 [batch_size, seq_len, embedding_size]
        embedding = tf.add(inputs, gamma * joint_pos, name="composed_embedding")

        return embedding

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

    def train_method(self):
        """
        定义训练方法和损失
        :return:
        """
        self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # [batch_size, de_seq_len]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.decoder_outputs)
        decoder_mask = tf.sequence_mask(self.decoder_length, self.decoder_max_len,
                                        dtype=tf.float32, name='target_masks')
        losses = tf.boolean_mask(loss, decoder_mask)
        self.loss = tf.reduce_mean(losses, name="loss")

        optimizer = tf.train.MomentumOptimizer(learning_rate=0.25, momentum=0.99, use_nesterov=True)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 对梯度进行梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params), name="train_op")

        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

    def build_network(self):
        with tf.name_scope("encoder"):
            encoder_embedded, encoder_output = self.encoder(self.encoder_inputs, self.encoder_length,
                                                            self.encoder_max_len, self.training)

        with tf.name_scope("decoder"):
            self.logits = self.decoder(self.decoder_inputs, self.decoder_length, self.decoder_max_len,
                                       encoder_embedded, encoder_output, self.encoder_length, self.training)

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
