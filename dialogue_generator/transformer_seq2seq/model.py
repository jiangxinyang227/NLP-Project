import tensorflow as tf


class Seq2SeqTransformer(object):
    def __init__(self, config, vocab_size, word_vectors=None, is_training=True):
        self.vocab_size = vocab_size  # vocab size
        self.word_vectors = word_vectors  # 词向量得到的词嵌入矩阵
        self.embedding_size = config["embedding_size"]
        self.num_heads = config["num_heads"]  # multi head 的头数
        self.num_blocks = config["num_blocks"]  # transformer block的数量
        self.hidden_size = config["hidden_size"]  # feed forward层的隐层大小
        self.learning_rate = config["learning_rate"]  # 学习速率
        self.epsilon = config["ln_epsilon"]  # layer normalization 中除数中的极小值
        self.smooth_rate = config["smooth_rate"]  # smooth label的比例
        self.warmup_step = config["warmup_step"]  # 学习速率预热的步数
        self.decode_step = config["decode_step"]  # 解码的最大长度

        # pad字符在vocab中的数值表示
        self.pad_token = 0

        # placeholder 值
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
        self.decoder_outputs = tf.placeholder(tf.int32, [None, None], name="decoder_outputs")
        self.encoder_length = tf.placeholder(tf.int32, [None], name="encoder_length")
        self.decoder_length = tf.placeholder(tf.int32, [None], name="decoder_length")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")
        # 得到encoder decoder batch 的最大长度
        self.encoder_max_len = tf.reduce_max(self.encoder_length, name="encoder_max_len")
        self.decoder_max_len = tf.reduce_max(self.decoder_length, name="decoder_max_len")

        self.go_token = 2

        # 编码和解码共享embedding矩阵，若是不同语言的，如机器翻译，就各定义一个embedding矩阵
        self.embedding_matrix = self._get_embedding_matrix()

        self.build_model(is_training)
        self.init_saver()

    def encode(self, encoder_inputs, encoder_max_len):
        """
        定义decode部分
        :param encoder_inputs: encoder的原始输入
        :param encoder_max_len: encoder句子的最大长度
        :return:
        """
        with tf.name_scope("encoder"):
            # 词嵌入层，并加上位置向量 [batch_size, sequence_length, embedding_size]
            embedded_word = tf.nn.embedding_lookup(self.embedding_matrix, encoder_inputs)
            embedded_word *= self.hidden_size ** 0.5
            # [batch_size, sequence_length, embedding_size]
            embedded_word += self._add_position_embedding(embedded_word, encoder_max_len, "encoder_position_embedding")
            # [batch_size, sequence_length, embedding_size]
            embedded_word = tf.nn.dropout(embedded_word, self.keep_prob)

            # transformer结构
            for i in range(self.num_blocks):
                with tf.name_scope("transformer_{}".format(i)):
                    # multihead attention层
                    with tf.name_scope("self_attention"):
                        # [batch_size, sequence_length, embedding_size]
                        multihead_atten = self._multihead_attention(raw_queries=encoder_inputs,
                                                                    raw_keys=encoder_inputs,
                                                                    queries=embedded_word,
                                                                    keys=embedded_word,
                                                                    query_max_len=self.encoder_max_len,
                                                                    key_max_len=self.encoder_max_len,
                                                                    scope="encoder_sa_" + str(i))
                    # feed forward 层
                    with tf.name_scope("feed_forward"):
                        # [batch_size, sequence_length, embedding_size]
                        embedded_word = self._feed_forward(multihead_atten, encoder_max_len, "ff_" + str(i))
        return embedded_word

    def decode(self, encoder_inputs, decoder_inputs, encoder_outputs, decoder_max_len):
        """
        decode部分
        :param encoder_inputs: encoder的原始输入
        :param decoder_inputs: decoder的原始输入
        :param encoder_outputs: encoder的输出
        :param decoder_max_len: decoder句子的最大长度
        :return:
        """
        with tf.name_scope("decoder"):
            # embedding [batch_size, sequence_length, embedding_size]
            embedded_word = tf.nn.embedding_lookup(self.embedding_matrix, decoder_inputs)
            embedded_word *= self.hidden_size ** 0.5
            embedded_word = self._add_position_embedding(embedded_word, decoder_max_len, "decoder_position_embedding")
            embedded_word = tf.nn.dropout(embedded_word, self.keep_prob)

            for i in range(self.num_blocks):
                with tf.name_scope("transformer_{}".format(i)):
                    # self attention 层
                    with tf.name_scope("self_attention"):
                        multihead_atten = self._multihead_attention(raw_queries=decoder_inputs,
                                                                    raw_keys=decoder_inputs,
                                                                    queries=embedded_word,
                                                                    keys=embedded_word,
                                                                    query_max_len=decoder_max_len,
                                                                    key_max_len=decoder_max_len,
                                                                    scope="decoder_sa_" + str(i),
                                                                    causality=True)

                    # Vanilla attention 层， 用来连接encoder和decoder，在这里的query是decoder的输入multiatten之后的结果
                    # keys是encoder 的output的结果
                    with tf.name_scope("vanilla_attention"):
                        vanilla_atten = self._multihead_attention(raw_queries=decoder_inputs,
                                                                  raw_keys=encoder_inputs,
                                                                  queries=multihead_atten,
                                                                  keys=encoder_outputs,
                                                                  query_max_len=decoder_max_len,
                                                                  key_max_len=self.encoder_max_len,
                                                                  scope="decoder_va_" + str(i),
                                                                  causality=False)
                    # Feed Forward 层
                    with tf.name_scope("feed_forward"):
                        embedded_word = self._feed_forward(vanilla_atten, decoder_max_len, "ff" + str(i))

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embedding_matrix)  # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', embedded_word, weights)  # (N, T2, vocab_size)

        return logits

    def _get_embedding_matrix(self, zero_pad=True):
        """
        词嵌入层
        :param zero_pad:
        :return:
        """
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embeddings = tf.get_variable('embedding_w',
                                         dtype=tf.float32,
                                         shape=(self.vocab_size, self.embedding_size),
                                         initializer=tf.glorot_normal_initializer())
            if zero_pad:
                embeddings = tf.concat((tf.zeros(shape=[1, self.embedding_size]),
                                        embeddings[1:, :]), 0)

        return embeddings

    def _add_position_embedding(self, inputs, max_len, scope):
        """
        生成位置向量
        :return:
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # [embedding_size, seq_len]
            pos = tf.cast(tf.tile(tf.expand_dims(tf.range(max_len), axis=0), multiples=[self.embedding_size, 1]),
                          tf.float32)
            # [embedding_size, seq_len]
            i = tf.cast(tf.tile(tf.expand_dims(tf.range(self.embedding_size), axis=1), multiples=[1, max_len]),
                        tf.float32)

            # 定义正弦和余弦函数
            sine = tf.sin(tf.divide(pos, tf.pow(float(10 ** 4), tf.divide(i, self.embedding_size))))  # [E, T]
            cosine = tf.cos(tf.divide(pos, tf.pow(float(10 ** 4), tf.divide(i, self.embedding_size))))  # [E, T]
            cosine = tf.manip.roll(cosine, shift=1, axis=0)

            # 生成正弦和余弦的分段函数
            even_mask = tf.equal(tf.mod(tf.range(self.embedding_size), 2), 0)  # [embedding_size]
            joint_pos = tf.where(condition=even_mask, x=sine, y=cosine)  # [embedding_size, seq_len]
            joint_pos = tf.transpose(joint_pos)  # [seq_len, embedding_size]

            # 添加位置向量 [batch_size, seq_len, embedding_size]

            embedding = tf.add(inputs, joint_pos, name="composed_embedding")

            return embedding

    def _layer_normalization(self, inputs, scope):
        """
        对最后维度的结果做归一化，也就是说对每个样本每个时间步输出的向量做归一化
        :param inputs:
        :return:
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + self.epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def _multihead_attention(self, raw_queries, raw_keys, queries, keys, query_max_len, key_max_len, scope,
                             causality=False, num_units=None):
        """
        计算多头注意力
        :param raw_queries: 原始queries，用于计算mask
        :param raw_keys: 原始keys，用于计算mask
        :param queries: 添加了位置向量的词向量
        :param keys: 添加了位置向量的词向量
        :param query_max_len:
        :param key_max_len:
        :param scope:
        :param num_units: 计算多头注意力后的向量长度，如果为None，则取embedding_size
        :return:
        """
        if num_units is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            num_units = queries.get_shape().as_list()[-1]

        # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
        # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
        # Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
        Q = self.dense_layer(queries, query_max_len, num_units, num_units, scope + "Q")
        K = self.dense_layer(keys, key_max_len, num_units, num_units, scope + "K")
        V = self.dense_layer(keys, key_max_len, num_units, num_units, scope + "V")

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similarity = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值 [batch_size * num_heads, queries_len, key_len]
        similarity = similarity / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。在这里我们不仅要对keys做mask，还要对querys做mask
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 利用tf.tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        key_masks = tf.tile(raw_keys, [self.num_heads, 1])

        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        # tf.ones_like生成元素全为1，维度和similarity相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(similarity) * (-2 ** 32 + 1)

        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        masked_similarity = tf.where(tf.equal(key_masks, 0), paddings,
                                     similarity)  # 维度[batch_size * numHeads, queries_len, key_len]

        """
        在解码的过程中，因为只能看到当前解码的前面的词，因此需要设计相似度矩阵，使得相似度矩阵在计算attention的时候只考虑
        前面的词，这个时候只要用下三角矩阵就可以实现，因此需要对上三角做mask
        下三角矩阵：[[1, 0, 0], [0.8, 0.2, 0], [0.8, 0.1, 0.1]]（此处不考虑padding），这个attention权重矩阵就能实现上述的解码
        在解码第一个词时用到第一个向量，第二个词用到第二个向量，因此类推。
        """
        if causality:
            diag_vals = tf.ones_like(masked_similarity[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            diag_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(masked_similarity)[0], 1, 1])  # (N, T_q, T_k)

            diag_paddings = tf.ones_like(diag_masks) * (-2 ** 32 + 1)
            masked_similarity = tf.where(tf.equal(diag_masks, 0), diag_paddings, masked_similarity)

        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len], 在这里因为上面的paddings是负无穷
        # 大的值，因此在softmax之后就是0了，相当于对raw_keys中padding的token分配的权重就是0
        weights = tf.nn.softmax(masked_similarity)

        # 因为key和query是相同的输入，当存在padding时，计算出来的相似度矩阵应该是行和列都存在mask的部分，上面的key_masks是
        # 对相似度矩阵中的列mask，mask完之后，还要对行做mask，列mask时用负无穷来使得softmax（在这里的softmax是对行来做的）
        # 计算出来的非mask部分的值相加还是为1，行mask就直接去掉就行了，以上的分析均针对batch_size等于1.
        """
        mask的相似度矩阵：[[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]]
        初始的相似度矩阵:[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        一，key_masks + 行softmax：[[0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]]
        二，query_masks后：[[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]]
        """
        # 利用tf.tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        query_masks = tf.tile(raw_queries, [self.num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        mask_weights = tf.where(tf.equal(query_masks, 0), paddings,
                                weights)  # 维度[batch_size * numHeads, queries_len, key_len]

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(mask_weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, self.keep_prob)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层 [batch_size, sequence_length, embedding_size]
        outputs = self._layer_normalization(outputs, "ma_ln")
        return outputs

    def _feed_forward(self, inputs, max_len, scope):
        """
        前向连接层
        :param inputs:
        :return:
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            input_reshape = tf.reshape(inputs, [-1, self.embedding_size])
            w_1 = tf.get_variable("w_1", shape=[self.embedding_size, self.hidden_size],
                                  initializer=tf.glorot_normal_initializer())
            b_1 = tf.get_variable("b_1", shape=[self.hidden_size],
                                  initializer=tf.glorot_normal_initializer())
            output_1 = tf.nn.relu(tf.nn.xw_plus_b(input_reshape, w_1, b_1))

            w_2 = tf.get_variable("w_2", shape=[self.hidden_size, self.embedding_size],
                                  initializer=tf.glorot_normal_initializer())
            b_2 = tf.get_variable("b_2", shape=[self.embedding_size],
                                  initializer=tf.glorot_normal_initializer())
            output_2 = tf.nn.relu(tf.nn.xw_plus_b(output_1, w_2, b_2))

            output_reshape = tf.reshape(output_2, [-1, max_len, self.embedding_size])
            # 残差链家而
            output_reshape += inputs

            # 归一化
            outputs = self._layer_normalization(output_reshape, "ff_ln")

            return outputs

    def dense_layer(self, inputs, max_len, input_size, output_size, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            input_reshape = tf.reshape(inputs, [-1, input_size])
            w = tf.get_variable("w", shape=[input_size, output_size],
                                initializer=tf.glorot_normal_initializer())
            b = tf.get_variable("b", shape=[output_size],
                                initializer=tf.glorot_normal_initializer())
            outputs = tf.nn.relu(tf.nn.xw_plus_b(input_reshape, w, b))
            output_reshape = tf.reshape(outputs, [-1, max_len, output_size])
            return output_reshape

    @staticmethod
    def label_smoothing(inputs, smooth_rate):
        """
        标签平滑，将原本的one-hot真实标签向量变成一个不含0的标签向量
        :param inputs:
        :param smooth_rate: 标签平滑率
        :return:
        """
        V = inputs.get_shape().as_list()[-1]
        return ((1 - smooth_rate) * inputs) + (smooth_rate / V)

    def train_method(self):
        """
        定义训练方法
        :return:
        """
        self.predictions = tf.to_int32(tf.argmax(self.logits, axis=-1))
        # train scheme
        # 对真实的标签做平滑处理
        y_ = self.label_smoothing(tf.one_hot(self.decoder_outputs, depth=self.vocab_size), self.smooth_rate)
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_)
        # 取出非0部分，即非padding部分
        non_padding = tf.to_float(tf.not_equal(self.decoder_outputs, self.pad_token))
        self.loss = tf.reduce_sum(losses * non_padding) / (tf.reduce_sum(non_padding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        # 动态的修改初始的学习速率
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(self.warmup_step, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = self.learning_rate * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        self.learning_rate = (
                (1.0 - is_warmup) * self.learning_rate + is_warmup * warmup_learning_rate)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("lr", self.learning_rate)
        self.summary_op = tf.summary.merge_all()

    def infer(self, encoder_inputs, encoder_outputs):
        """
        预测部分
        :param encoder_inputs:
        :param encoder_outputs:
        :return:
        """

        # 在验证时，没有真实的decoder_inputs，此时需要构造一个初始的输入，初始的输入用初始符
        init_decoder_inputs = tf.ones((1, 1), tf.int32) * self.go_token
        init_decoder_max_len = tf.constant(1)

        def cond(decoder_inputs, decoder_max_len):
            return True

        def body(decoder_inputs, decoder_max_len):
            logits = self.decode(encoder_inputs, decoder_inputs, encoder_outputs, decoder_max_len)
            y_pred = tf.argmax(logits, axis=-1)
            decoder_inputs = tf.concat((init_decoder_inputs, tf.cast(y_pred, dtype=tf.int32)), 1)
            decoder_max_len += 1
            return [decoder_inputs, decoder_max_len]

        decoder_inputs, decoder_max_len = body(init_decoder_inputs, init_decoder_max_len)

        decoder_inputs, decoder_max_len = tf.while_loop(
            cond=cond, body=body, loop_vars=[decoder_inputs, decoder_max_len],
            shape_invariants=[tf.TensorShape([1, None]),
                              tf.TensorShape([])],
            maximum_iterations=self.decode_step
        )
        self.predictions = decoder_inputs[:, 1:]

    def build_model(self, is_training=True):
        """
        搭建计算图
        :return:
        """
        encoder_outputs = self.encode(self.encoder_inputs, self.encoder_max_len)
        if is_training:
            self.logits = self.decode(self.encoder_inputs, self.decoder_inputs, encoder_outputs, self.decoder_max_len)
            self.train_method()
        else:
            self.infer(self.encoder_inputs, encoder_outputs)

    def init_saver(self):
        """
        初始化saver对象
        :return:
        """
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

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

    def predict(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch["encoder_inputs"],
                     self.encoder_length: batch["encoder_length"],
                     self.keep_prob: 1.0
                     }
        predictions = sess.run(self.predictions, feed_dict=feed_dict)
        return predictions
