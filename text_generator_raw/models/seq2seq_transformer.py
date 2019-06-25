from tqdm import tqdm
import logging
import numpy as np
import tensorflow as tf


class Seq2SeqTransformer():
    def __init__(self, config):
        self.epsilon = config["lr_epsilon"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.num_heads = config["num_heads"]
        self.num_blocks = config["num_blocks"]
        self.batch_size = config["batch_size"]
        self.sequence_length = config["sequence_length"]
        self.keep_prob = config["keep_prob"]
        self.hidden_size = config["hidden_size"]
        self.learning_rate = config["learning_rate"]
        self.smooth_rate = config["smooth_rate"]
        self.warmup_step = config["warmup_step"]
        self.decode_step = config["decode_step"]

        # 编码和解码共享embedding矩阵，若是不同语言的，如机器翻译，就各定义一个embedding矩阵
        self.embedding_matrix = self._get_embedding_matrix()

        self.pad_token = 0
        self.start_token = 2

    def encode(self, encoder_inputs):
        """
        定义decode部分
        :param encoder_inputs:
        :return:
        """
        with tf.name_scope("encoder"):
            # 词嵌入层，并加上位置向量
            embedded_word = tf.nn.embedding_lookup(self.embedding_matrix, encoder_inputs)

            embedded_word += self._position_embedding(encoder_inputs)
            embedded_word = tf.nn.dropout(embedded_word, self.keep_prob)

            # transformer结构
            for i in range(self.num_blocks):
                with tf.name_scope("transformer_{}".format(i)):
                    # multihead attention层
                    with tf.name_scope("self_attention"):
                        multihead_atten = self._multihead_attention(raw_queries=encoder_inputs,
                                                                    raw_keys=encoder_inputs,
                                                                    queries=embedded_word,
                                                                    keys=embedded_word)
                    # feed forward 层
                    with tf.name_scope("feed_forward"):
                        embedded_word = self._feed_forward(multihead_atten)
        return embedded_word

    def decode(self, encoder_inputs, decoder_inputs, encoder_outputs):
        """
        decode部分
        :param encoder_inputs:
        :param decoder_inputs:
        :param encoder_outputs:
        :return:
        """
        with tf.name_scope("decoder"):
            # embedding
            embedded_word = tf.nn.embedding_lookup(self.embedding_matrix, decoder_inputs)

            embedded_word += self._position_embedding(decoder_inputs)
            embedded_word = tf.nn.dropout(embedded_word, self.keep_prob)

            for i in range(self.num_blocks):
                with tf.name_scope("transformer_{}".format(i)):
                    # self attention 层
                    with tf.name_scope("self_attention"):
                        multihead_atten = self._multihead_attention(raw_queries=decoder_inputs,
                                                                    raw_keys=decoder_inputs,
                                                                    queries=embedded_word,
                                                                    keys=embedded_word,
                                                                    causality=True)

                    # Vanilla attention 层
                    with tf.name_scope("vanilla_attention"):
                        vanilla_atten = self._multihead_attention(raw_queries=decoder_inputs,
                                                                  raw_keys=encoder_inputs,
                                                                  queries=multihead_atten,
                                                                  keys=encoder_outputs,
                                                                  causality=False)
                    # Feed Forward 层
                    with tf.name_scope("feed_forward"):
                        embedded_word = self._feed_forward(vanilla_atten)

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embedding_matrix)  # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', embedded_word, weights)  # (N, T2, vocab_size)
        y_pred = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_pred

    def train(self, encoder_inputs, decoder_inputs, y_true):
        """
        预测模型
        :param encoder_inputs:
        :param decoder_inputs:
        :param y_true:
        :return:
        """
        # forward
        encoder_outputs = self.encode(encoder_inputs)
        logits, y_pred = self.decode(encoder_inputs, decoder_inputs, encoder_outputs)

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

    def eval(self, encoder_inputs, decoder_inputs, y_true):
        """
        验证模型
        :param encoder_inputs:
        :param decoder_inputs:
        :param y_true:
        :return:
        """
        # forward
        encoder_outputs = self.encode(encoder_inputs)
        logits, y_pred = self.decode(encoder_inputs, decoder_inputs, encoder_outputs)

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

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.decode_step)):
            logits, y_pred = self.decode(encoder_inputs, decoder_inputs, encoder_outputs)
            if tf.reduce_sum(y_pred, 1) == self.pad_token: break

            decoder_inputs = tf.concat((decoder_inputs, y_pred), 1)

        return y_pred

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

    def _position_embedding(self, inputs, masking=True):
        """
        生成位置向量
        :return:
        """
        with tf.variable_scope("position_embedding", reuse=tf.AUTO_REUSE):
            # 生成位置的索引，并扩张到batch中所有的样本上
            position_index = tf.tile(tf.expand_dims(tf.range(self.sequence_length), 0), [self.batch_size, 1])

            # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
            position_embedding = np.array([[pos / np.power(10000, (i - i % 2) / self.embedding_size)
                                            for i in range(self.embedding_size)]
                                           for pos in range(self.sequence_length)])

            # 然后根据奇偶性分别用sin和cos函数来包装
            position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])
            position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])

            # 将positionEmbedding转换成tensor的格式
            position_embedding = tf.cast(position_embedding, dtype=tf.float32)

            # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
            embedded_position = tf.nn.embedding_lookup(position_embedding, position_index)

            # 对位置向量按照输入的一样，对padding部分对应的位置向量用0代替，这样方便在之后计算attention
            if masking:
                embedded_position = tf.where(tf.equal(inputs, 0), inputs, embedded_position)

            return embedded_position

    def _layer_normalization(self, inputs):
        """
        对最后维度的结果做归一化，也就是说对每个样本每个时间步输出的向量做归一化
        :param inputs:
        :return:
        """
        with tf.variable_scope("layer_norm", reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + self.epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def _multihead_attention(self, raw_queries, raw_keys, queries, keys, causality=False, num_units=None):
        """
        计算多头注意力
        :param raw_queries: 原始quers，用于计算mask
        :param raw_keys: 原始keys，用于计算mask
        :param queries: 添加了位置向量的词向量
        :param keys: 添加了位置向量的词向量
        :param num_units: 计算多头注意力后的向量长度，如果为None，则取embedding_size
        :return:
        """
        if num_units is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            num_units = queries.get_shape().as_list()[-1]

        # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
        # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
        # Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similarity = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        similarity = similarity / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。在这里我们不仅要对keys做mask，还要对querys做mask
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
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
            diag_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(raw_keys)[0], 1, 1])  # (N, T_q, T_k)

            diag_paddings = tf.ones_like(diag_masks) * (-2 ** 32 + 1)
            masked_similarity = tf.where(tf.equal(diag_masks, 0), diag_paddings, masked_similarity)

        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
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
        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        query_masks = tf.tile(raw_queries, [self.num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        mask_weights = tf.where(tf.equal(query_masks, 0), paddings,
                                weights)  # 维度[batch_size * numHeads, queries_len, key_len]

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(mask_weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layer_normalization(outputs)
        return outputs

    def _feed_forward(self, inputs):
        """
        前向连接层
        :param inputs:
        :return:
        """
        # 隐藏层
        outputs = tf.layers.dense(inputs, self.hidden_size, activation=tf.nn.relu)

        # 输出层
        outputs = tf.layers.dense(outputs, self.embedding_size)

        # 残差链家而
        outputs += inputs

        # 归一化
        outputs = self._layer_normalization(outputs)

        return outputs

    def label_smoothing(self, inputs):
        """
        标签平滑，将原本的one-hot真实标签向量变成一个不含0的标签向量
        :param inputs:
        :param epsilon:
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



