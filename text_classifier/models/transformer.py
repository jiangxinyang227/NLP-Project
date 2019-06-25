import tensorflow as tf
import numpy as np
from .base import BaseModel


class TransformerModel(BaseModel):
    def __init__(self, config, vocab_size, word_vectors):
        super(TransformerModel, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
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

        with tf.name_scope("positionEmbedding"):
            embedded_position = self._position_embedding()

        embedded_representation = embedded_words + embedded_position

        with tf.name_scope("transformer"):
            for i in range(self.config["num_blocks"]):
                with tf.name_scope("transformer-{}".format(i + 1)):
                    with tf.name_scope("multi_head_atten"):
                        # 维度[batch_size, sequence_length, embedding_size]
                        multihead_atten = self._multihead_attention(inputs=self.inputs,
                                                                    queries=embedded_representation,
                                                                    keys=embedded_representation)
                    with tf.name_scope("feed_forward"):
                        # 维度[batch_size, sequence_length, embedding_size]
                        embedded_representation = self._feed_forward(multihead_atten,
                                                                     [self.config["filters"],
                                                                      self.config["embedding_size"]])

            outputs = tf.reshape(embedded_representation,
                                 [-1, self.config["sequence_length"] * self.config["embedding_size"]])

        output_size = outputs.get_shape()[-1].value

        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(outputs, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        # 计算交叉熵损失
        self.loss = self.cal_loss() + self.config["l2_reg_lambda"] * self.l2_loss
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()

    def _layer_normalization(self, inputs):
        """
        对最后维度的结果做归一化，也就是说对每个样本每个时间步输出的向量做归一化
        :param inputs:
        :return:
        """
        epsilon = self.config["ln_epsilon"]

        inputs_shape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]
        params_shape = inputs_shape[-1]

        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        # mean, variance的维度都是[batch_size, sequence_len, 1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)

        gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        outputs = gamma * normalized + beta

        return outputs

    def _multihead_attention(self, inputs, queries, keys, num_units=None):
        """
        计算多头注意力
        :param inputs: 原始输入，用于计算mask
        :param queries: 添加了位置向量的词向量
        :param keys: 添加了位置向量的词向量
        :param num_units: 计算多头注意力后的向量长度，如果为None，则取embedding_size
        :return:
        """
        num_heads = self.config["num_heads"]  # multi head 的头数

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
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similarity = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        similarity = similarity / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。在这里我们不仅要对keys做mask，还要对querys做mask
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        mask = tf.tile(inputs, [num_heads, 1])

        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        key_masks = tf.tile(tf.expand_dims(mask, 1), [1, tf.shape(queries)[1], 1])

        # tf.ones_like生成元素全为1，维度和similarity相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(similarity) * (-2 ** 32 + 1)

        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        masked_similarity = tf.where(tf.equal(key_masks, 0), paddings,
                                     similarity)  # 维度[batch_size * numHeads, queries_len, key_len]

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
        query_masks = tf.tile(tf.expand_dims(mask, -1), [1, 1, tf.shape(keys)[1]])
        mask_weights = tf.where(tf.equal(query_masks, 0), paddings,
                                weights)  # 维度[batch_size * numHeads, queries_len, key_len]

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(mask_weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layer_normalization(outputs)
        return outputs

    def _feed_forward(self, inputs, filters):
        """
        用卷积网络来做全连接层
        :param inputs: 接收多头注意力计算的结果作为输入
        :param filters: 卷积核的数量
        :return:
        """

        # 内层
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # 外层
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
        # 维度[batch_size, sequence_length, embedding_size]
        outputs = tf.layers.conv1d(**params)

        # 残差连接
        outputs += inputs

        # 归一化处理
        outputs = self._layer_normalization(outputs)

        return outputs

    def _position_embedding(self):
        """
        生成位置向量
        :return:
        """
        batch_size = self.config["batch_size"]
        sequence_length = self.config["sequence_length"]
        embedding_size = self.config["embedding_size"]

        # 生成位置的索引，并扩张到batch中所有的样本上
        position_index = tf.tile(tf.expand_dims(tf.range(sequence_length), 0), [batch_size, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        position_embedding = np.array([[pos / np.power(10000, (i - i % 2) / embedding_size)
                                        for i in range(embedding_size)]
                                       for pos in range(sequence_length)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        embedded_position = tf.nn.embedding_lookup(position_embedding, position_index)

        return embedded_position
