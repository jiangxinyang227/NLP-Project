import tensorflow as tf
from .base import BaseModel


class TextCnnModel(BaseModel):
    def __init__(self, config, vocab_size, word_vectors):
        super(TextCnnModel, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):
        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            # if self.word_vectors is not None:
            #     embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
            #                               name="embedding_w")
            # else:
            embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config["embedding_size"]],
                                          initializer=tf.contrib.layers.xavier_initializer())

            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs)
            # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
            embedded_words_expand = tf.expand_dims(embedded_words, -1)

        # 创建卷积和池化层
        pooled_outputs = []
        # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        for i, filter_size in enumerate(self.config["filter_sizes"]):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                filter_shape = [filter_size, self.config["embedding_size"], 1, self.config["num_filters"]]
                conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_w")
                conv_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_filters"]]), name="conv_b")
                conv = tf.nn.conv2d(
                    embedded_words_expand,
                    conv_w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")
                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config["sequence_length"] - filter_size + 1, 1, 1],
                    # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

        # 得到CNN网络的输出长度
        num_filters_total = self.config["num_filters"] * len(self.config["filter_sizes"])

        # 池化后的维度不变，按照最后的维度channel来concat
        h_pool = tf.concat(pooled_outputs, 3)

        # 摊平成二维的数据输入到全连接层
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[num_filters_total, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(h_drop, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        # 计算交叉熵损失
        self.loss = self.cal_loss() + self.config["l2_reg_lambda"] * self.l2_loss
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()

