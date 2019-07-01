import tensorflow as tf


# 构建模型
class TextCNN(object):
    """
    Text CNN 用于文本分类
    """
    def __init__(self, embedded_chars, filter_sizes, num_filter, labels, num_label,
                 dropout_rate, max_len, is_training):
        """
        构建text-cnn模型
        :param embedded_chars: bert输出的词向量
        :param filter_sizes: 卷积核的尺寸
        :param num_filter: 卷积核的数量
        :param labels: 标签
        :param num_label: 标签的数量
        :param dropout_rate: dropout比例
        :param is_training:
        """
        self.embedded_chars = embedded_chars
        self.filter_sizes = filter_sizes
        self.num_filter = num_filter
        self.labels = labels
        self.num_label = num_label
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.is_training = is_training
        self.embedding_size = embedded_chars.shape[-1].value

    def _text_cnn(self):
        # 创建卷积和池化层
        embedded_chars_expands = tf.expand_dims(self.embedded_chars, -1)
        pooled_outputs = []
        # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filter]
                W = tf.get_variable("cnn_w",
                                    initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable("cnn_b",
                                    initializer=tf.constant(0.1, shape=[self.num_filter]))
                conv = tf.nn.conv2d(
                    embedded_chars_expands,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_len - filter_size + 1, 1, 1],
                    # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

        # 得到CNN网络的输出长度
        num_filters_total = self.num_filter * len(self.filter_sizes)

        # 池化后的维度不变，按照最后的维度channel来concat
        h_pool = tf.concat(pooled_outputs, 3)

        # 摊平成二维的数据输入到全连接层
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        return h_pool_flat, num_filters_total

    def _output_layer(self, input, input_shape):
        # dropout

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(input, self.dropout_rate)

        # 全连接层的输出
        with tf.variable_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[input_shape, self.num_label],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.get_variable("output_b",
                                       initializer=tf.constant(0.1, shape=[self.num_label]))

            logits = tf.nn.xw_plus_b(h_drop, output_w, output_b, name="predictions")

        return logits

    def _cal_loss(self, logits):
        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
            loss = tf.reduce_mean(losses)

        return loss

    def _get_prediction(self, logits):
        return tf.argmax(logits, -1)

    def gen_result(self):
        input, input_shape = self._text_cnn()
        logits = self._output_layer(input, input_shape)
        loss = self._cal_loss(logits)
        predictions = self._get_prediction(logits)

        return (loss, logits, predictions)