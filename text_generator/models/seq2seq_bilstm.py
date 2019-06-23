import tensorflow as tf
from tensorflow.python.util import nest

from .base import BaseModel


class Seq2SeqBiLstmModel(BaseModel):
    def __init__(self, config, vocab_size=None, word_vectors=None, mode="train"):
        super(Seq2SeqBiLstmModel, self).__init__(config)
        self.learning_rate = self.config["learning_rate"]
        self.embedding_size = self.config["embedding_size"]
        self.encoder_hidden_sizes = self.config["encoder_hidden_sizes"]
        self.decoder_hidden_sizes = self.config["decoder_hidden_sizes"]
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors
        self.mode = mode
        self.use_attention = self.config["use_attention"]
        self.beam_search = self.config["beam_search"]
        self.beam_size = self.config["beam_size"]
        self.max_gradient_norm = self.config["max_grad_norm"]

        # 如果是解码状态batch_size设为1
        if self.mode == "decode":
            self.batch_size = 1
        else:
            self.batch_size = config["batch_size"]

        # 主要在decode阶段用来做beam search
        self.beam_batch_size = self.batch_size

        self.go_token = 2
        self.eos_token = 3

        # 初始化构建模型
        self.build_model()

        # 初始化saver对象
        self.init_saver()

    def _multi_rnn_cell(self, hidden_sizes):
        """
        创建多层cell
        :return:
        """
        def get_lstm_cell(hidden_size):
            """
            创建单个cell ，并添加dropout
            :param hidden_size:
            :return:
            """
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True,
                                                initializer=tf.orthogonal_initializer())

            drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)

            return drop_cell

        # 创建多层cell
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(hidden_size) for hidden_size in hidden_sizes])

        return multi_cell

    def encoder(self):
        """
        定义encoder部分
        :return: 编码结果，编码后的隐层状态，
        """
        with tf.name_scope("encoder"):

            # embedding 层
            with tf.variable_scope("embedding_layer"):
                if self.word_vectors is not None:
                    embedding = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                            name="embedding_w")
                else:
                    embedding = tf.get_variable("embedding_w", shape=[self.vocab_size, self.embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())
                embedded_words = tf.nn.embedding_lookup(embedding, self.encoder_inputs)

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
        return embedded_words, tuple_states, embedding

    def decode(self, encoder_output, encoder_state, encoder_inputs_length, embedding):
        """
        定义decoder部分， 训练模式下会返回输出，解码模式下没有返回值
        :param encoder_output: encoder的输入
        :param encoder_state: encoder的状态
        :param encoder_inputs_length: encoder的输入长度
        :param embedding: 共享encoder的embedding
        :return:
        """

        with tf.name_scope("decoder"):
            if self.beam_search and self.mode == "decode":
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                # 该方法和tf.tile有所不同，这个方法得到的结果是b0, b0, b1, b1, ...
                encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=self.beam_size)
                # 对encoder state中各元素做上面相同的操作，在这里用到了nest.map_structure函数，
                # 是因为state中包括lstm中的h和c
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, multiplier=self.beam_size),
                                                   encoder_state)
                # 对encoder_inputs_length做上面相同的操作
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(encoder_inputs_length,
                                                                      multiplier=self.beam_size)
                # 如果使用beam_search则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次
                self.beam_batch_size *= self.beam_size

            # 定义要使用的attention机制，传入了memory_sequence_length就会对<pad>字符进行mask处理
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.encoder_hidden_sizes[-1] * 2,
                                                                       memory=encoder_output,
                                                                       memory_sequence_length=encoder_inputs_length)

            # 定义decoder阶段要用的LSTMCell，然后为其封装attention wrapper
            decoder_cell = self._multi_rnn_cell(self.decoder_hidden_sizes)
            # 引入attention的context输入到decoder中
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                               attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.encoder_hidden_sizes[-1] * 2,
                                                               name='Attention_Wrapper')

            # 定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            decoder_initial_state = decoder_cell.zero_state(batch_size=self.beam_batch_size, dtype=tf.float32).clone(
                cell_state=encoder_state)

            # 用在后面解码阶段
            output_layer = tf.layers.Dense(self.vocab_size,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            if self.mode == "train":
                # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<pad>,并进行embedding。
                # 将最后一个time step去掉，这种做法是因为在t时刻解码时输入的词应该时t-1时刻的词，类似于用上一个词预测当前词。
                ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
                # 加上开始符
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.go_token), ending], 1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)

                # 下面是训练时解码的固定搭配
                training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs=decoder_inputs_embedded,
                    sequence_length=self.decoder_targets_length,
                    embedding=embedding,
                    sampling_probability=0.4,
                    time_major=False, name="training_helper")

                # training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                #                                                     sequence_length=self.decoder_targets_length,
                #                                                     time_major=False,
                #                                                     name='training_helper')
                # cell是定义好的带attention的decoder层，
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                                   initial_state=decoder_initial_state,
                                                                   output_layer=output_layer)

                # 调用dynamic_decode进行解码，返回decoder_outputs, decode_state, decode_length
                # decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
                # sample_id: [batch_size, decoder_target_length], tf.int32，保存最终的编码结果。可以表示最后的答案
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=training_decoder, impute_finished=True, maximum_iterations=self.max_target_sequence_length)

                # 训练模式下定义训练的方法
                self.train_method(decoder_outputs)

            elif self.mode == 'decode':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.go_token
                end_token = self.eos_token
                # decoder阶段根据是否使用beam_search决定不同的组合，
                # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
                # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码
                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=embedding,
                                                                             start_tokens=start_tokens,
                                                                             end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                               start_tokens=start_tokens,
                                                                               end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                        helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)

                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder)
                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
                # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                # sample_id: [batch_size, decoder_targets_length], tf.int32

                # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
                # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
                # beam_search_decoder_output: (scores, predicted_ids, parent_ids)
                # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
                if self.beam_search:
                    self.predictions = decoder_outputs.predicted_ids
                else:
                    self.predictions = decoder_outputs.sample_id

    def train_method(self, decoder_outputs):
        """
        定义训练方法
        :param decoder_outputs: 训练时解码的输出
        :return:
        """

        # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
        self.logits = tf.identity(decoder_outputs.rnn_output)
        self.predictions = tf.argmax(self.logits, axis=-1, name='decoder_pred_train')

        # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志，mask的位置不会加入计算
        self.loss = self.cal_loss()

        # 定义训练的op入口
        self.train_op, self.summary_op = self.get_train_op()

    def build_model(self):
        """
        构建计算图
        :return:
        """
        # 构建encoder部分
        encoder_output, encoder_state, embedding = self.encoder()

        # 构建decoder部分
        self.decode(encoder_output, encoder_state, self.encoder_inputs_length, embedding)
