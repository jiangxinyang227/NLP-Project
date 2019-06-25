import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from bpemb import BPEmb
import tensorflow as tf
from predict_base import PredictorBase
from models import Seq2SeqLstmModel, Seq2SeqBiLstmModel


class Predictor(PredictorBase):
    def __init__(self, config):
        super(Predictor, self).__init__(config)
        self.config = config
        self.model = None
        self.sess = None
        # self.builder = tf.saved_model.builder.SavedModelBuilder("savedModel")

        if self.config["use_bpe"]:
            self.bpe_zh = BPEmb(lang="zh", vs=config["vocab_size"])
        else:
            # 加载词汇表
            self.word_to_idx = self.load_vocab()
            self.idx_to_label = {value: key for key, value in self.word_to_idx.items()}

        # 初始化模型
        self.create_model()
        print("load model finished")
        # 加载计算图
        self.load_graph()
        print("load graph finished")

    def load_vocab(self):
        # 将词汇-索引映射表加载出来
        with open(os.path.join(self.output_path, "word_to_index.pkl"), "rb") as f:
            word_to_index = pickle.load(f)

        return word_to_index

    def sentence_to_encode(self, sentence):
        """
        创建数据对象
        :return:
        """
        if not sentence:
            return None

        if len(sentence) > 20:
            return None

        if self.config["use_bpe"]:
            word_idx = self.bpe_zh.encode_ids(sentence)
            word_idx = list(map(lambda x: x+1, word_idx))
        else:
            word_idx = [self.word_to_idx.get(token, self.word_to_idx["UNK"]) for token in sentence]

        new_word_idx = self.process_data(word_idx)
        return new_word_idx

    @staticmethod
    def process_data(sentence):
        """
        对数据做预处理
        :param sentence:
        :return:
        """
        questions = [sentence]
        question_length = [len(sentence)]
        return dict(questions=questions, question_length=question_length)

    def response(self, tokens_list):
        sents = []
        for i in range(self.config["beam_size"]):
            sent_token = tokens_list[:, i]
            if self.config["use_bpe"]:
                sent = self.bpe_zh.decode_ids(list(map(lambda x: x-1, sent_token)))
            else:
                sent = "".join([self.idx_to_label[token] for token in sent_token])
            sents.append(sent)

        return sents

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        if self.config["model_name"] == "seq2seq_lstm":
            self.model = Seq2SeqLstmModel(config=self.config, vocab_size=len(self.word_to_idx),
                                          word_vectors=None, mode="decode")

        if self.config["model_name"] == "seq2seq_bilstm":
            self.model = Seq2SeqBiLstmModel(config=self.config, vocab_size=len(self.word_to_idx),
                                            word_vectors=None, mode="decode")

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                                          self.config["ckpt_model_path"]))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))
        # inputs = {"inputs": tf.saved_model.utils.build_tensor_info(self.model.encoder_inputs),
        #           "inputs_length": tf.saved_model.utils.build_tensor_info(self.model.encoder_inputs_length),
        #           "keep_prob": tf.saved_model.utils.build_tensor_info(self.model.keep_prob)}
        #
        # outputs = {"predictions": tf.saved_model.utils.build_tensor_info(self.model.predictions)}
        #
        # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
        #                                                                               outputs=outputs,
        #                                                                               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        # legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        # self.builder.add_meta_graph_and_variables(self.sess, [tf.saved_model.tag_constants.SERVING],
        #                                           signature_def_map={"dialogue": prediction_signature},
        #                                           legacy_init_op=legacy_init_op)

        # self.builder.save()

    def predict(self, sentence):
        """
         给定一条句子，预测结果
        :return:
        """
        sentence_ids = self.sentence_to_encode(sentence)
        prediction = self.model.infer(self.sess, sentence_ids).reshape(-1, self.config["beam_size"])
        print(prediction.shape)
        response = self.response(prediction)
        return response

