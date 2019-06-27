import os
import pickle
import random

from .data_base import EvalPredictDataBase


class EvalData(EvalPredictDataBase):
    def __init__(self, config):
        super(EvalData, self).__init__(config)

        self._eval_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config["eval_data"])
        self._output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                         config["output_path"])

        self.pad_token = 0
        self.go_token = 2
        self.eos_token = 3

    def read_data(self):
        """
        读取数据
        :return: 返回分词后的文本内容和标签，questions, responses = [[]]
        """
        with open(self._eval_data_path, "r", encoding="utf8") as f:
            requests = []
            responses = []
            for line in f.readlines():
                request, response = line.strip().split("<SEP>")
                requests.append(request.strip().split(" "))
                responses.append(response.strip().split(" "))
        return requests, responses

    def load_vocab(self):
        """
        加载词汇和标签的映射表
        :return:
        """
        # 将词汇-索引映射表加载出来
        with open(os.path.join(self._output_path, "word_to_index.pkl"), "rb") as f:
            word_to_index = pickle.load(f)

        return word_to_index

    @staticmethod
    def trans_to_index(data, word_to_index):
        """
        将输入转化为索引表示
        :param data: 输入的是questions 和 responses
        :param word_to_index: 词汇-索引映射表
        :return:
        """
        data_idx = []
        for sentence in data:
            sentence_idx = []
            for word in sentence:
                if word in word_to_index:
                    sentence_idx.append(word_to_index[word])
                else:
                    sentence_idx.append(word_to_index["<UNK>"])
            data_idx.append(sentence_idx)

        return data_idx

    def padding(self, batch):
        """
        对每个batch数据按数据集中最大长度的句子进行补全
        :param batch:
        :return:
        """
        # 对question添加结束符
        questions = [sample[0] + [self.eos_token] for sample in batch]
        question_length = [len(question) for question in questions]
        max_question_length = max(question_length)
        encoder_inputs = [question + [self.pad_token] * (max_question_length - len(question))
                          for question in questions]

        # 在这里先对response加上一个开始符和结束符<eos>
        responses = [[self.go_token] + sample[1] + [self.eos_token] for sample in batch]
        decoder_inputs = [response[:-1] for response in responses]
        decoder_outputs = [response[1:] for response in responses]

        decoder_inputs_length = [len(response) for response in decoder_inputs]
        max_decoder_inputs_length = max(decoder_inputs_length)

        decoder_outputs_length = [len(response) for response in decoder_outputs]
        max_decoder_outputs_length = max(decoder_outputs_length)

        # 按最大长度补齐
        decoder_inputs = [response + [self.pad_token] * (max_decoder_inputs_length - len(response))
                          for response in decoder_inputs]
        decoder_outputs = [response + [self.pad_token] * (max_decoder_outputs_length - len(response))
                           for response in decoder_outputs]

        return dict(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, decoder_outputs=decoder_outputs,
                    encoder_max_len=max_question_length, decoder_max_len=max_decoder_inputs_length)

    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        # 如果不是第一次数据预处理，则直接读取
        if os.path.exists(os.path.join(self._output_path, "eval_data.pkl")):
            print("load existed eval data")
            with open(os.path.join(self._output_path, "eval_data.pkl"), "rb") as f:
                eval_data = pickle.load(f)
            return eval_data

        # 1，读取原始数据
        questions, responses = self.read_data()

        # 2，得到词汇表
        word_to_index = self.load_vocab()

        # 3，输入转索引
        questions_idx = self.trans_to_index(questions, word_to_index)
        responses_idx = self.trans_to_index(responses, word_to_index)

        # 生成数据并保存下来
        eval_data = [[questions_idx[i], responses_idx[i]] for i in range(len(questions_idx))]
        with open(os.path.join(self._output_path, "eval_data.pkl"), "wb") as fw:
            pickle.dump(eval_data, fw)
        return eval_data

    def next_batch(self, data, batch_size):
        """
        生成batch数据集
        :param data: 输入
        :param batch_size: 批量的大小
        :return:
        """
        random.shuffle(data)
        batch_num = len(data) // batch_size

        for i in range(batch_num):
            batch_data = data[batch_size * i: batch_size * (i + 1)]
            new_batch = self.padding(batch_data)
            yield new_batch