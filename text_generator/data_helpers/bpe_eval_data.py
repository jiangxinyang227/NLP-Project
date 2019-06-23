import os
import pickle
import random

from bpemb import BPEmb


class BpeEvalData(object):
    def __init__(self, config):

        self._eval_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config["eval_data"])
        self._output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                         config["output_path"])

        self.bpe_zh = BPEmb(lang="zh", vs=config["vocab_size"])
        self.pad_token = 0
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
                requests.append(request.strip())
                responses.append(response.strip())
        return requests, responses

    def trans_to_index(self, data):
        """
        将输入转化为索引表示
        :param data: 输入的是questions 和 responses

        :return:
        """
        data_ids = []
        for sentence in data:
            token_ids = self.bpe_zh.encode_ids(sentence)
            # 因为在bpe的vocab中的0位置添加了<pad>字符，bpe vocab对应的索引是从1开始，因此对这里的ids都加1
            token_ids = list(map(lambda x: x + 1, token_ids))
            data_ids.append(token_ids)
        return data_ids

    def padding(self, batch):
        """
        对每个batch数据按数据集中最大长度的句子进行补全
        :param batch:
        :return:
        """
        question_length = [len(sample[0]) for sample in batch]
        max_question_length = max(question_length)
        questions = [sample[0] + [self.pad_token] * (max_question_length - len(sample[0]))
                     for sample in batch]

        # 在这里先对response加上一个终止符<eos>
        responses = [sample[1] + [self.eos_token] for sample in batch]
        response_length = [len(response) for response in responses]
        max_response_length = max(response_length)

        # 对response按最大长度补齐
        pad_responses = [response + [self.pad_token] * (max_response_length - len(response)) for response in responses]

        return dict(questions=questions, responses=pad_responses,
                    question_length=question_length, response_length=response_length)

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

        # 3，输入转索引
        questions_idx = self.trans_to_index(questions)
        responses_idx = self.trans_to_index(responses)

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