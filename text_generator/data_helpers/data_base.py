

class DataBase(object):
    def __init__(self, config):
        self.config = config

    def read_data(self):
        """
        读取数据
        """
        raise NotImplementedError

    @staticmethod
    def trans_to_index(data, word_to_index):
        """
        将数据转换成索引表示
        """
        raise NotImplementedError

    def padding(self, batch):
        """
        对序列进行补全
        :return:
        """
        raise NotImplementedError

    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        raise NotImplementedError

    def next_batch(self, data, batch_size):
        """
        生成batch数据
        :return:
        """
        raise NotImplementedError


class TrainDataBase(DataBase):
    def __init__(self, config):
        super(TrainDataBase, self).__init__(config)
        self.questions = None
        self.responses = None

        self.vocab_size = None
        self.word_vectors = None

    def read_data(self):
        """
        读取数据
        """
        raise NotImplementedError

    def get_word_vectors(self, words):
        """
        加载词向量
        :return:
        """
        raise NotImplementedError

    def gen_vocab(self, questions, responses):
        """
        根据问答对生成词汇表
        :param questions: 问题
        :param responses: 回答
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def trans_to_index(data, word_to_index):
        """
        将数据转换成索引表示
        """
        raise NotImplementedError

    def padding(self, batch):
        """
        对序列进行补全
        :return:
        """
        raise NotImplementedError

    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        raise NotImplementedError

    def next_batch(self, data, batch_size):
        """
        生成batch数据
        :return:
        """
        raise NotImplementedError


class EvalPredictDataBase(DataBase):
    def __init__(self, config):
        super(EvalPredictDataBase, self).__init__(config)

        self.word_to_index = None

    def read_data(self):
        """
        读取数据
        """
        raise NotImplementedError

    def load_vocab(self):
        """
        生成词汇表
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def trans_to_index(data, word_to_index):
        """
        将数据转换成索引表示
        """
        raise NotImplementedError

    def padding(self, batch):
        """
        对序列进行补全
        :return:
        """
        raise NotImplementedError

    def gen_data(self):
        """
        生成可导入到模型中的数据
        :return:
        """
        raise NotImplementedError

    def next_batch(self, data, batch_size):
        """
        生成batch数据
        :return:
        """
        raise NotImplementedError
