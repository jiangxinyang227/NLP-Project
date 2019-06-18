import os


class PredictorBase(object):
    def __init__(self, config):

        self.output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                        config["output_path"])

    def load_vocab(self):
        """
        加载词汇表
        :return:
        """
        raise NotImplementedError

    def sentence_to_idx(self, sentence):
        """
        创建数据对象
        :return:
        """
        raise NotImplementedError

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        raise NotImplementedError

    def predict(self, sentence):
        """
        训练模型
        :return:
        """
        raise NotImplementedError


