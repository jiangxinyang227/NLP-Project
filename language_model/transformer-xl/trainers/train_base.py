
class TrainerBase(object):
    def __init__(self):

        self.model = None  # 模型的初始化对象
        self.config = None  # 模型的配置参数
        self.current_step = 0

    def load_data(self):
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

    def train(self):
        """
        训练模型
        :return:
        """
        raise NotImplementedError


