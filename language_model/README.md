### 语言模型

#### 数据预处理
中文语言模型基本都是基于字的模型，因此不需要做太多的操作

#### 文件结构介绍
* config文件：配置各种模型的配置参数
* data：存放训练集和测试集
* data_helpers：提供数据处理的方法
* ckpt_model：存放checkpoint模型文件
* pb_model：存放pb模型文件
* outputs：存放vocab，word_to_index, label_to_index, 处理后的数据
* models：存放模型代码
* trainers：存放训练代码
* predictors：存放预测代码

#### 训练模型
* python train.py --config_path="config/char_rnn_config.json"

#### 预测模型
* 预测代码都在predictors/predict.py中，初始化Predictor对象，调用predict方法即可。

#### 模型的配置参数详述

#### char rn：字符级的rnn，基于字符的语言模型
* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* hidden_sizes：rnn隐层大小
* batch_size：批样本大小
* sequence_length：序列长度
* vocab_size：词汇表大小
* keep_prob：保留神经元的比例
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径