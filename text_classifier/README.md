### 文本分类

#### 数据预处理
要求训练集和测试集分开存储，对于中文的数据必须先分词，对分词后的词用空格符分开，并且将标签连接到每条数据的尾部，标签和句子用分隔符\<SEP>分开。具体的如下：
* 今天 的 天气 真好\<SEP>积极

#### 文件结构介绍
* config文件：配置各种模型的配置参数
* data：存放训练集和测试集
* ckpt_model：存放checkpoint模型文件
* data_helpers：提供数据处理的方法
* pb_model：存放pb模型文件
* outputs：存放vocab，word_to_index, label_to_index, 处理后的数据
* models：存放模型代码
* trainers：存放训练代码
* predictors：存放预测代码

#### 训练模型
* python train.py --config_path="config/textcnn_config.json"

#### 预测模型
* 预测代码都在predictors/predict.py中，初始化Predictor对象，调用predict方法即可。

#### 模型的配置参数详述

##### textcnn：基于textcnn的文本分类
* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* num_filters：卷积核的数量
* filter_sizes：卷积核的尺寸
* batch_size：批样本大小
* sequence_length：序列长度
* vocab_size：词汇表大小
* num_classes：样本的类别数，二分类时置为1，多分类时置为实际类别数
* keep_prob：保留神经元的比例
* l2_reg_lambda：L2正则化的系数，主要对全连接层的参数正则化
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* stop_word：停用词表的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径

##### bilstm：基于bilstm的文本分类
* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* hidden_sizes：lstm的隐层大小，列表对象，支持多层lstm，只要在列表中添加相应的层对应的隐层大小
* batch_size：批样本大小
* sequence_length：序列长度
* vocab_size：词汇表大小
* num_classes：样本的类别数，二分类时置为1，多分类时置为实际类别数
* keep_prob：保留神经元的比例
* l2_reg_lambda：L2正则化的系数，主要对全连接层的参数正则化
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* stop_word：停用词表的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径

##### bilstm atten：基于bilstm + attention 的文本分类
* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* hidden_sizes：lstm的隐层大小，列表对象，支持多层lstm，只要在列表中添加相应的层对应的隐层大小
* batch_size：批样本大小
* sequence_length：序列长度
* vocab_size：词汇表大小
* num_classes：样本的类别数，二分类时置为1，多分类时置为实际类别数
* keep_prob：保留神经元的比例
* l2_reg_lambda：L2正则化的系数，主要对全连接层的参数正则化
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* stop_word：停用词表的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径

##### rcnn：基于rcnn的文本分类
* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* hidden_sizes：lstm的隐层大小，列表对象，支持多层lstm，只要在列表中添加相应的层对应的隐层大小
* output_size：lstm层的输出非线性映射的神经元大小
* batch_size：批样本大小
* sequence_length：序列长度
* vocab_size：词汇表大小
* num_classes：样本的类别数，二分类时置为1，多分类时置为实际类别数
* keep_prob：保留神经元的比例
* l2_reg_lambda：L2正则化的系数，主要对全连接层的参数正则化
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* stop_word：停用词表的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径

##### transformer：基于transformer的文本分类
* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* filters：feed forward是用卷积实现的，这里为第一层卷积层的卷积核数量，第二层的卷积核数量要等于embedding size，在这里也可以用全连接层
* num_heads：self attention 的头数
* num_blocks：encoder块的数量，类似于卷积的层数
* ln_epsilon：layer norm中的除法中引入的极小数值
* batch_size：批样本大小
* sequence_length：序列长度
* vocab_size：词汇表大小
* num_classes：样本的类别数，二分类时置为1，多分类时置为实际类别数
* keep_prob：保留神经元的比例
* l2_reg_lambda：L2正则化的系数，主要对全连接层的参数正则化
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* stop_word：停用词表的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径