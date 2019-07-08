### 文本生成

#### 数据预处理
要求训练集和测试集分开存储，对于中文的数据必须先分词，对分词后的词用空格符分开，并且将原句子和木匾句子拼接在一行，用分隔符\<SEP>分开。具体的如下：
* 在线 欧 弟 强悍 表演 绝对 经典 \<SEP> 欧 弟 这 小子 确实 有才 呀 ， 推荐 一看

#### 文件结构介绍
* config文件：配置各种模型的配置参数
* data：存放训练集和测试集
* data_helpers：提供数据处理的方法，其中含bpe的文件是提供了bpe分词。
* ckpt_model：存放checkpoint模型文件
* pb_model：存放pb模型文件
* outputs：存放vocab，word_to_index, label_to_index, 处理后的数据
* models：存放模型代码
* trainers：存放训练代码
* predictors：存放预测代码

#### 训练模型
* python train.py --config_path="config/seq2seq_bigru_config.json"

#### 预测模型
* 预测代码都在predictors/predict.py中，初始化Predictor对象，调用predict方法即可。

#### 模型的配置参数详述

##### bigru：基于bigru的文本生成
* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* encoder_hidden_sizes：encoder层的隐层大小，列表对象，支持多层
* decoder_hidden_sizes：decoder层的隐层大小，列表对象，层数尽量保持和encoder一致
* batch_size：批样本大小
* vocab_size：词汇表大小
* keep_prob：保留神经元的比例
* warmup_step：预热步数，作用在学习速率上的
* smooth_rate：标签平滑的比例，标签平滑技术
* decode_step：预测时解码最大步数
* schedule_sample：是否采用计划采样
* beam_search：是否采用beam search
* beam_size：beam size大小
* use_bpe：是否采用bpe分词
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* stop_word：停用词表的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径

##### conv：基于conv的文本生成
* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* hidden_size：全连接层的隐层大小
* batch_size：批样本大小
* vocab_size：词汇表大小
* keep_prob：保留神经元的比例
* num_layers：卷积的层数
* kernel_size：卷积核的大小
* num_filters：卷积的数量，必须取hidden_size的2倍
* warmup_step：预热步数，作用在学习速率上的
* smooth_rate：标签平滑的比例，标签平滑技术
* decode_step：预测时解码最大步数
* schedule_sample：是否采用计划采样
* beam_search：是否采用beam search
* beam_size：beam size大小
* use_bpe：是否采用bpe分词
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* stop_word：停用词表的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径

##### transformer：基于transformer的文本生成
* model_name：模型名称
* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小，必须为num_heads的倍数
* hidden_size：feed forward层采用全连接层，这个为全连接层的隐层大小
* batch_size：批样本大小
* vocab_size：词汇表大小
* keep_prob：保留神经元的比例
* num_heads：transformer中attention的头数
* num_blocks：transformer的层数
* ln_epsilon：layer normalization中除数中的极小值
* warmup_step：预热步数，作用在学习速率上的
* smooth_rate：标签平滑的比例，标签平滑技术
* decode_step：预测时解码最大步数
* schedule_sample：是否采用计划采样
* beam_search：是否采用beam search
* beam_size：beam size大小
* use_bpe：是否采用bpe分词
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* stop_word：停用词表的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径

