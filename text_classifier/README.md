### 语言模型

#### textcnn：基于textcnn的多标签分类模型，多标签分类模型主要在损失函数的设计上不一样
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
* num_classes：样本的类别数
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