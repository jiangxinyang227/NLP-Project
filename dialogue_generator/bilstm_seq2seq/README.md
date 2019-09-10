### bigru seq2seq

#### config文件

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
* beam_search：是否采用beam search
* beam_size：beam size大小
* max_grad_norm：梯度阶段临界值
* train_data：训练数据的存储路径
* eval_data：验证数据的存储路径
* stop_word：停用词表的存储路径
* output_path：输出路径，用来存储vocab，处理后的训练数据，验证数据
* word_vectors_path：词向量的路径
* ckpt_model_path：checkpoint 模型的存储路径
* pb_model_path：pb 模型的存储路径
