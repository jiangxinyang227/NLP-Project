### lstm_siamese
#### 基于lstm的siamese网络
***
#### 数据格式（也就是知识库的数据格式）
整个数据的读取在data_hepler.py文件中的load_data方法中定义，返回的结果必须是
[[标准问1，相似问1_1，相似问1_2，......], [标准问2， 相似问2_1, 相似问2_2], ......]
然后返回的句子必须是已经分好词的，例如：
* 标准问1：["请", "帮我", "查询", "xxx"]

#### 模型讲解

1. 构造正样本对和负样本对来训练模型，比例1：1，标签为0，1。

2. 实际预测时和训练时不一致，预测时需要输出句子之间的相似度值。
   
***

#### config文件
* epochs ：迭代的步数
* checkpoint_every：训练多少步保存一次模型
* eval_every：训练多少步验证一次模型
* optimization：优化算法，建议adam，一般都能快速收敛
* learning_rate：学习速率
* embedding_size：嵌入层大小
* hidden_sizes：lstm单元数量
* batch_size：
* keep_prob：dropout后的比例
* low_freq：低频词的阈值
* neg_threshold：对比损失函数中的负例临界值
* n_tasks：每个epoch时的采样数量
* max_grad_norm：梯度截断值
* train_data：训练数据路径
* stop_word：停用词文件路径
* output_path：输出文件路径
* word_vector_path：词向量文件路径
*ckpt_model_path：checkpoint 文件路径
