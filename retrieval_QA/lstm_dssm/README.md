### lstm_siamese
#### 基于lstm的siamese网络
***
#### 数据格式（也就是知识库的数据格式）
整个数据的读取在data_hepler.py文件中的load_data方法中定义，返回的结果必须是
[[标准问1，相似问1_1，相似问1_2，......], [标准问2， 相似问2_1, 相似问2_2], ......]
然后返回的句子必须是已经分好词的，例如：
* 标准问1：["请", "帮我", "查询", "xxx"]

#### 模型讲解

1. 构造训练的样本对，query为单个句子，sim_query为多个句子，其中第一个为正例，其余的为负例。

2. 预测时的代码可以参考dnn_dssm中的预测方法，和训练时不同

3. 我的实践测试triplet_loss的效果要优于softmax

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
* neg_samples：每条训练数据中负样本的数量，当为1时使用triplet_loss，当>1时，使用softmax+交叉熵损失
* n_tasks：每个epoch时的采样数量
* margin：triplet_loss中的间隔值
* max_grad_norm：梯度截断值
* train_data：训练数据路径
* stop_word：停用词文件路径
* output_path：输出文件路径
* word_vector_path：词向量文件路径
*ckpt_model_path：checkpoint 文件路径