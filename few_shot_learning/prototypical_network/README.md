### prototypical_network


#### config文件
* epochs ：迭代的步数
* checkpoint_every：训练多少步保存一次模型
* eval_every：训练多少步验证一次模型
* optimization：优化算法，建议adam，一般都能快速收敛
* learning_rate：学习速率
* embedding_size：嵌入层大小
* hidden_sizes：lstm单元数量
* num_support：shot的数量，即支撑集的数量
* num_queries：训练时query的数量
* num_classes：way的数量，即类别的数量
* num_tasks：每个epoch中采样的数量
* batch_size：
* keep_prob：dropout后的比例
* sequence_length：序列固定长度
* low_freq：低频词的阈值
* neg_threshold：对比损失函数中的负例临界值
* l2_reg_lambda：l2正则化系数
* max_grad_norm：梯度截断值
* train_data：训练数据路径
* eval_data：验证数据路径
* stop_word：停用词文件路径
* output_path：输出文件路径
* word_vector_path：词向量文件路径
* ckpt_model_path：checkpoint 文件路径
