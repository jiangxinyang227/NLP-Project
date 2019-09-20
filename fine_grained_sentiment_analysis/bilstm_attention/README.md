#### 多粒度情感分析

* 数据来源于 AI challenger 2018中的情感分析数据集。从整个数据集中选择了环境，卫生和服务三个维度的数据
* 链接：https://pan.baidu.com/s/1Ikr70E7PR42bhBuqqp_Zyg 提取码：6esv 

#### bilstm + attention
* 首先执行word_embedded中的word2vec.py文件，生成词向量
* 然后执行python train.py --config_path=config.json训练模型

#### config文件

* epochs：全样本迭代次数
* checkpoint_every：迭代多少步保存一次模型文件
* eval_every：迭代多少步验证一次模型
* learning_rate：学习速率
* optimization：优化算法
* embedding_size：embedding层大小
* hidden_sizes：lstm的隐层大小，列表对象，支持多层lstm，只要在列表中添加相应的层对应的隐层大小
* low_seq：低频词的阈值
* batch_size：批样本大小
* sequence_length：序列长度
* num_aspects：即aspect的数量，在这里有环境，服务，卫生三个
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