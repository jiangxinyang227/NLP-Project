### 基于bert做下游任务
#### 文件存放
* bert：存放bert源码，主要会用到其中的modeling，optimizer，tokenizer等文件。
* models：存放下游任务的模型文件，如text_cnn，bilstm_crf等。
* bert_blstm_crf.py：bert + bilstm_crf用于NER任务
* bert_cnn.py：bert + text_cnn用于分类任务
* run_classifier.py：bert用于分类任务

#### 训练模型的启动命令行都在run.sh文件中

#### 训练时的几点建议
* 对于句子较短时，尽量将max_seq_length调小，这样可以加快训练速度，此外在GPU资源一定的时候，
   减小句子长度，可以适当提高batch_size，不过fine-tuning的batch_size不宜过大，16或者32比较合适。
   
* 训练后得到的模型可以将保存的和预测时无关的参数变量去掉，只要变量名含有adam的变量都可以去掉。
   这样可以大大的缩小模型的内存大小。
   
* 绝大部分分类问题，都可以直接使用bert来分类，不需要再接如text_cnn这样的下游任务，这里只是提供一种和
   下游模型融合的做法。

* 对于会有多种途径的数据融合时，可以在create_model 方法中进行融合，也就是将bert最后一层输出的句子向量和其他的
   特征向量融合，比如tf-idf向量。