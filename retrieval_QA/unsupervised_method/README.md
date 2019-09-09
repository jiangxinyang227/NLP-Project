### 无监督的检索式问答
#### 目前提供了tf-idf和编辑距离，这里提供了相同的api来计算相似度。而且输入的问答对的数据格式也是一致的。
***
#### 问答对的数据格式（也就是知识库的数据格式）
[{
"question_id": "0", "query_id": "0", "query": "你好啊", "answer": "你好"
},
{"question_id": "0", "query_id": "1", "query": "你好", "answer": "你好"}
]

上面提供了两条数据，因为在实际的数据中一般会有一个标准问和很多相似问，以及对应的一个标准答案。
所以上面的question_id表示一个整体问题的id（包括标准问和相似问，因此question_id存在重复的）,query_id表示一个独立问题的id（即标准问和相似问的不一样，
因此query_id是唯一的），query就是对应的问句，例如上面的”你好啊“作为标准问，则“你好”就是它的相似问，
answer就是对应的答案，标准问和相似问有相同的答案。

#### api

1. get_top_n_answer(self, sentence, n=15, interval=0.2, answer_num=5)
 
   是用来做推荐回复的，即有些情况下会回复多个回答。
   * sentence： 用户的输入
   * n: 初步筛选出来的候选问答对
   * interval：和最高分数问答对之间的间隔值，如果小于该间隔值，则将这些问答对也加入回复中
   * answer_num：支持的最大回答的数量

2. max_mean_score_answer(self, sentence, n)

   基于候选问答对根据question_id做聚合，然后对每个question_id下的问答对的分数取平均值，然后输出最大平均值对应的question_id
   对应的回答，参数和上面的含义一样
   
3. vote_answer(self, sentence, n=1)
    
    和上面的基本一致，只是上面是取平均，而这里是采用投票的方式，但是这种投票的方法对于相似问比较少的问题是不怎么友好的。
