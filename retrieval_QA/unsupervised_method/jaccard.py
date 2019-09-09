import jieba
from collections import Counter, defaultdict


class JaccardDistance(object):
    def __init__(self, supports, stop_word=None, n_gram=None):
        """
        jaccard 距离计算相似度
        :param supports: 知识库数据
        :param stop_word: 停用词
        :param n_gram: 支持int 和 float类型
        """
        # 加载停用词
        if stop_word:
            with open(stop_word, "r", encoding="utf8") as fr:
                self.stop_words = [line.strip() for line in fr.readlines()]
        else:
            self.stop_words = []

        self.n_gram = n_gram

        # 加载并处理知识库，便于之后快速预测
        # 给定query_id和question_id 之间的映射，query_id是对应知识库中每个问答对的问题，相似问之间也有不同的query_id。
        # 但相似问的question_id是相同的。
        self.query_to_question = {support["query_id"]: support["question_id"] for support in supports}
        # 给出question_id到answer和标准问之间的映射
        self.question_id_to_answer, self.question_id_to_question = self.get_question_to_answer(supports)
        # 取出知识库中所有的query用于之后计算相似度
        self.queries = {support["query_id"]: support["query"] for support in supports}
        # 将所有的query预先分好词，并去除停用词
        self.queries_token = {query_id: self.get_tokens(query, n_gram=n_gram)
                              for query_id, query in self.queries.items()}

    @staticmethod
    def get_question_to_answer(supports):
        """
        得到question_id到标准问和answer之间的映射
        :param supports:
        :return:
        """
        question_id_to_answer = {}
        question_id_to_question = {}
        id_flag = None
        for support in supports:
            question_id = support["question_id"]
            answer = support["answer"]
            if question_id != id_flag:
                question_id_to_question[question_id] = support["query"]
                id_flag = question_id
            if question_id_to_answer.get(question_id):
                continue
            question_id_to_answer[question_id] = answer
        return question_id_to_answer, question_id_to_question

    @staticmethod
    def get_n_gram(tokens, n_gram):
        """
        返回n_gram分词结果
        :param tokens:
        :param n_gram:
        :return:
        """
        if n_gram is None:
            return tokens

        if isinstance(n_gram, int):
            n_gram_tokens = ["".join(tokens[i: i + n_gram]) for i in range(len(tokens) - n_gram + 1)]
            new_tokens = tokens + n_gram_tokens
            return new_tokens

        if isinstance(n_gram, list):
            n_gram_tokens = ["".join(tokens[i: i + item]) for item in n_gram for i in range(len(tokens) - item + 1)]
            new_tokens = tokens + n_gram_tokens
            return new_tokens

    def get_tokens(self, sentence, n_gram=None):
        """
        分词并去除停用词
        :param sentence:
        :param n_gram:
        :return:
        """
        tokens = jieba.lcut(sentence)
        tokens = [token for token in tokens if token not in self.stop_words]
        new_tokens = self.get_n_gram(tokens, n_gram)
        return new_tokens

    def jaccard(self, query, question):
        """
        计算jaccard 距离
        :param query:
        :param question:
        :return:
        """
        try:
            intersection = len(list(set(query) & set(question)))
            union = len(list(set(query) | set(question)))
            score = round(intersection / union, 4)
        except:
            score = 0
        return score

    def match_scores(self, sentence):
        """
        返回用户查询句和知识库中每个问题的相似分数
        :param sentence: 用户查询的句子
        :return:
        """
        tokens = self.get_tokens(sentence, n_gram=self.n_gram)
        scores = {query_id: self.jaccard(tokens, query_token)
                  for query_id, query_token in self.queries_token.items()}
        sort_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sort_scores

    def get_top_n_scores(self, sentence, n=10):
        """
        返回前n个query和对应的分数
        :param sentence:
        :param n:
        :return:
        """
        sort_scores = self.match_scores(sentence)
        return sort_scores[:n]

    def get_top_n_answer(self, sentence, n=15, interval=0.2, answer_num=5):
        """
        通过给定一个间隔值来决定返回一个或多个回答
        :param sentence: 用户输入的句子
        :param n: 候选的答案
        :param interval: 和最高分数之间的间隔，如果小于该间隔，则返回两个或多个
        :param answer_num: 最大的回答数量
        :return:
        """
        question_scores = defaultdict(list)
        # 取出top n 个query
        sort_scores = self.get_top_n_scores(sentence, n)
        # 按照question_id聚合，一个question下可能会有多个query，因此也会有多个分数
        for item in sort_scores:
            question_scores[self.query_to_question[item[0]]].append(item[1])
        # 取出每个question_id对应的最大分数
        question_max_scores = [(question, max(scores)) for question, scores in question_scores.items()]
        question_max_scores = sorted(question_max_scores, key=lambda x: x[1], reverse=True)
        # 如果只有一个question_id，则直接返回该question_id对应的question 和answer
        if len(question_max_scores) == 1:
            question_id = question_max_scores[0][0]
            question_answer_pair = dict(question=self.question_id_to_question[question_id],
                                        answer=self.question_id_to_answer[question_id])
            return question_answer_pair

        # 判断其他question和最大分数的question之间的间隔，如果小于该间隔，则加入到返回的answer中
        max_scores = question_max_scores[0][1]
        question_ids = [question_max_scores[0][0]]
        for item in question_max_scores[1:]:
            if max_scores - item[1] > interval:
                break
            question_ids.append(item[0])
            if len(question_ids) >= answer_num:
                break
        question_answer_pair = [dict(question=self.question_id_to_question[question_id],
                                     answer=self.question_id_to_answer[question_id])
                                for question_id in question_ids]
        return question_answer_pair

    def max_mean_score_answer(self, sentence, n):
        """
        根据question_id对应的分数列表取平均值来决定选择哪个question对应的answer
        :param sentence:
        :param n: 候选答案的数量
        :return:
        """
        question_scores = defaultdict(list)
        # 取出top n 个query
        sort_scores = self.get_top_n_scores(sentence, n)
        # 按照question_id聚合，一个question下可能会有多个query，因此也会有多个分数
        for item in sort_scores:
            question_scores[self.query_to_question[item[0]]].append(item[1])
        # 取出每个question_id对应的平均分数
        question_mean_scores = [(question, sum(scores) / len(scores)) for question, scores in question_scores.items()]
        question_mean_scores = sorted(question_mean_scores, key=lambda x: x[1], reverse=True)
        question_id = question_mean_scores[0][0]
        question_answer_pair = dict(question=self.question_id_to_question[question_id],
                                    answer=self.question_id_to_answer[question_id])
        return question_answer_pair

    def vote_answer(self, sentence, n=1):
        """

        :param sentence:
        :param n: 候选的数量
        :return:
        """
        sort_scores = self.get_top_n_scores(sentence, n)
        question_ids = [self.query_to_question[item[0]] for item in sort_scores]
        question_count = Counter(question_ids)
        question_sort = sorted(question_count.items(), key=lambda x: x[1], reverse=True)
        question_id = question_sort[0][0]
        question_answer_pair = dict(question=self.question_id_to_question[question_id],
                                    answer=self.question_id_to_answer[question_id])
        return question_answer_pair