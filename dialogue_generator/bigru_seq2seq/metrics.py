"""
定义各类性能指标
"""
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def bleu(reference, candidate, weight=(0.25, 0.25, 0.25, 0.25)):
    """
    利用nltk工具包计算bleu值
    :param reference: 二维数组
    :param candidate: 一维数组
    :param weight: 从1-4gram的权重
    :return:
    """
    # 定义平滑函数对象
    smooth = SmoothingFunction()
    score = sentence_bleu(reference, candidate, weights=weight, smoothing_function=smooth.method1)
    return score


def get_bleu(true_y, pred_y, weight=(0.25, 0.25, 0.25, 0.25)):
    """
    计算batch数据的bleu值
    :param true_y: 真实值
    :param pred_y: 预测值
    :param weight:
    :return:
    """
    bleus = []
    for i in range(len(true_y)):
        score = bleu([true_y[i]], pred_y[i], weight)
        bleus.append(score)
    return mean(bleus)


