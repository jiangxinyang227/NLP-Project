"""
定义各类性能指标
"""
from sklearn import metrics


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def get_metrics(y, y_pre):
    """
    计算整体的性能指标
    :param y:
    :param y_pre:
    :return:
    """
    hamming_loss = metrics.hamming_loss(y, y_pre)
    macro_f1 = metrics.f1_score(y, y_pre, average='macro')
    macro_precision = metrics.precision_score(y, y_pre, average='macro')
    macro_recall = metrics.recall_score(y, y_pre, average='macro')
    micro_f1 = metrics.f1_score(y, y_pre, average='micro')
    micro_precision = metrics.precision_score(y, y_pre, average='micro')
    micro_recall = metrics.recall_score(y, y_pre, average='micro')
    return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall


