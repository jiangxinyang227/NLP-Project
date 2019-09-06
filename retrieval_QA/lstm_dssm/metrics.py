"""
定义各类性能指标
"""


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_ys):
    """

    :param pred_ys:
    :return:
    """
    correct = 0
    for pred_y in pred_ys:
        if pred_y == 0:
            correct += 1

    return round(correct / len(pred_ys), 4)