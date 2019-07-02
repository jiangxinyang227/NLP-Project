import json
import jieba
from predict import Predictor


with open("E:/jiangxinyang/pa_smart_city_nlp/text_generator/config/seq2seq_bigru_config.json", "r") as fr:
    config = json.load(fr)

predictor = Predictor(config)
text = jieba.lcut("你们想干嘛")
print(text)
result = predictor.predict(text)
print(result)
