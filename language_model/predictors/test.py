import json
from predict import Predictor


with open("E:/jiangxinyang/pa_smart_city_nlp/language_model/config/char_rnn_config.json", "r") as fr:
    config = json.load(fr)

predictor = Predictor(config)

result = predictor.predict("春天", 300).split("\n")
res = []
for item in result:
    if len(item) == 12:
        res.append(item)
    if len(res) == 4:
        break

print("\n".join(res))
