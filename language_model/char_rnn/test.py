import json

from predict import Predictor


with open("config.json", "r", encoding="utf8") as fr:
    config = json.load(fr)
predictor = Predictor(config=config)

start = "川田"
res = predictor.predict(start, 100)
print(res)


