import json
import os
from predict import Predictor


with open(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "config/textcnn_config.json"), "r") as fr:
    config = json.load(fr)

predictor = Predictor(config)

text = "please see the content of this report"
result = predictor.predict(text.split(" "))
print(result)

