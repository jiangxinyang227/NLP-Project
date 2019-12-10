import json
import os

from gensim import corpora, models
from predict import Predictor

with open("config.json", "r") as fr:
    config = json.load(fr)

dictionary = corpora.Dictionary.load_from_text(os.path.join(config["output_path"], "dict.txt"))
tfidf_model = models.TfidfModel.load(os.path.join(config["output_path"], "tfidf.model"))
init_size = len(dictionary.token2id)

candidates = ["什么情况", "你好"]
predictor = Predictor(config, init_size, len(candidates))
text = "你们怎么啦？"
res = predictor.predict(candidates, text, tfidf_model, dictionary)
print(res)