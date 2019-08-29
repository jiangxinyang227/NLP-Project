import json
import jieba
from predict import Predictor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="config path of model")

with open(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), args.config_path), "r") as fr:
    config = json.load(fr)
 

with open("test_data.txt", "r", encoding="utf8") as f:
    data = [line for line in f.readlines()]
    inputs = []
    labels = []
    for line in data:
        try:
            x, y = line.strip().split("<SEP>")
            inputs.append(x.strip())
            labels.append(y.strip())
        except:
            print(line)

# text = " ".join([" ".join(jieba.lcut(line)) for line in data])

predictor = Predictor(config)

total = len(labels)
print(set(labels))
corr = 0
for i in range(len(inputs)):
    result = predictor.predict(inputs[i].split(" "))
    if result == labels[i]:
        corr += 1
    else:
        print(inputs[i])
print(corr / total)

