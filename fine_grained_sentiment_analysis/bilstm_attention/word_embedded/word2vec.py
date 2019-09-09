import logging
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = LineSentence("corpus.txt")

model = word2vec.Word2Vec(sentences, size=200, workers=6, min_count=5, sg=1, iter=5)
# model.save(save_model_file)
model.wv.save_word2vec_format("size200_min_count5_skip_gram_iter5")


