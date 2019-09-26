"""
data process
"""

import os
import json
import random
import copy
from collections import Counter
from itertools import chain
from typing import Dict, Tuple, Optional, List, Union

import gensim
import numpy as np


class SiameseData(object):
    def __init__(self, output_path: str, sequence_length: int = 200, neg_samples: int = 1,
                 stop_word_path: Optional[str] = None,
                 embedding_size: Optional[int] = None, low_freq: int = 0, num_sample_of_category: int = 10000,
                 word_vector_path: Optional[str] = None, is_training: bool = True):
        """
        init method
        :param output_path: path of train/eval data
        :param neg_samples: num of neg sample, default 1
        :param stop_word_path: path of stop word file
        :param embedding_size: embedding size
        :param low_freq: frequency of words
        :param num_sample_of_category: The number of samples in each category
        :param word_vector_path: path of word vector file(eg. word2vec, glove)
        :param is_training: bool
        """

        self.__output_path = output_path
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self.__sequence_length = sequence_length
        self.__neg_samples = neg_samples
        self.__stop_word_path = stop_word_path
        self.__embedding_size = embedding_size
        self.__num_sample_of_category = num_sample_of_category
        self.__low_freq = low_freq
        self.__word_vector_path = word_vector_path
        self.__is_training = is_training

        self.vocab_size = None
        self.word_vectors = None
        self.current_category_index = 0  # record current sample category

    @staticmethod
    def load_data(file_path: str) -> Dict[str, List[List[str]]]:
        """
        read train/eval data
        :param file_path:
        :return: dict.
        """
        category_dirs = os.listdir(file_path)
        categories_data = {}
        for category_dir in category_dirs:
            category_file_path = os.path.join(file_path, category_dir)
            category_files = os.listdir(category_file_path)
            category_data = []
            for category_file in category_files:
                with open(os.path.join(category_file_path, category_file), "r", encoding="utf8") as fr:
                #     import jieba
                #     lines = [line.strip() for line in fr.readlines()]
                #     lines = [" ".join(jieba.lcut(line)) for line in lines]
                #     content = " ".join(lines)
                #     print(content)
                # with open(os.path.join(category_file_path, category_file), "w", encoding="utf8") as fr:
                #     fr.write(content)
                    content = fr.read().strip().split(" ")
                    category_data.append(content)
            categories_data[category_dir] = category_data

        return categories_data

    def remove_stop_word(self, data: Dict[str, List[List[str]]]) -> List[str]:
        """
        remove low frequency words and stop words, construct vocab
        :param data:
        :return:
        """
        all_words = list(chain(*chain(*list(data.values()))))
        word_count = Counter(all_words)  # statistic the frequency of words
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # remove low frequency word
        words = [item[0] for item in sort_word_count if item[1] > self.__low_freq]

        # if stop word file exists, then remove stop words
        if self.__stop_word_path:
            with open(self.__stop_word_path, "r", encoding="utf8") as fr:
                stop_words = [line.strip() for line in fr.readlines()]
            words = [word for word in words if word not in stop_words]

        return words

    def get_word_vectors(self, vocab: List[str]) -> np.ndarray:
        """
        load word vector file,
        :param vocab: vocab
        :return:
        """
        pad_vector = np.zeros(self.__embedding_size)  # set the "<PAD>" vector to 0
        word_vectors = (1 / np.sqrt(len(vocab) - 1) * (2 * np.random.rand(len(vocab) - 1, self.__embedding_size) - 1))
        if os.path.splitext(self.__word_vector_path)[-1] == ".bin":
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self.__word_vector_path, binary=True)
        else:
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self.__word_vector_path)

        for i in range(1, len(vocab)):
            try:
                vector = word_vec.wv[vocab[i]]
                word_vectors[i, :] = vector
            except:
                print(vocab[i] + "not exist word vector file")
        word_vectors = np.vstack((pad_vector, word_vectors))
        return word_vectors

    def gen_vocab(self, words: List[str]) -> Dict[str, int]:
        """
        generate word_to_index mapping table
        :param words:
        :return:
        """
        if self.__is_training:
            vocab = ["<PAD>", "<UNK>"] + words

            self.vocab_size = len(vocab)

            if self.__word_vector_path:
                word_vectors = self.get_word_vectors(vocab)
                self.word_vectors = word_vectors
                # save word vector to npy file
                np.save(os.path.join(self.__output_path, "word_vectors.npy"), self.word_vectors)

            word_to_index = dict(zip(vocab, list(range(len(vocab)))))

            # save word_to_index to json file
            with open(os.path.join(self.__output_path, "word_to_index.json"), "w") as f:
                json.dump(word_to_index, f)
        else:
            with open(os.path.join(self.__output_path, "word_to_index.json"), "r") as f:
                word_to_index = json.load(f)

        return word_to_index

    @staticmethod
    def trans_to_index(data: Dict[str, List[List[str]]], word_to_index: Dict[str, int]) -> Dict[str, List[List[int]]]:
        """
        transformer token to id
        :param data:
        :param word_to_index:
        :return:
        """
        data_ids = {category: [[word_to_index.get(token, word_to_index["<UNK>"]) for token in line] for line in value]
                    for category, value in data.items()}
        return data_ids

    def train_samples(self, data_ids: Dict[str, List[List[int]]]) -> List[Tuple[List[int], List[int], int]]:
        """
        positive and negative sample from raw data
        :param data_ids:
        :return:
        """
        category_list = list(data_ids.keys())
        new_data = []
        for category, value in data_ids.items():
            category_data = data_ids[category]
            neg_category_list = copy.copy(category_list)
            neg_category_list.remove(category)
            neg_category_data = [data_ids[neg_category] for neg_category in neg_category_list]
            neg_category_data = list(chain(*neg_category_data))
            for i in range(self.__num_sample_of_category):
                pos_samples = random.sample(category_data, 3)
                neg_sample = random.choice(neg_category_data)
                new_data.append((pos_samples[0], pos_samples[1], 1))
                new_data.append((pos_samples[2], neg_sample, 0))
        return new_data

    @staticmethod
    def eval_sample(data_ids: Dict[str, List[List[int]]]) \
            -> Tuple[Dict[str, int], List[List[List[int]]], List[Tuple[List[int], int]]]:
        """
        sample for eval stage
        :return:
        """
        category_list = list(data_ids.keys())
        label_to_idx = dict(zip(category_list, range(len(category_list))))
        all_support_data = []  # support set
        all_query_data = []  # query set
        for category in category_list:
            category_data = data_ids[category]
            support_data = random.sample(category_data, 1)
            all_support_data.append(support_data)
            other_data = copy.copy(category_data)
            [other_data.remove(data) for data in support_data]
            query_data = random.sample(other_data, len(category_data) - 1)
            query_data = [(data, label_to_idx[category]) for data in query_data]
            all_query_data.extend(query_data)
        return label_to_idx, all_support_data, all_query_data

    def gen_data(self, file_path: str) -> Union[List[Tuple[List[int], List[int], int]],
                                                Tuple[Dict[str, int], List[List[List[int]]],
                                                      List[Tuple[List[int], int]]]]:
        """
        Generate data that is eventually input to the model
        :return:
        """
        data = self.load_data(file_path)
        words = self.remove_stop_word(data)
        word_to_index = self.gen_vocab(words)
        data_ids = self.trans_to_index(data, word_to_index)
        if self.__is_training:
            new_data = self.train_samples(data_ids)
        else:
            new_data = self.eval_sample(data_ids)
        return new_data

    def padding(self, first_sentences: List[List[int]], second_sentences: List[List[int]], labels: List[int]) \
            -> Dict[str, Union[List[List[int]], List[int]]]:
        """
        padding according to the max length
        :param first_sentences:
        :param second_sentences:
        :param labels:
        :return:
        """
        first_sentence_pad = [sentence[:self.__sequence_length] if len(sentence) > self.__sequence_length
                              else sentence + [0] * (self.__sequence_length - len(sentence))
                              for sentence in first_sentences]
        second_sentence_pad = [sentence[:self.__sequence_length] if len(sentence) > self.__sequence_length
                               else sentence + [0] * (self.__sequence_length - len(sentence))
                               for sentence in second_sentences]
        return dict(first=first_sentence_pad, second=second_sentence_pad, labels=labels)

    def next_batch(self, data: List[Tuple[List[int], List[int], int]], batch_size: int) \
            -> Dict[str, Union[List[List[int]], List[int]]]:
        """

        :param data:
        :param batch_size:
        :return:
        """
        random.shuffle(data)
        x, y, label = zip(*data)
        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = x[start: end]
            batch_y = y[start: end]
            batch_label = label[start: end]

            yield self.padding(batch_x, batch_y, batch_label)
