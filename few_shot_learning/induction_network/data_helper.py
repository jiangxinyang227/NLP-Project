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


class PrototypicalData(object):
    def __init__(self, output_path: str, sequence_length: int = 200, num_classes: int = 3, num_support: int = 5,
                 num_queries: int = 50, num_tasks: int = 10000, stop_word_path: Optional[str] = None,
                 embedding_size: Optional[int] = None, low_freq: int = 0,
                 word_vector_path: Optional[str] = None, is_training: bool = True):
        """
        init method
        :param output_path: path of train/eval data
        :param num_classes: number of support class
        :param num_support: number of support sample per class
        :param num_queries: number of query sample per class
        :param num_tasks: number of pre-sampling tasks, this will speeding up train
        :param stop_word_path: path of stop word file
        :param embedding_size: embedding size
        :param low_freq: frequency of words
        :param word_vector_path: path of word vector file(eg. word2vec, glove)
        :param is_training: bool
        """

        self.__output_path = output_path
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self.__sequence_length = sequence_length
        self.__num_classes = num_classes
        self.__num_support = num_support
        self.__num_queries = num_queries
        self.__num_tasks = num_tasks
        self.__stop_word_path = stop_word_path
        self.__embedding_size = embedding_size
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
        :return: dict. {class_name: [sample_1, ...., sample_n]}
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
        :param data: {class_name: [sample_1, ...., sample_n]}
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
        :return: {class_name: [sample_1, ..., sample_n]}
        """
        data_ids = {category: [[word_to_index.get(token, word_to_index["<UNK>"]) for token in line] for line in value]
                    for category, value in data.items()}
        return data_ids

    def choice_support_query(self, category_data: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        selecting support data and query data form all data for a category
        :param category_data: all data for a category
        :return:
        """
        support_data = random.sample(category_data, self.__num_support)
        other_data = copy.copy(category_data)
        [other_data.remove(data) for data in support_data]
        if self.__is_training:
            query_data = random.sample(other_data, self.__num_queries)
        else:
            query_data = other_data

        # padding
        support_data = self.padding(support_data)
        query_data = self.padding(query_data)
        return support_data, query_data

    def train_samples(self, data_ids: Dict[str, List[List[int]]]) \
            -> List[Dict[str, Union[List[List[List[int]]], List[List[int]], List[int]]]]:
        """
        positive and negative sample from raw data
        :param data_ids:
        :return:
        """
        category_list = list(data_ids.keys())

        new_data = []
        for i in range(self.__num_tasks):
            support_category = random.sample(category_list, self.__num_classes)
            support_set = []  # [num_classes, num_support, sequence_length]
            query_set = []  # [num_classes * num_queries, sequence_length]
            labels = []
            for idx, category in enumerate(support_category):
                category_data = data_ids[category]
                support_data, query_data = self.choice_support_query(category_data)
                label = [idx] * len(query_data)
                support_set.append(support_data)
                query_set.extend(query_data)
                labels.extend(label)
            new_data.append(dict(support=support_set, queries=query_set, labels=labels))
        return new_data

    def eval_sample(self, data_ids: Dict[str, List[List[int]]]) \
            -> List[Dict[str, Union[List[List[List[int]]], List[List[int]], List[int]]]]:
        """
        sample for eval stage
        :return:
        """
        category_list = list(data_ids.keys())
        support_category = random.sample(category_list, self.__num_classes)
        support_set = []  # [num_classes, num_support, sequence_length]
        query_set = []  # [num_classes * num_queries, sequence_length]
        labels = []
        for idx, category in enumerate(support_category):
            category_data = data_ids[category]
            support_data, query_data = self.choice_support_query(category_data)
            label = [idx] * len(query_data)
            support_set.append(support_data)
            query_set.extend(query_data)
            labels.extend(label)

        # split eval data into batches, avoiding memory overflow in eval stage
        tasks = []
        batch_size = self.__num_classes * self.__num_queries
        num_batches = len(query_set) // (self.__num_classes * self.__num_queries)
        for i in range(num_batches):
            query_batch = query_set[i * batch_size: (i + 1) * batch_size]
            label_batch = labels[i * batch_size: (i + 1) * batch_size]
            tasks.append(dict(support=support_set, queries=query_batch, labels=label_batch))

        return tasks

    def gen_data(self, file_path: str) -> List[Dict[str, Union[List[List[List[int]]], List[List[int]], List[int]]]]:
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

    def padding(self, sentences: List[List[int]]) -> List[List[int]]:
        """
        padding according to the predefined sequence length
        :param sentences:
        :return:
        """
        sentence_pad = [sentence[:self.__sequence_length] if len(sentence) > self.__sequence_length
                        else sentence + [0] * (self.__sequence_length - len(sentence))
                        for sentence in sentences]
        return sentence_pad

    @staticmethod
    def next_batch(tasks: List[Dict[str, Union[List[List[List[int]]], List[List[int]], List[int]]]]) \
            -> Dict[str, Union[List[List[List[int]]], List[List[int]], List[int]]]:
        """
        train a task at every turn
        :param tasks:
        :return:
        """
        random.shuffle(tasks)
        for task in tasks:
            yield task
