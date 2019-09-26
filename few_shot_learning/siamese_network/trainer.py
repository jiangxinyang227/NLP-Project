import json
import os
import argparse

import tensorflow as tf
import numpy as np
from data_helper import SiameseData
from model import SiameseModel
from metrics import get_binary_metrics, get_multi_metrics

RATE = 0.2


class SiameseTrainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)

        # self.builder = tf.saved_model.builder.SavedModelBuilder("../pb_model/weibo/bilstm/savedModel")
        # load date set
        self.train_data_obj = self.load_data()
        self.eval_data_obj = self.load_data(is_training=False)
        self.train_data_ids = self.train_data_obj.gen_data(self.config["train_data"])
        self.label_to_idx, self.support_data, self.query_data = self.eval_data_obj.gen_data(self.config["eval_data"])

        print("train data numbers: ", len(self.train_data_ids))

        self.model = self.create_model()

    def load_data(self, is_training=True):
        """
        init data object
        :return:
        """
        data_obj = SiameseData(self.config["output_path"], sequence_length=self.config["sequence_length"],
                               neg_samples=self.config["neg_samples"], is_training=is_training)
        return data_obj

    def create_model(self):
        """
        init model object
        :return:
        """
        model = SiameseModel(config=self.config, vocab_size=self.train_data_obj.vocab_size,
                             word_vectors=self.train_data_obj.word_vectors)
        return model

    def padding(self, first_sentences, second_sentences):
        """
        padding for eval data according to the max length
        :param first_sentences:
        :param second_sentences:
        :return:
        """
        sequence_length = self.config["sequence_length"]
        first_sentence_pad = [sentence[:sequence_length] if len(sentence) > sequence_length
                              else sentence + [0] * (sequence_length - len(sentence))
                              for sentence in first_sentences]
        second_sentence_pad = [sentence[:sequence_length] if len(sentence) > sequence_length
                               else sentence + [0] * (sequence_length - len(sentence))
                               for sentence in second_sentences]
        return dict(first=first_sentence_pad, second=second_sentence_pad)

    @staticmethod
    def get_prediction(sims):
        """
        Final prediction results are obtained by similarity value in the eval stage
        :param sims: similarity value
        :return:
        """
        np_sims = np.array(sims)
        np_sims_transpose = np.transpose(np_sims)
        predictions = np.argmax(np_sims_transpose, axis=-1).tolist()
        return predictions

    def eval_model(self, sess):
        """
        eval model
        :param sess: session object of tensorflow
        :return:
        """
        true_label = []
        pred_label = []
        # split query data into batches
        num_batch = len(self.query_data) // self.config["batch_size"]
        for i in range(num_batch):
            batch_query = self.query_data[i * self.config["batch_size"]: (i + 1) * self.config["batch_size"]]
            first = [query[0] for query in batch_query]
            label = [query[1] for query in batch_query]
            # save the true label of query sample
            true_label.extend(label)
            category_sim = []
            # calculate the similarity between query batch and support set
            for category_vec in self.support_data:
                second = category_vec * self.config["batch_size"]
                batch = self.padding(first, second)
                sim = self.model.infer(sess, batch)
                category_sim.append(sim)
            pred = self.get_prediction(category_sim)
            pred_label.extend(pred)
        return true_label, pred_label

    def train(self):
        """
        train model
        :return:
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # init all variable in graph
            sess.run(tf.global_variables_initializer())
            current_step = 0

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))
                for batch in self.train_data_obj.next_batch(self.train_data_ids, self.config["batch_size"]):
                    loss, predictions = self.model.train(sess, batch, self.config["keep_prob"])

                    acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batch["labels"])
                    print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                        current_step, loss, acc, recall, prec, f_beta))

                    current_step += 1
                    if current_step % self.config["checkpoint_every"] == 0:
                        true_label, pred_label = self.eval_model(sess)
                        label_list = list(self.label_to_idx.values())
                        eval_acc, eval_recall, eval_prec, eval_f1 = get_multi_metrics(pred_label, true_label,
                                                                                      label_list)
                        print("\n")
                        print("eval: acc: {}, recall: {}, precision: {}, f_beta: {}".format(eval_acc, eval_recall,
                                                                                            eval_prec, eval_f1))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = os.path.join(os.path.abspath(os.getcwd()),
                                                     self.config["ckpt_model_path"])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

            # inputs = {"inputs": tf.saved_model.utils.build_tensor_info(self.model.inputs),
            #           "keep_prob": tf.saved_model.utils.build_tensor_info(self.model.keep_prob)}
            #
            # outputs = {"predictions": tf.saved_model.utils.build_tensor_info(self.model.predictions)}
            #
            # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
            #                                                                               outputs=outputs,
            #                                                                               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            # legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
            # self.builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
            #                                           signature_def_map={"classifier": prediction_signature},
            #                                           legacy_init_op=legacy_init_op)
            #
            # self.builder.save()


if __name__ == "__main__":
    # Read the input information by the user on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    trainer = SiameseTrainer(args)
    trainer.train()
