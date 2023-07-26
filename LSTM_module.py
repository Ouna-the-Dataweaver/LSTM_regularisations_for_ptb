import math
import pathlib
from typing import List
import random
import pickle
from IPython.display import clear_output
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from matplotlib import pyplot as plt
import numpy as np


def string_to_ids(sentence: str, words_dict: dict, mode="list"):
    ans = None
    if mode == "list":
        ans = [words_dict["<bos>"]]
        ans.extend([words_dict[word] for word in sentence.strip().split()])
        ans.append(words_dict["<eos>"])
    elif mode == "np":
        sent_list = sentence.strip().split()
        ans = np.zeros(len(sent_list) + 2, dtype=int)
        ans[0] = words_dict["<bos>"]
        for i in range(len(sent_list)):
            ans[i + 1] = words_dict[sent_list[i]]
        ans[-1] = words_dict["<eos>"]
    return ans


class DropConnectLSTM(torch.nn.LSTM):
    """
    Customized LSTM with built in dropconnect. Based on PyTorch LSTM
    """

    def __init__(self, *args, weight_dropout=0.0, drop_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_dropout = weight_dropout
        self.drop_bias = drop_bias
        self.weight_hh_l0_raw = nn.Parameter(self.weight_hh_l0)
        if self.drop_bias:
            self.bias_hh_l0_raw = nn.Parameter(self.bias_hh_l0)

    def forward(self, input, hx=None):
        weight_hh_l0 = torch.nn.functional.dropout(self.weight_hh_l0_raw, p=self.weight_dropout,
                                                   training=self.training)
        self.weight_hh_l0 = nn.Parameter(weight_hh_l0)

        if self.drop_bias:
            bias_hh_l0 = torch.nn.functional.dropout(self.bias_hh_l0_raw, p=self.weight_dropout, training=self.training)
            self.bias_hh_l0 = nn.Parameter(bias_hh_l0)
        return super().forward(input, hx)


class WordPredictionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_arr, weight_tying, drop_connect=0.0,
                 padding_idx=None):
        super(WordPredictionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

        self.dropout_embedding_lstm_0 = MyDropout(dropout_arr[0])

        if drop_connect == 0:
            self.lstm_0 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        else:
            self.lstm_0 = DropConnectLSTM(embedding_dim, hidden_dim, batch_first=True, weight_dropout=drop_connect)
        self.dropout_lstm_0_lstm_1 = MyDropout(dropout_arr[1])

        if drop_connect == 0:
            self.lstm_1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.lstm_1 = DropConnectLSTM(hidden_dim, hidden_dim, batch_first=True, weight_dropout=drop_connect)
        self.dropout_lstm_1_lstm_2 = MyDropout(dropout_arr[2])

        if drop_connect == 0:
            self.lstm_2 = nn.LSTM(hidden_dim, embedding_dim, batch_first=True)
        else:
            self.lstm_2 = DropConnectLSTM(hidden_dim, embedding_dim, batch_first=True, weight_dropout=drop_connect)
        self.dropout_lstm_2_fc = MyDropout(dropout_arr[3])

        self.fc = nn.Linear(embedding_dim, vocab_size)
        if weight_tying:
            self.fc.weight = self.embedding.weight

    def forward(self, x, *state_arr):

        if len(state_arr) > 0:
            state_0 = [state_arr[0]]
            state_1 = [state_arr[1]]
            state_2 = [state_arr[2]]
        else:
            state_0 = []
            state_1 = []
            state_2 = []

        embedded = self.embedding(x)

        embedding_lstm_dropout = self.dropout_embedding_lstm_0(embedded)

        lstm_0_out, state_0_out = self.lstm_0(embedding_lstm_dropout, *state_0)

        lstm_0_out_dropout = self.dropout_lstm_0_lstm_1(lstm_0_out)

        lstm_1_out, state_1_out = self.lstm_1(lstm_0_out_dropout, *state_1)

        lstm_1_out_dropout = self.dropout_lstm_1_lstm_2(lstm_1_out)

        lstm_2_out, state_2_out = self.lstm_2(lstm_1_out_dropout, *state_2)

        lstm_2_out_dropout = self.dropout_lstm_2_fc(lstm_2_out)

        logits = self.fc(lstm_2_out_dropout)

        return logits, [state_0_out, state_1_out, state_2_out], [lstm_2_out, lstm_2_out_dropout]


class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise Exception(f"dropout {p} not in [0, 1] range")
        self.p = p

    def forward(self, X):
        if self.training:
            X_part_copy = torch.ones_like(X[:, 0:1, :])
            mask_per_one_in_seq = X_part_copy.bernoulli_(p=(1 - self.p))
            mask = mask_per_one_in_seq.repeat((1, X.shape[1], 1))
            return X * mask * (1.0 / (1 - self.p))
        return X


class Trainer:
    def __init__(self, device):
        # technical parameters
        self.train_array = None
        self.data_train = None
        self.data_valid = None
        self.data_valid_size = None
        self.data_test = None
        self.data_test_size = None
        self.words_dict = None
        self.model = None
        self.vocab_size = None
        self.eos_token = None
        self.bos_token = None
        self.device = device

        # training hyperparameters
        self.embedding_dim = 400
        self.hidden_dim = 1150

        self.weight_tying = True

        self.dropout_words = 0.2
        self.dropout_emb_lstm_0 = 0.15
        self.dropout_lstm_0_lstm_1 = 0.29
        self.dropout_lstm_1_lstm_2 = 0.31
        self.dropout_lstm_2_fc = 0.26

        self.drop_connect = 0.5

        self.batch_size = 60
        self.base_seq_len = 70
        self.learning_rate_sgd = 20
        self.learning_rate_adam = 8e-3
        # self.lr = None
        self.max_norm = 1.5
        self.decay = 1e-6

        self.alpha = 2
        self.beta = 1

        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # state is for LSTM state
        self.state_arr = None

        # training state
        self.epochs = 0
        self.training_history_dict = dict()

        self.training_history_dict["epochs"] = []

        self.training_history_dict["lr"] = []
        self.training_history_dict["max_norm"] = []
        self.training_history_dict["decay"] = []

        self.training_history_dict["ce_loss_train"] = []
        self.training_history_dict["L2_loss_train"] = []
        self.training_history_dict["full_loss_train"] = []

        self.training_history_dict["alpha"] = []
        self.training_history_dict["beta"] = []

        self.training_history_dict["ppl_train"] = []
        self.training_history_dict["ppl_valid"] = []

        self.training_history_dict["dp_input"] = []
        # self.training_history_dict["dp_emb"] = []
        self.training_history_dict["dp_emb_lstm"] = []
        self.training_history_dict["dp_lstm_0_1"] = []
        self.training_history_dict["dp_lstm_1_2"] = []
        self.training_history_dict["dp_lstm_2_fc"] = []
        self.training_history_dict["drop_connect"] = []


    def init_train(self):
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate_sgd, momentum=0.7)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_adam, weight_decay=self.decay)

        self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=1, total_iters=1)
        self.criterion = nn.CrossEntropyLoss()

    def init_model(self):
        if self.model is not None:
            del self.model
        params = dict()
        params["vocab_size"] = self.vocab_size
        params["embedding_dim"] = self.embedding_dim
        params["hidden_dim"] = self.hidden_dim

        params["dropout_arr"] = []
        params["dropout_arr"].append(self.dropout_emb_lstm_0)
        params["dropout_arr"].append(self.dropout_lstm_0_lstm_1)
        params["dropout_arr"].append(self.dropout_lstm_1_lstm_2)
        params["dropout_arr"].append(self.dropout_lstm_2_fc)

        params["weight_tying"] = self.weight_tying
        params["drop_connect"] = self.drop_connect

        model = WordPredictionLSTM(**params)
        model = model.to(self.device)

        self.model = model
        return None

    def set_param(self, param: str, val):
        if param == "dropout_words":
            self.dropout_words = val
        if param == "dropout_emb_lstm_0":
            self.model.dropout_embedding_lstm_0.p = val
            self.dropout_emb_lstm_0 = val
        if param == "dropout_lstm_0_lstm_1":
            self.model.dropout_lstm_0_lstm_1.p = val
            self.dropout_lstm_0_lstm_1 = val
        if param == "dropout_lstm_1_lstm_2":
            self.model.dropout_lstm_1_lstm_2.p = val
            self.dropout_lstm_1_lstm_2 = val
        if param == "dropout_lstm_2_fc":
            self.model.dropout_lstm_2_fc.p = val
            self.dropout_lstm_2_fc = val
        if param == "lr":
            self.optimizer.param_groups[0]['lr'] = val
            self.lr = val
        if param == "alpha":
            self.alpha = val
        if param == "beta":
            self.beta = val
        if param == "max_norm":
            self.max_norm = val
        if param == "base_seq_len":
            self.base_seq_len = val
        if param == "decay":
            self.optimizer.param_groups[0]['weight_decay'] = val
            self.decay = val

    def load_dictionary(self, words_dict_path: pathlib.Path):
        words_dict = dict()
        with open(words_dict_path) as words_file:
            for line in words_file:
                word, word_id = line.strip().split()
                word_id = int(word_id)
                words_dict[word] = word_id
        max(words_dict.values())
        words_dict["<bos>"] = 1
        self.bos_token = words_dict["<bos>"]
        last_idx_words_dict = max(words_dict.values())
        words_dict["<eos>"] = last_idx_words_dict + 1
        self.eos_token = words_dict["<eos>"]
        self.words_dict = words_dict
        self.vocab_size = max(words_dict.values()) + 1
        return None

    def load_data(self, path: pathlib.Path, data_type: str):
        """
        opens specified path and loads data from it
        converts valid/test to array of tensors
        train to just array of lists of ids
        """
        with open(path, 'r') as data_file:
            array = [string_to_ids(sentence_string, words_dict=self.words_dict) for sentence_string in data_file]
        if data_type == "train":
            self.train_array = array
        elif data_type == "valid" or data_type == "test":
            dict_for_data = dict()
            for sentence_list in array:
                if len(sentence_list) not in dict_for_data:
                    dict_for_data[len(sentence_list)] = []
                dict_for_data[len(sentence_list)].append(sentence_list)
            tensor_array, data_size = self.valid_test_to_tensor(dict_for_data)
            if data_type == "valid":
                self.data_valid = tensor_array
                self.data_valid_size = data_size
            elif data_type == "test":
                self.data_test = tensor_array
                self.data_test_size = data_size
        else:
            raise Exception("please specify what is being loaded: 'train', 'valid' or 'test'")
        return None

    def valid_test_to_tensor(self, sentence_dict: dict):
        arr_of_tensors = []
        data_size = 0
        for _, sent_arr in sentence_dict.items():
            new_tensor = torch.tensor(sent_arr, dtype=torch.int64, device=self.device)
            new_size = new_tensor.shape[0] * (new_tensor.shape[1] - 1)
            arr_of_tensors.append(torch.tensor(sent_arr, dtype=torch.int64, device=self.device))
            data_size += new_size
        return arr_of_tensors, data_size

    def generate_train_tensor(self, shuffle=True, batch_size=None):
        if shuffle:
            random.shuffle(self.train_array)
        if batch_size is None:
            batch_size = self.batch_size
        flattened_data = [item for sublist in self.train_array for item in sublist]
        total_elems = len(flattened_data)
        tensor_len = total_elems // batch_size
        flattened_truncated_data = flattened_data[:batch_size * tensor_len]
        tensor_flat = torch.tensor(flattened_truncated_data, dtype=torch.int64, device=self.device)
        tensor_train = tensor_flat.view(batch_size, tensor_len)
        # dropout, we should not ruin bos and eos tokens!

        bos_mask_bool = torch.eq(tensor_train, self.bos_token)
        eos_mask_bool = torch.eq(tensor_train, self.eos_token)
        mask_important_tokens = torch.logical_or(bos_mask_bool, eos_mask_bool)

        mask = torch.ones_like(tensor_train)
        mask = mask.bernoulli_(p=(1 - self.dropout_words))
        mask = torch.where(mask_important_tokens, torch.ones_like(tensor_train), mask)

        tensor_target = tensor_train
        tensor_train = mask * tensor_train

        return tensor_train, tensor_target

    def train_step(self, data, target, first_in_batch=True):
        self.optimizer.zero_grad()
        if first_in_batch:
            pred, state_arr, last_layer_hidden = self.model(data)
        else:
            pred, state_arr, last_layer_hidden = self.model(data, *self.state_arr)
        lstm_out, lstm_out_dropped = last_layer_hidden[0], last_layer_hidden[1]
        pred_flattened = pred.flatten(0, 1)
        target_flattened = target.flatten()

        loss_cross_entropy = self.criterion(pred_flattened, target_flattened)

        if self.alpha > 0:
            loss_ar = self.alpha * lstm_out_dropped.pow(2).mean()
        else:
            loss_ar = 0

        if self.beta > 0:
            loss_tar = self.beta * (lstm_out[:, :-1, :] - lstm_out[:, 1:, :]).pow(2).mean()
        else:
            loss_tar = 0

        total_loss = loss_cross_entropy + loss_ar + loss_tar

        total_loss.backward()

        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

        self.optimizer.step()
        for tuple_state in state_arr:
            for tensor in tuple_state:
                tensor.detach_()
        self.state_arr = state_arr
        return loss_cross_entropy.item(), total_loss.item()

    def loss_on_data(self, dataset="valid"):
        if dataset == "valid":
            tensor_array = self.data_valid
            data_size = self.data_valid_size
        elif dataset == "train":
            tensor_array = self.data_test
            data_size = self.data_test_size
        else:
            raise Exception("you can calculate loss only on validation set due to reasons")
        total_loss = 0
        with torch.no_grad():
            for tensor in tensor_array:
                data = tensor[:, 0:-1]
                target = tensor[:, 1:]
                pred, _, _ = self.model(data)

                pred_flattened = pred.flatten(0, 1)
                target_flattened = target.flatten()

                loss = self.criterion(pred_flattened, target_flattened)
                loss_val = loss.item()
                total_loss += loss_val * (tensor.shape[0] * (tensor.shape[1] - 1))
        total_loss = total_loss / data_size
        return total_loss

    def sequence_len_generator(self, cut_in_half=5):
        ans = random.randint(self.base_seq_len - 10, self.base_seq_len + 10)
        if random.randint(0, 99) < cut_in_half:
            ans = ans // 2
        return ans

    def draw_graph(self, print_graphs):
        x = self.training_history_dict["epochs"]
        ppl_train = self.training_history_dict["ppl_train"]
        ppl_valid = self.training_history_dict["ppl_valid"]

        ce_loss_train = self.training_history_dict["ce_loss_train"]
        L2_loss_train = self.training_history_dict["L2_loss_train"]
        full_loss_train = self.training_history_dict["full_loss_train"]

        alpha = self.training_history_dict["alpha"]
        beta = self.training_history_dict["beta"]
        max_norm = self.training_history_dict["max_norm"]
        decay = self.training_history_dict["decay"]
        lr = self.training_history_dict["lr"]

        dp_input = self.training_history_dict["dp_input"]
        dp_emb_lstm = self.training_history_dict["dp_emb_lstm"]
        dp_lstm_0_1 = self.training_history_dict["dp_lstm_0_1"]
        dp_lstm_1_2 = self.training_history_dict["dp_lstm_1_2"]
        dp_lstm_2_fc = self.training_history_dict["dp_lstm_2_fc"]
        drop_connect = self.training_history_dict["drop_connect"]

        if print_graphs:
            plt.figure()
            plt.rcParams.update({'font.size': 6})

            latest_ppl = plt.subplot2grid(shape=(3, 3), loc=(0, 0), colspan=2)
            dp = plt.subplot2grid(shape=(3, 3), loc=(0, 2))

            all_ppl = plt.subplot2grid(shape=(3, 3), loc=(1, 0), colspan=2)
            ab_norm = plt.subplot2grid(shape=(3, 3), loc=(1, 2))

            loss = plt.subplot2grid(shape=(3, 3), loc=(2, 0))
            lr_dec = plt.subplot2grid(shape=(3, 3), loc=(2, 1))
            decay_etc = plt.subplot2grid(shape=(3, 3), loc=(2, 2))

            latest_ppl.plot(x[-10:], ppl_train[-10:], label='train')
            latest_ppl.plot(x[-10:], ppl_valid[-10:], label='valid')
            latest_ppl.legend()

            dp.plot(x, dp_input, label="dp_input")
            dp.plot(x, dp_emb_lstm, label="dp_emb_lstm")
            dp.plot(x, dp_lstm_0_1, label="dp_lstm_0_1")
            dp.plot(x, dp_lstm_1_2, label="dp_lstm_1_2")
            dp.plot(x, dp_lstm_2_fc, label="dp_lstm_2_fc")
            dp.plot(x, drop_connect, label="drop_connect")
            dp.legend(fontsize=4)

            all_ppl.plot(x, ppl_train, label='train')
            all_ppl.plot(x, ppl_valid, label='valid')
            all_ppl.set_yscale('log')
            all_ppl.legend()

            ab_norm.plot(x, alpha, label='alpha')
            ab_norm.plot(x, beta, label='beta')
            ab_norm.legend()

            loss.plot(x, ce_loss_train, label='ce_loss_train')
            loss.plot(x, L2_loss_train, label='L2_loss_train')
            loss.plot(x, full_loss_train, label='full_loss_train')
            loss.set_yscale('log')
            loss.legend(fontsize=4)

            lr_dec.plot(x, lr, label='lr')
            # lr_dec.plot(x, max_norm, label='max_norm')
            # lr_dec.set_yscale('log')
            lr_dec.legend()

            decay_etc.plot(x, decay, label='decay')
            decay_etc.legend()

            plt.show()

        return min(ppl_valid), ppl_train[-1], ppl_valid[-1]

    def train(self, epoch_num: int = 1, print_train=True, print_graphs=True):
        start_time = time.time()
        for epoch in range(epoch_num):
            cur_epoch_global = self.epochs + epoch
            # self.training_history[cur_epoch_global] = dict()

            tensor_train, tensor_target = self.generate_train_tensor(shuffle=True)
            tensor_len = tensor_train.shape[1]
            batch_idx = 0
            tensor_pos = 0
            batch_start = True
            train_loss_ce_arr = []
            train_loss_total_arr = []
            print("Epoch:", epoch+1, end="_")
            self.model.train()
            progress = [-100, 0]
            while True:
                # printing status of train epoch
                progress[1] = int(tensor_pos / tensor_len * 100)
                if progress[1] - progress[0] >= 5:
                    print(progress[1], end='|')
                    progress[0] = progress[1]
                # sample sequence len
                seq_len = self.sequence_len_generator()
                if seq_len + tensor_pos > tensor_len:
                    break

                train = tensor_train[:, tensor_pos:tensor_pos + seq_len - 1]

                target = tensor_target[:, tensor_pos + 1:tensor_pos + seq_len]

                # adjustment for lr based on seq len
                lr_adjustment = seq_len / self.base_seq_len
                for g in self.optimizer.param_groups:
                    g['lr'] *= lr_adjustment

                loss_cross_entropy, loss_total = self.train_step(train, target, first_in_batch=batch_start)
                train_loss_ce_arr.append(loss_cross_entropy)
                train_loss_total_arr.append(loss_total)

                batch_start = False

                # return to old LR so schedule works correctly!
                for g in self.optimizer.param_groups:
                    g['lr'] /= lr_adjustment

                # -1 so we actually don't miss training on last element!
                tensor_pos += (seq_len - 1)
                # print("batch: ", batch_idx)
                batch_idx += 1

            self.training_history_dict["epochs"].append(cur_epoch_global)

            curr_lr = self.optimizer.param_groups[0]['lr']
            self.training_history_dict["lr"].append(curr_lr)
            self.training_history_dict["max_norm"].append(self.max_norm)
            self.training_history_dict["decay"].append(self.decay)

            ce_loss_train = sum(train_loss_ce_arr) / len(train_loss_ce_arr)

            full_loss_train = sum(train_loss_total_arr) / len(train_loss_total_arr)
            L2_loss_train = full_loss_train - ce_loss_train

            self.training_history_dict["ce_loss_train"].append(ce_loss_train)
            self.training_history_dict["L2_loss_train"].append(L2_loss_train)
            self.training_history_dict["full_loss_train"].append(full_loss_train)

            train_ppl = math.exp(ce_loss_train)
            self.training_history_dict["ppl_train"].append(train_ppl)

            self.training_history_dict["alpha"].append(self.alpha)
            self.training_history_dict["beta"].append(self.beta)

            # val
            self.model.eval()
            validation_loss = self.loss_on_data()
            validation_ppl = math.exp(validation_loss)

            self.training_history_dict["ppl_valid"].append(validation_ppl)

            self.training_history_dict["dp_input"].append(self.dropout_words)
            # self.training_history_dict["dp_emb"].append()
            self.training_history_dict["dp_emb_lstm"].append(self.dropout_emb_lstm_0)
            self.training_history_dict["dp_lstm_0_1"].append(self.dropout_lstm_0_lstm_1)
            self.training_history_dict["dp_lstm_1_2"].append(self.dropout_lstm_1_lstm_2)
            self.training_history_dict["dp_lstm_2_fc"].append(self.dropout_lstm_2_fc)
            self.training_history_dict["drop_connect"].append(self.drop_connect)

            self.scheduler.step()

            # graph
            if print_graphs:
                clear_output(wait=True)
            min_ppx_val, last_ppx_train, last_ppx_val = self.draw_graph(print_graphs)
            print(
                f'min val ppx:  {min_ppx_val:.2f}; last val ppx: {last_ppx_val:.2f} last train ppx: {last_ppx_train:.2f}')
            end_time = time.time()
            total_time = end_time - start_time
            total_time = int(total_time)
            print(f"total elapsed time:  {total_time // 60} mins {total_time % 60} seconds")

            self.scheduler.step()

        self.epochs += epoch_num

    def load_valid_test(self, paths: List[pathlib.Path]):
        pass
