import os
import json
import pdb
import sys
from collections import Counter

import numpy as np
import pandas as pd
import sklearn
import torch
from coral_pytorch.dataset import proba_to_label
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from sklearn import preprocessing
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder


# ----------------------------------------------------------------------------------------------------------------------
import utils
from utils import load_json, save_json
from pathlib import Path

def collate_fn_mk(batch):
    """Collects together univariate or multivariate sequences into a single batch, arranged in descending order of length.
    Also returns the corresponding lengths and labels as :class:`torch:torch.LongTensor` objects.
    Parameters
    ----------
    batch: list of tuple(torch.FloatTensor, int)
        Collection of :math:`B` sequence-label pairs, where the :math:`n^\\text{th}` sequence is of shape :math:`(T_n \\times D)` or :math:`(T_n,)` and the label is an integer.
    Returns
    -------
    padded_sequences: :class:`torch:torch.Tensor` (float)
        A tensor of size :math:`B \\times T_\\text{max} \\times D` containing all of the sequences in descending length order, padded to the length of the longest sequence in the batch.
    lengths: :class:`torch:torch.Tensor` (long/int)
        A tensor of the :math:`B` sequence lengths in descending order.
    labels: :class:`torch:torch.Tensor` (long/int)
        A tensor of the :math:`B` sequence labels in descending length order.
    """
    # print("start collate")
    batch_size = len(batch)

    # Sort the (sequence, label) pairs in descending order of duration
    batch.sort(key=(lambda x: len(x[0])), reverse=True)
    # Shape: list(tuple(tensor(TxD), int)) or list(tuple(tensor(T), int))

    # Create list of sequences, and tensors for lengths and labels
    exps, fngs, lengths_exp, lengths_fng, labels = [], [], torch.zeros(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)
    paths = []
    for i, (exp, fng, label, p), in enumerate(batch):
        lengths_fng[i], lengths_exp[i], labels[i] = len(fng), len(exp), label
        # print(f'fng: {fng.shape} - exp: {exp.shape}')
        fngs.append(fng)
        exps.append(exp)
        paths.append(p)


    # Combine fngs into a padded matrix

    # print(' - '.join([str(s.shape) for s in fngs]))
    # print(' - '.join([str(p) for p in paths]))
    padded_fngs = torch.nn.utils.rnn.pad_sequence(fngs, batch_first=True)
    padded_exps = torch.nn.utils.rnn.pad_sequence(exps, batch_first=True)
    # Shape: (B x T_max x D) or (B x T_max)

    # If a vector input was given for the fngs, expand (B x T_max) to (B x T_max x 1)
    if padded_exps.ndim == 2:
        padded_exps.unsqueeze_(-1)
    elif padded_exps.ndim == 4:
        padded_exps.squeeze_()
    if padded_fngs.ndim == 2:
        padded_fngs.unsqueeze_(-1)
    elif padded_fngs.ndim == 4:
        padded_fngs.squeeze_()

    # print("end collate")
    return padded_exps, padded_fngs, lengths_exp, lengths_fng, labels, paths


def collate_fn(batch):
    """Collects together univariate or multivariate sequences into a single batch, arranged in descending order of length.
    Also returns the corresponding lengths and labels as :class:`torch:torch.LongTensor` objects.
    Parameters
    ----------
    batch: list of tuple(torch.FloatTensor, int)
        Collection of :math:`B` sequence-label pairs, where the :math:`n^\\text{th}` sequence is of shape :math:`(T_n \\times D)` or :math:`(T_n,)` and the label is an integer.
    Returns
    -------
    padded_sequences: :class:`torch:torch.Tensor` (float)
        A tensor of size :math:`B \\times T_\\text{max} \\times D` containing all of the sequences in descending length order, padded to the length of the longest sequence in the batch.
    lengths: :class:`torch:torch.Tensor` (long/int)
        A tensor of the :math:`B` sequence lengths in descending order.
    labels: :class:`torch:torch.Tensor` (long/int)
        A tensor of the :math:`B` sequence labels in descending length order.
    """
    # print("start collate")
    batch_size = len(batch)
    # print(batch)
    # Sort the (sequence, label) pairs in descending order of duration
    batch.sort(key=(lambda x: len(x[0])), reverse=True)
    # Shape: list(tuple(tensor(TxD), int)) or list(tuple(tensor(T), int))

    # Create list of sequences, and tensors for lengths and labels
    feats, lengths_feat, labels = [],  torch.zeros(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)
    paths = []
    for i, (feat, label, p), in enumerate(batch):
        lengths_feat[i], labels[i] = len(feat), label
        # print(f'fng: {fng.shape} - exp: {exp.shape}')
        feats.append(feat)
        paths.append(p)


    # Combine fngs into a padded matrix

    # print(' - '.join([str(s.shape) for s in fngs]))
    # print(' - '.join([str(p) for p in paths]))
    padded_feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    # Shape: (B x T_max x D) or (B x T_max)

    # If a vector input was given for the fngs, expand (B x T_max) to (B x T_max x 1)
    # if padded_exps.ndim == 2:
    #     padded_exps.unsqueeze_(-1)
    # elif padded_exps.ndim == 4:
    #     padded_exps.squeeze_()
    if padded_feats.ndim == 2:
        padded_feats.unsqueeze_(-1)
    elif padded_feats.ndim == 4:
        padded_feats.squeeze_()

    # print("end collate")
    return padded_feats, lengths_feat, labels, paths


def load_rep(klass):
    if klass == "rep_velocity_mikrokosmos":
        path = os.path.join('representations', 'mikrokosmos', 'rep_velocity.pickle')
    elif klass == "rep_note_mikrokosmos":
        path = os.path.join('representations', 'mikrokosmos', 'rep_note.pickle')
    elif klass == "rep_nakamura_mikrokosmos":
        path = os.path.join('representations', 'mikrokosmos', 'rep_nakamura.pickle')
    elif klass == "rep_fng_henle":
        path = os.path.join('representations', 'henle', 'rep_velocity.pickle')
    elif klass == "rep_note_henle":
        path = os.path.join('representations', 'henle', 'rep_note.pickle')
    elif klass == "rep_nakamura_henle":
        path = os.path.join('representations', 'henle', 'rep_nakamura.pickle')



    data = utils.load_binary(path)
    ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    return ans


def load_rep_info(klass):
    if klass == "rep_velocity_mikrokosmos":
        path = os.path.join('representations', 'mikrokosmos', 'rep_velocity.pickle')
    elif klass == "rep_note_mikrokosmos":
        path = os.path.join('representations', 'mikrokosmos', 'rep_note.pickle')
    elif klass == "rep_nakamura_mikrokosmos":
        path = os.path.join('representations', 'mikrokosmos', 'rep_nakamura.pickle')
    elif klass == "rep_fng_henle":
        path = os.path.join('representations', 'henle', 'rep_velocity.pickle')
    elif klass == "rep_note_henle":
        path = os.path.join('representations', 'henle', 'rep_note.pickle')
    elif klass == "rep_nakamura_henle":
        path = os.path.join('representations', 'henle', 'rep_nakamura.pickle')

    data = utils.load_binary(path)
    ans = np.array([k for k, x in data.items()])
    return ans


def get_henle_expressivity():
    data = load_json("henleXmus/index_clean.json")
    X_train, X_val, X_test, X_test_split, y_train, y_val, y_test, y_test_split, ids_train, ids_val, ids_test, ids_test_split = \
        [], [], [], [], [], [], [], [], [], [], [], []
    # train
    for path, info in data.items():
        # if path == '2364046':
        #     print()

        embedding = torch.load(f'henle_embedding/{path}.pt')['total_note_cat'].transpose(0, 1) # torch.rand(10, 64)
        if info["subset"] == "train":
            X_train.append(embedding)
            y_train.append(info["henle"])
            ids_train.append(path)
        elif info["subset"] == "val":
            X_val.append(embedding)
            y_val.append(info["henle"])
            ids_val.append(path)
        elif info["subset"] == "test":
            X_test.append(embedding)
            y_test.append(info["henle"])
            ids_test.append(path)
        elif info["subset"] == "split":
            X_test_split.append(embedding)
            y_test_split.append(info["henle"])
            ids_test_split.append(path)

    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(X_test_split),\
           np.array(y_train), np.array(y_val), np.array(y_test), np.array(y_test_split),\
           np.array(ids_train), np.array(ids_val), np.array(ids_test), np.array(ids_test_split),


def get_henle_fingering_excerpts(representation_type):
    data = load_json("henleXmus/index_clean.json")
    X_train, X_val, X_test, X_test_split, y_train, y_val, y_test, y_test_split, ids_train, ids_val, ids_test, ids_test_split =\
        [], [], [], [], [], [], [], [], [], [], [], []
    # train
    for path, info in data.items():
        # if path == '2364046':
        #     print()
        embedding = torch.load(f'henle_excerpts/{representation_type}/{path}.pt').cpu()# torch.rand(10, 64)
        # pdb.set_trace()
        if info["subset"] == "train":
            X_train.append(embedding)
            y_train.append(int(info["henle"]) - 1)
            ids_train.append(path)
        elif info["subset"] == "val":
            X_val.append(embedding)
            y_val.append(int(info["henle"]) - 1)
            ids_val.append(path)
        elif info["subset"] == "test":
            X_test.append(embedding)
            y_test.append(int(info["henle"]) - 1)
            ids_test.append(path)
        elif info["subset"] == "split":
            X_test_split.append(embedding)
            y_test_split.append(int(info["henle"]) - 1)
            ids_test_split.append(path)

    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(X_test_split), \
           np.array(y_train), np.array(y_val), np.array(y_test), np.array(y_test_split), \
           np.array(ids_train), np.array(ids_val), np.array(ids_test), np.array(ids_test_split)

def get_excerpts(representation_type):
    if representation_type == "exp":
        return get_henle_expressivity()
    else:
        return get_henle_fingering_excerpts(representation_type)


def get_split_expressivity(random_state):
    split = load_json("Mikrokosmos-difficulty/splits.json")[str(random_state)]
    X_train, X_test, y_train, y_test, ids_train, ids_test = [], [], [], [], [], []
    # train
    for y, idx in zip(split['y_train'], split['ids_train']):
        path = Path(idx).stem
        X_train.append(torch.load(f'mikrokosmos_embedding/{path}.pt')['total_note_cat'])
        y_train.append(y)
        ids_train.append(path)

    # test
    for y, idx in zip(split['y_test'], split['ids_test']):
        path = Path(idx).stem
        X_test.append(torch.load(f'mikrokosmos_embedding/{path}.pt')['total_note_cat'])
        y_test.append(y)
        ids_test.append(path)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), np.array(ids_train), np.array(ids_test)


class mikrokosmos_expressivity(torch.utils.data.Dataset):
    def __init__(
            self,
            random_start=True,
            subset=0,
            split_number=0,
    ):
        """
        """
        self.set = subset
        self.X_train, self.X_test, self.y_train, \
        self.y_test, self.paths_train, self.paths_test = get_split_expressivity(split_number)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        ### get the file with index in the corresponding subset
        if self.set == 0:
            matrix = self.X_train[index].transpose(0, 1)
            label = torch.tensor(self.y_train[index], dtype=torch.float)
            path = self.paths_train[index]
        else:  # self.set == 1:
            matrix = self.X_test[index].transpose(0, 1)
            label = torch.tensor(self.y_test[index], dtype=torch.float)
            path = self.paths_test[index]

        return matrix, label, path


    def get_path(self, index):
        if self.set == 0:
            path = self.paths_train[index]
        else:   # self.set == 1:
            path = self.paths_test[index]
        return path


    def __len__(self):
        if self.set == 0:
            return len(self.X_train)
        else:
            return len(self.X_test)


soft_labels = False


def get_split_fingering(random_state, X, paths):
    split = load_json("Mikrokosmos-difficulty/splits.json")[str(random_state)]
    X_train, X_test, y_train, y_test, ids_train, ids_test = [], [], [], [], [], []
    paths = np.array([p.replace('mikrokosmos', 'Mikrokosmos-difficulty') for p in paths])
    # train'
    for y, idx in zip(split['y_train'], split['ids_train']):
        index = np.where(paths == idx)[0][0]
        X_train.append(X[index])
        y_train.append(y)
        ids_train.append(int(os.path.basename(idx)[:-4]))

    # test
    for y, idx in zip(split['y_test'], split['ids_test']):
        index = np.where(paths == idx)[0][0]
        X_test.append(X[index])
        y_test.append(y)
        ids_test.append(int(os.path.basename(idx)[:-4]))
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), np.array(ids_train), np.array(ids_test)



def get_henle_fingering(X, paths):
    data = load_json("henleXmus/index_clean.json")
    X_train, X_val, X_test, X_test_split, y_train, y_val, y_test, y_test_split, ids_train, ids_val, ids_test, ids_test_split = \
        [], [], [], [], [], [], [], [], [], [], [], []
    # train
    for path, info in data.items():
        index = np.where(paths == path)[0][0]
        if info["subset"] == "train":
            X_train.append(X[index])
            y_train.append(int(info["henle"]) - 1)
            ids_train.append(path)
        elif info["subset"] == "val":
            X_val.append(X[index])
            y_val.append(int(info["henle"]) - 1)
            ids_val.append(path)
        elif info["subset"] == "test":
            X_test.append(X[index])
            y_test.append(int(info["henle"]) - 1)
            ids_test.append(path)
        elif info["subset"] == "split":
            X_test_split.append(X[index])
            y_test_split.append(int(info["henle"]) - 1)
            ids_test_split.append(path)

    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(X_test_split),\
           np.array(y_train), np.array(y_val), np.array(y_test), np.array(y_test_split),\
           np.array(ids_train), np.array(ids_val), np.array(ids_test), np.array(ids_test_split)



def create_dataset_mikrokosmos_expressivity(split_number, representation_type):
    train_dataset = mikrokosmos_expressivity(subset=0, split_number=split_number)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=64, num_workers=4,
                                               pin_memory=True)

    test_dataset = mikrokosmos_expressivity(subset=1, split_number=split_number)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=len(list(test_dataset)),
                                                  num_workers=4, pin_memory=True)

    return train_loader, test_loader


class mikrokosmos_fingering(torch.utils.data.Dataset):
    def __init__(
            self,
            random_start=True,
            subset=0,
            split_number=0,
            representation_type='rep_velocity',
    ):
        """
        """
        X, y = load_rep(representation_type)
        paths = load_rep_info(representation_type)

        max_X = np.max([np.max(x) for x in X])
        X = [np.array([xx / max_X for xx in x]) for x in X]

        self.set = subset
        self.random_start = random_start

        #### build the train subsets: train, test using train_test_split, a stratified split with the labels
        self.X_train, self.X_test, self.y_train, self.y_test, self.paths_train, self.paths_test = get_split_fingering(split_number, X, paths)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        #### get the file with index in the corresponding subset
        if self.set == 0:
            matrix = torch.from_numpy(self.X_train[index].astype(np.float64))
            label = torch.tensor(self.y_train[index], dtype=torch.float)
            path = self.paths_train[index]
        else:  # self.set == 1:
            matrix = torch.from_numpy(self.X_test[index].astype(np.float64))
            label = torch.tensor(self.y_test[index], dtype=torch.float)
            path = self.paths_test[index]
        return matrix, label, path


    def get_path(self, index):
        if self.set == 0:
            path = self.paths_train[index]
        else:   # self.set == 1:
            path = self.paths_test[index]
        return path


    def __len__(self):
        if self.set == 0:
            return len(self.X_train)
        else:
            return len(self.X_test)


soft_labels = False


def create_dataset_mikrokosmos_fingering(split_number, representation_type):
    train_dataset = mikrokosmos_fingering(subset=0, split_number=split_number, representation_type=representation_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=64, num_workers=4,
                                               pin_memory=True)

    test_dataset = mikrokosmos_fingering(subset=1, split_number=split_number, representation_type=representation_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=len(list(test_dataset)),
                                                  num_workers=4, pin_memory=True)

    return train_loader, test_loader


class mikrokosmos_full(torch.utils.data.Dataset):
    def __init__(
            self,
            random_start=True,
            subset=0,
            split_number=0,
    ):
        """
        """
        self.set = subset
        self.X_train_exp, self.X_test_exp, self.y_train, \
        self.y_test, self.paths_train, self.paths_test = get_split_expressivity(split_number)


        X, y = load_rep(f'rep_{sys.argv[3]}_mikrokosmos')
        paths = load_rep_info(f'rep_{sys.argv[3]}_mikrokosmos')

        max_X = np.max([np.max(x) for x in X])
        X = [np.array([xx / max_X for xx in x]) for x in X]
        self.X_train_fng, self.X_test_fng, self.y_train_fng, \
        self.y_test_fng, self.paths_train_fng, self.paths_test_fng = get_split_fingering(split_number, X, paths)




    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        ### get the file with index in the corresponding subset
        if self.set == 0:
            fng = torch.from_numpy(self.X_train_fng[index].astype(np.float64))
            exp = self.X_train_exp[index].transpose(0, 1)
            label = torch.tensor(self.y_train[index], dtype=torch.float)
            path = self.paths_train[index]
        else:  # self.set == 1:
            fng = torch.from_numpy(self.X_test_fng[index].astype(np.float64))
            exp = self.X_test_exp[index].transpose(0, 1)
            label = torch.tensor(self.y_test[index], dtype=torch.float)
            path = self.paths_test[index]

        return exp, fng, label, path


    def get_path(self, index):
        if self.set == 0:
            path = self.paths_train[index]
        else:   # self.set == 1:
            path = self.paths_test[index]
        return path


    def __len__(self):
        if self.set == 0:
            return len(self.X_train_fng)
        else:
            return len(self.X_test_fng)



def create_dataset_mikrokosmos_full(split_number):
    train_dataset = mikrokosmos_full(subset=0, split_number=split_number)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn_mk, batch_size=64, num_workers=4,
                                               pin_memory=True)

    test_dataset = mikrokosmos_full(subset=1, split_number=split_number)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn_mk, batch_size=len(list(test_dataset)),
                                                  num_workers=4, pin_memory=True)

    return train_loader, test_loader


def normalize_list_numpy(list_numpy):
    normalized_list = minmax_scale(list_numpy)
    return normalized_list


def recompute_excerpts(model, paths, epoch, representation):
    print(f"recomputing {representation}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ans_paths = []

    model.eval()
    for p in paths:
        fng = torch.load(f"henle_fingering_embedding/{representation}/{p}.pt").unsqueeze_(dim=0)
        fng = fng.to(device)
        if epoch is None:
            hop_size = 650
        else:
            hop_size = 300 + epoch * 30
        if fng.shape[1] <= hop_size:
            print(f"Less than {hop_size} samples!!")
            ans_paths.append(fng.cpu().squeeze_())
            # pdb.set_trace()
        else:
            print(f"START!!!!!!!!!!!!  {p}.pdf")
            logits = []
            # pdb.set_trace()
            fs = fng.unfold(step=1, size=hop_size, dimension=1).transpose(0, 1).transpose(2, 3).squeeze(dim=1)


            for idx in range(0, fs.shape[0], 100):
                jdx = idx + 100 if idx + 100 < fs.shape[0] else fs.shape[0]
                f = fs[idx:jdx, :, :]
                y = model(f.float(), torch.Tensor([hop_size] * (jdx - idx)))
                logits.extend(y.cpu().detach().numpy())

            logits = normalize_list_numpy(logits)
            index_better, best_logit = 0, 0.0
            fragment_class = np.argmax(logits, axis=1)
            max_class = np.max(fragment_class)
            for idx, f in enumerate(fragment_class):
                if max_class == f and logits[idx][f] > best_logit:
                    index_better = idx
                    best_logit = logits[idx][f]
            best_excerpt = torch.squeeze(fs[index_better]).cpu()
            ans_paths.append(best_excerpt)
            #  pdb.set_trace()
            print(f"best rank index!!! {index_better} of {fs.shape[0]},  size {best_excerpt.shape}")
    # pdb.set_trace()
    return np.array(ans_paths)


class henle_full(torch.utils.data.Dataset):
    def __init__(
            self,
            representation,
            random_start=True,
            subset=0,
            subset_type='full',
            split_number=0,
    ):
        """
        """
        self.set_type = subset_type
        self.set = subset
        # self.X_train_exp, self.X_val_exp, self.X_test_exp, \
        # _, _, _, \
        # self.paths_train, self.paths_val, self.paths_test = get_henle_expressivity()
        self.representation_type = representation


        if self.set_type == "excerpt":
            self.X_train, self.X_val, self.X_test, self.X_test_split,\
            self.y_train, self.y_val, self.y_test, self.y_test_split, \
            self.paths_train, self.paths_val, self.paths_test, self.paths_test_split = get_henle_fingering_excerpts(self.representation_type)
        elif self.set_type == "full":
            if self.representation_type in ["fng", "nakamura", "note"]:
                X, y = load_rep(f'rep_{self.representation_type}_henle')
                paths = load_rep_info(f'rep_{self.representation_type}_henle')


                maximum_per_piece = []
                for x, p in enumerate(zip(X, paths)):
                    maximum_per_piece.append(np.max(x))

                max_X = np.max(maximum_per_piece)
                X = [np.array([xx / max_X for xx in x]) for x in X]

                self.X_train, self.X_val, self.X_test, self.X_test_split, \
                self.y_train, self.y_val, self.y_test, self.y_test_split, \
                self.paths_train, self.paths_val, self.paths_test, self.paths_test_split = get_henle_fingering(X, paths)
            elif self.representation_type == "exp":
                self.X_train, self.X_val, self.X_test, self.X_test_split,\
                self.y_train, self.y_val, self.y_test, self.y_test_split,\
                self.paths_train, self.paths_val, self.paths_test, self.paths_test_split = get_henle_expressivity()
                # pdb.set_trace()
                self.y_train, self.y_val, self.y_test, self.test_split = self.y_train.astype(int) - 1, \
                                                        self.y_val.astype(int) - 1, \
                                                        self.y_test.astype(int) - 1, \
                                                        self.y_test_split.astype(int) - 1

        count = dict(Counter(self.y_train))
        print(len(self.y_train), len(self.y_val), len(self.y_test), len(self.y_test_split))
        # pdb.set_trace()
        weight_class = []
        for ii in range(0, 9):
            weight_class.append(count[ii])

        # train: Counter({ 0: 9, 1: 23, 2: 27, 3: 49, 4: 62, 5: 67, 6: 50,  7: 23, 8: 20})
        # tensor([0.1111, 0.0526, 0.0588, 0.0250, 0.0208, 0.0217, 0.0270, 0.0500, 0.0526])
        print(weight_class)

        self.weight_class = 1 / torch.Tensor(weight_class)
        print(self.weight_class)


    def recompute_excerpts(self, model, epoch):
        print("recomputing!!!")
        # pdb.set_trace()
        if self.set == 0 and self.set_type == "excerpt":
            self.X_train = recompute_excerpts(model, self.paths_train, epoch, self.representation_type)
        elif self.set == 1 and self.set_type == "excerpt":
            self.X_val = recompute_excerpts(model, self.paths_val, epoch, self.representation_type)
        elif self.set == 2 and self.set_type == "excerpt":
            self.X_test = recompute_excerpts(model, self.paths_test, epoch, self.representation_type)
        elif self.set == 3 and self.set_type == "excerpt":
            self.X_test_split = recompute_excerpts(model, self.paths_test_split, epoch, self.representation_type)
        # pdb.set_trace()


    def __getitem__(self, index):
        # print("start data loader")
        if torch.is_tensor(index):
            index = index.tolist()
        ### get the file with index in the corresponding subset
        if self.set == 0:
            if self.set_type == "full" and self.representation_type != 'exp':
                fng = torch.from_numpy(self.X_train[index].astype(np.float64))
            elif self.set_type == "excerpt" or self.representation_type == 'exp':
                fng = self.X_train[index]
            label = torch.tensor(int(self.y_train[index]), dtype=torch.float)
            path = self.paths_train[index]
        elif self.set == 1:
            if self.set_type == "full" and self.representation_type != 'exp':
                fng = torch.from_numpy(self.X_val[index].astype(np.float64))
            elif self.set_type == "excerpt" or self.representation_type == 'exp':
                fng = self.X_val[index]
            label = torch.tensor(int(self.y_val[index]), dtype=torch.float)
            path = self.paths_val[index]
        elif self.set == 2:
            if self.set_type == "full" and self.representation_type != 'exp':
                fng = torch.from_numpy(self.X_test[index].astype(np.float64))
            elif self.set_type == "excerpt" or self.representation_type == 'exp':
                # print(self.X_test[index])
                fng = self.X_test[index]
            label = torch.tensor(int(self.y_test[index]), dtype=torch.float)
            path = self.paths_test[index]
        elif self.set == 3:
            if self.set_type == "full" and self.representation_type != 'exp':
                fng = torch.from_numpy(self.X_test_split[index].astype(np.float64))
            elif self.set_type == "excerpt" or self.representation_type == 'exp':
                # print(self.X_test_split[index])
                fng = self.X_test_split[index]
            label = torch.tensor(int(self.y_test_split[index]), dtype=torch.float)
            path = self.paths_test_split[index]
        return fng, label, path


    def get_path(self, index):
        if self.set == 0:
            path = self.paths_train[index]
        elif self.set == 1:
            path = self.paths_val[index]
        elif self.set == 2:
            path = self.paths_test[index]
        elif self.set == 3:
            path = self.paths_test[index]
        return path


    def __len__(self):
        if self.set == 0:
            return len(self.X_train)
        if self.set == 1:
            return len(self.X_val)
        if self.set == 2:
            return len(self.X_test)
        if self.set == 3:
            return len(self.X_test_split)


def create_dataset_henle_full(subset_type, representation):
    train_dataset = henle_full(subset=0, subset_type=subset_type, representation=representation)
    weight_class = train_dataset.weight_class
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=64, num_workers=0,
                                               pin_memory=True)

    val_dataset = henle_full(subset=1, subset_type=subset_type, representation=representation)
    val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, batch_size=len(list(val_dataset)),
                                              num_workers=0, pin_memory=True)

    test_dataset = henle_full(subset=2, subset_type=subset_type, representation=representation)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=len(list(test_dataset)),
                                                  num_workers=0, pin_memory=True)

    test_full_dataset = henle_full(subset=2, subset_type='full', representation=representation)
    test_full_loader = torch.utils.data.DataLoader(test_full_dataset, collate_fn=collate_fn, batch_size=len(list(test_full_dataset)),
                                                   num_workers=0, pin_memory=True)

    test_split_dataset = henle_full(subset=3, subset_type=subset_type, representation=representation)
    test_split_loader = torch.utils.data.DataLoader(test_split_dataset, collate_fn=collate_fn,
                                                   batch_size=len(list(test_full_dataset)),
                                                   num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, test_full_loader, test_split_loader, weight_class


def create_dataset_henle_full_1(rep):
    train_dataset = henle_full(subset=0, subset_type='full', representation=rep)
    weight_class = train_dataset.weight_class
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, num_workers=2,
                                               pin_memory=True)

    val_dataset = henle_full(subset=1, subset_type='full', representation=rep)
    val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1,
                                              num_workers=2, pin_memory=True)

    test_dataset = henle_full(subset=2, subset_type='full', representation=rep)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=1,
                                                  num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, weight_class



if __name__ == '__main__':
    train_loader, val_loader, test_loader, test_full_loader, weight_class = create_dataset_henle_full('excerpt', 'exp')
