

"""
    File name: loader_representations.py
    Author: Pedro Ramoneda
    Python Version: 3.7
"""
import csv
import io
import os
import json
import pdb
import sys
from statistics import mean

import PIL
import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, kendalltau

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn import preprocessing
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder
from data import *
from torchvision.transforms import ToTensor

# ----------------------------------------------------------------------------------------------------------------------
import utils
from henle_fingering import load_xmls

from utils import load_json, save_json

import os
import json
import pdb
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, kendalltau
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

from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.losses import coral_loss
from coral_pytorch.layers import CoralLayer


# ----------------------------------------------------------------------------------------------------------------------
import utils
from utils import load_json, save_json


logging = True


def save_model(path, epoch, model, optimizer, criterion):

    if len(sys.argv) == 3 and sys.argv[2] == "cluster":
        path = f"/homedtic/pramoneda/gnn_fingering/{path}"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion': criterion
    }, path)



def start_logging(args, lr, n_epochs, patience):
    if not os.path.exists(f"/{args['alias']}"):
        os.mkdir(f"/{args['alias']}/")
    if not os.path.exists(f"/{args['alias']}/{args['architecture']}"):
        os.mkdir(f"/{args['alias']}/{args['architecture']}")

    writer = SummaryWriter(f"/{args['alias']}/{args['architecture']}", comment=args['architecture'])
    params = {"learning_rate": lr, "max_epochs": n_epochs, "patience": patience}
    for k, v in params.items():
        print(k)
        writer.add_scalar("parameters/" + k, v)
    return writer


class DeepGRU(nn.Module):
    def __init__(self, num_features, num_classes, representation_type, device=None, representation=None):
        super(DeepGRU, self).__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_features = num_features
        self.num_classes = num_classes
        self.representation_type = representation

        input_size_gru = None
        if representation_type == 'exp':
            input_size_gru = 64
        elif representation_type == 'note':
            input_size_gru = 88
        else:
            input_size_gru = 10

        # Branch fingering
        self.gru_fng_1 = nn.GRU(input_size_gru, 512, 2, dropout=0, batch_first=True)
        self.gru_fng_2 = nn.GRU(512, 256, 2, dropout=0, batch_first=True)
        self.gru_fng_3 = nn.GRU(256, 128, 1, dropout=0, batch_first=True)
        self.gru_fng_exp = nn.GRU(input_size_gru, 128, 3, dropout=0.9, batch_first=True)
        self.attention_fng = Attention(128, device=device)

        input_classification_size = 256  #  if self.representation_type in ['fng', 'exp'] else 512

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_classification_size),
            nn.Dropout(0.5),
            nn.Linear(input_classification_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            CoralLayer(size_in=128, num_classes=num_classes) if len(sys.argv) == 3 and sys.argv[2] in ["coral"] else nn.Linear(128, num_classes)
        )

        self.last = None
        if len(sys.argv) == 3 and sys.argv[2] in ["soft", "coral"]:
            self.last = nn.Sigmoid()
        else:
            self.last = nn.LogSoftmax(dim=1)
        self.to(device)


    def forward(self, fng_padded, fng_lengths):
        # branch fng_padded
        fng_padded = packer(fng_padded, fng_lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Encode

        # if self.representation_type != "exp":
        # output_fng, _ = self.gru_fng_1(fng_padded.float())
        # output_fng, _ = self.gru_fng_2(output_fng)
        # output_fng, hidden = self.gru_fng_3(output_fng)
        # else:
        output_fng, hidden = self.gru_fng_exp(fng_padded.float())

        # Pass to attention with the original padding
        output_padded_fng, _ = padder(output_fng, batch_first=True)
        attn_fng_output = self.attention_fng(output_padded_fng, hidden[-1:])

        if len(sys.argv) == 3 and sys.argv[2] in ["coral"]:
            logits = self.classifier(attn_fng_output)
            probas = self.last(logits)
            return logits, probas
        else:
            return self.last(self.classifier(attn_fng_output))


    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, attention_dim, device):
        super(Attention, self).__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(128, 128, 1, batch_first=True)
        self.to(device)

    def forward(self, input_padded, hidden):
        e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1))
        context = context.permute(0, 2, 1)

        # Compute the auxiliary context, and concat
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output



def get_acc(model, device, loader, epoch, verbose=False):
    model.eval()
    # Retrieve test set as a single batch
    y_pred, y_labels, y_paths = [], [], []
    for i, (fng, fng_lengths, labels, paths) in enumerate(loader):
        fng, fng_lengths, labels = fng.to(device), fng_lengths.to(device), labels.to(device)

        if len(sys.argv) == 3 and sys.argv[2] == 'coral':
            _, proba = model(fng.float(), fng_lengths)
            ys = proba_to_label(proba).cpu()
        else:
            logits = model(fng.float(), fng_lengths)
            ys = torch.argmax(logits, dim=1).cpu().tolist()
        y_pred.extend(ys)
        y_paths.extend(paths)
        y_labels.extend(labels.cpu().tolist())

        # print(window_preds)
    # Calculate accuracy
    # pdb.set_trace()
    acc = balanced_accuracy_score(y_pred=y_pred, y_true=y_labels)
    return y_pred, y_labels, acc, y_paths


def normalize_list_numpy(list_numpy):
    normalized_list = minmax_scale(list_numpy)
    return normalized_list




def save_confusion_matrix(writer, model, device, test_loader, epoch, verbose, alias="test"):
    y_pred, y_labels, _, _ = get_acc(model, device, test_loader, epoch=epoch, verbose=verbose)

    # Calculate confusion matrix
    classes = range(9)
    cm = confusion_matrix(y_labels, y_pred, labels=classes)

    # Display accuracy and confusion matrix
    labels = range(9)

    df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(7, 7))
    sns.heatmap(df, annot=True, cbar=False)
    plt.title(f'Confusion matrix for {alias} set predictions', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # save into bufer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # buffer to pytorch
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    # grab in tensorboard
    writer.add_image(f"{alias}_cm9", image, global_step=epoch)


def save_mse(y_true, y_pred, alias, writer):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    writer.add_text(f"MSE_early_stopping_{alias}", str(mse))


def save_3_classes(y_true, y_pred, alias, writer):
    mask = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    bacc = balanced_accuracy_score(y_pred=[mask[yy] for yy in y_pred], y_true=[mask[yy] for yy in y_true])
    writer.add_text(f"3class_early_stopping_{alias}", str(bacc))


def save_1_ACC(y_true, y_pred, alias, writer):
    matches = [1 if pp in [tt-1, tt, tt+1] else 0 for tt, pp in zip(y_true, y_pred)]
    acc_plusless_1 = sum(matches) / len(matches)
    writer.add_text(f"acc_plusless_1_early_stopping_{alias}", str(acc_plusless_1))


def classification(train_loader,  val_loader, test_loader, test_full_loader, test_split_loader, verbose=True, representation_type="rep_velocity", writer=None, weight_class=None):
    print("classification")
    # Create a DeepGRU neural network model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    weight_class = weight_class.to(device)
    patience, trials = 100, 0
    best_acc = 0
    n_features = 2048
    n_grades = 9
    model = DeepGRU(n_features, n_grades, sys.argv[2], device=None)
    best_model = model

    soft_labels = True if len(sys.argv) == 3 and sys.argv[2] == 'soft' else False
    # Set loss function and optimizer

    if soft_labels:
        criterion = torch.nn.BCELoss()
    if len(sys.argv) == 3 and sys.argv[2] == 'coral':
        criterion = coral_loss
    else:
        criterion = torch.nn.NLLLoss(weight=weight_class)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

    # Toggle evaluation mode
    epochs = 2000

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        running_loss = []
        for i, (feat, feat_length, labels, paths) in enumerate(train_loader):


            feat, feat_length, labels = feat.to(device), \
                                        feat_length.to(device), \
                                        labels.to(device)
            model.train()
            optimizer.zero_grad()


            if soft_labels:
                prob = model(feat, feat_length)
                labels_one_hot = F.one_hot(labels, num_classes=n_grades).float()
                labels_soft = gaussian_filter1d(labels_one_hot.cpu().numpy(), 0.8)
                # pdb.set_trace()
                labels_soft = torch.Tensor(labels_soft).to(device)
                loss = criterion(prob.float(), labels_soft.float())
            elif len(sys.argv) == 3 and sys.argv[2] == 'coral':
                logits, _ = model(feat, feat_length)
                levels = levels_from_labelbatch(labels, num_classes=n_grades)
                levels = levels.to(device)
                loss = coral_loss(logits, levels, importance_weights=weight_class)
            else:
                prob = model(feat, feat_length)
                loss = criterion(prob.float(), labels)
            loss.backward()
            # Update the optimizer
            optimizer.step()
            running_loss.append(loss.item())

            # print(window_preds)

        train_pred, train_labels, train_acc, _ = get_acc(model, device, train_loader, epoch=epoch, verbose=verbose)
        print(f"Train:  {train_acc:2.2%}")

        val_pred, val_labels, val_acc, _ = get_acc(model, device, val_loader, epoch=epoch, verbose=verbose)
        # pdb.set_trace()
        print(f"Validation:  {val_acc:2.2%}")

        test_pred, test_labels, test_acc, test_paths = get_acc(model, device, test_loader, epoch=epoch, verbose=verbose)
        print(f"Test:  {test_acc:2.2%}")

        test_split_pred, test_split_labels, test_split_acc, test_split_paths = get_acc(model, device, test_split_loader, epoch=epoch, verbose=verbose)
        print(f"Test_split:  {test_split_acc:2.2%}")
        print(f"Loss = {mean(running_loss)}")

        if logging:
            writer.add_scalar(f"train", train_acc, epoch)
            writer.add_scalar(f"val", val_acc, epoch)
            writer.add_scalar(f"test", test_acc, epoch)
            writer.add_scalar(f"loss", mean(running_loss), epoch)

        if epoch % 5 == 0:
            print(
                f'Epoch: {epoch:3d}. Loss: {mean(running_loss):.4f}. val Acc: ({val_acc:2.2%} test ACC: ({test_acc:2.2%})')

        if epoch > 15 and epoch % 10 == 0:
            train_loader.dataset.recompute_excerpts(best_model, epoch)
            val_loader.dataset.recompute_excerpts(best_model, epoch)
            test_loader.dataset.recompute_excerpts(best_model, epoch)
            test_split_loader.dataset.recompute_excerpts(best_model, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            save_model(f'models/best_henle_{sys.argv[1]}.pth',
                       epoch, model, optimizer, criterion)
            writer.add_text("early_stopping_train", str(train_acc))
            writer.add_text("early_stopping_val", str(val_acc))
            writer.add_text("early_stopping_test", str(test_acc))
            writer.add_text("early_stopping_test_partial", str(test_split_acc))



            trials = 0
            save_confusion_matrix(writer, model, device, train_loader, epoch=epoch, verbose=verbose, alias='train')
            save_confusion_matrix(writer, model, device, val_loader, epoch=epoch, verbose=verbose, alias='val')
            save_confusion_matrix(writer, model, device, test_loader, epoch=epoch, verbose=verbose, alias='test')
            save_confusion_matrix(writer, model, device, test_split_loader, epoch=epoch, verbose=verbose, alias='partial')

            save_3_classes(y_true=train_labels, y_pred=train_pred, alias='train', writer=writer)
            save_3_classes(y_true=val_labels, y_pred=val_pred, alias='val', writer=writer)
            save_3_classes(y_true=test_labels, y_pred=test_pred, alias='test', writer=writer)
            save_3_classes(y_true=test_split_labels, y_pred=test_split_pred, alias='partial', writer=writer)

            save_1_ACC(y_true=train_labels, y_pred=train_pred, alias='train', writer=writer)
            save_1_ACC(y_true=val_labels, y_pred=val_pred, alias='val', writer=writer)
            save_1_ACC(y_true=test_labels, y_pred=test_pred, alias='test', writer=writer)
            save_1_ACC(y_true=test_split_labels, y_pred=test_split_pred, alias='partial', writer=writer)


            save_mse(y_true=train_labels, y_pred=train_pred, alias='train', writer=writer)
            save_mse(y_true=val_labels, y_pred=val_pred, alias='val', writer=writer)
            save_mse(y_true=test_labels, y_pred=test_pred, alias='test', writer=writer)
            save_mse(y_true=test_split_labels, y_pred=test_split_pred, alias='partial', writer=writer)


            test_full_pred, test_full_labels, test_full_acc, _ = get_acc(model, device, test_full_loader, epoch=epoch, verbose=verbose)
            writer.add_text("early_stopping_test_full", str(test_full_acc))
            writer.add_scalar("test_full", test_full_acc, epoch)

            save_confusion_matrix(writer, model, device, test_full_loader, epoch=epoch, verbose=verbose, alias='test_full')
            save_mse(y_true=test_full_labels, y_pred=test_full_pred, alias='test_full', writer=writer)
            save_3_classes(y_true=test_full_labels, y_pred=test_full_pred, alias='test_full', writer=writer)
            save_1_ACC(y_true=test_full_labels, y_pred=test_full_pred, alias='test_full', writer=writer)
            writer.add_scalar(f"test_full", test_full_acc, epoch)
            print(f"Test full:  {test_full_acc:2.2%}")

            writer.add_text("last labels test", str(test_labels))
            writer.add_text("last pred test", str(test_pred))
            writer.add_text("last path test", str(test_paths))

            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%} ')

        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break


def attention():
    if not os.path.exists('henle'):
        os.mkdir('henle')
    writer = SummaryWriter(f"runs/henle_{sys.argv[1]}/")
    # len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    train_loader, val_loader, test_loader, test_full_loader, test_split_loader, weight_class = create_dataset_henle_full(subset_type="excerpt", representation=sys.argv[2])
    # pdb.set_trace()≤
    classification(train_loader, val_loader, test_loader, test_full_loader, test_split_loader, writer=writer, weight_class=weight_class)


if __name__ == '__main__':
    attention()
