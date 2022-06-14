import pdb

import numpy as np
import torch
from sklearn.preprocessing import minmax_scale

from data import create_dataset_henle_full_1
import os

from run_model_full_mk import DeepGRU

def load_model(path, model, optimizer=None, device=None):
    if device is None:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], map_location=device)
    epoch = checkpoint['epoch']
    criterion = checkpoint['criterion']
    return model


def normalize_list_numpy(list_numpy):
    normalized_list = minmax_scale(list_numpy)
    return normalized_list


def create_excerpts(rep='fng', path_model='results/mikrokosmos_fng/split:2_epoch:20_rep_velocity.pkl', path_embeddings="henle_excerpts/fng"):

    os.makedirs(path_embeddings, exist_ok=True)

    train_loader, val_loader, test_loader, weight_class = create_dataset_henle_full_1(rep)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = DeepGRU(num_features=10 if rep != 'note' else 88, num_classes=3, device=device)
    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    # pdb.set_trace()
    for loader in [train_loader, val_loader, test_loader]: # len(train_loader) + len(val_loader) +  len(test_loader)
        for i, (fng, fng_lenghts, labels, paths) in enumerate(loader):
            torch.save(fng.squeeze_(), f"henle_fingering_embedding/{os.path.basename(path_embeddings)}/{paths[0]}.pt")
            # pdb.set_trace()
            if not os.path.exists(f'{path_embeddings}/{paths[0]}.pt'):
                fng, fng_lenghts, labels = fng.to(device).unsqueeze_(dim=0), \
                                           fng_lenghts.to(device), \
                                           labels.to(device)
                if fng.shape[1] <= 650:
                    # pdb.set_trace()
                    torch.save(fng.squeeze_().cpu(), f"{path_embeddings}/{paths[0]}.pt")
                else:
                    print(f"START!!!!!!!!!!!!  {paths[0]}.pdf")
                    logits = []
                    # pdb.set_trace()
                    fs = fng.unfold(step=1, size=650, dimension=1).transpose(0, 1).transpose(2, 3).squeeze(dim=1)
                    for idx in range(0, fs.shape[0], 200):
                        jdx = idx + 200 if idx + 200 < fs.shape[0] else fs.shape[0]
                        f = fs[idx:jdx, :, :]
                        y = model(torch.rand(1, 10, 64).to(device), f.float(), torch.Tensor([1]), torch.Tensor([650] * (jdx-idx)).to(device))
                        logits.extend(y.cpu().detach().numpy())
                    index_better, best_logit = 0, 0.0
                    # pdb.set_trace()
                    fragment_class = np.argmax(logits, axis=1)
                    max_class = np.max(fragment_class)
                    for idx, f in enumerate(fragment_class):
                        if max_class == f and logits[idx][f] > best_logit:
                            index_better = idx
                            best_logit = logits[idx][f]
                    torch.save(torch.squeeze(fs[index_better]).cpu(), f"{path_embeddings}/{paths[0]}.pt")
                    print(f"best rank index!!! {index_better} of {fs.shape[0]}")
            else:
                print(f"COMPUTADO {paths[0]}.pt")


def create_excerpts_exp(path_model, path_embeddings):
    os.makedirs(path_embeddings, exist_ok=True)

    train_loader, val_loader, test_loader, weight_class = create_dataset_henle_full_1("exp")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = DeepGRU(num_features=10, num_classes=3)
    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    for loader in [train_loader, val_loader, test_loader]:
        for i, (exp, exp_lengths, labels, paths) in enumerate(loader):

            torch.save(exp.squeeze_(), f"henle_fingering_embedding/{os.path.basename(path_embeddings)}/{paths[0]}.pt")
            exp = exp.unsqueeze_(dim=0)
            # pdb.set_trace()
            if not os.path.exists(f'{path_embeddings}/{paths[0]}.pt'):
                exp, exp_lengths, labels = exp.to(device), \
                                                             exp_lengths.to(device), \
                                                             labels.to(device)
                if exp.shape[1] <= 650:
                    # pdb.set_trace()
                    torch.save(exp.squeeze_().cpu(), f"{path_embeddings}/{paths[0]}.pt")
                else:
                    print(f"START!!!!!!!!!!!!  {paths[0]}.pdf")
                    logits = []
                    # pdb.set_trace()
                    exps = exp.unfold(step=1, size=650, dimension=1).transpose(0, 1).transpose(2, 3).squeeze(dim=1)
                    for idx in range(0, exps.shape[0], 200):
                        jdx = idx + 200 if idx + 200 < exps.shape[0] else exps.shape[0]
                        ee = exps[idx:jdx, :, :]
                        # pdb.set_trace()
                        y = model(ee.float(), torch.rand(1, 10, 10).to(device), torch.Tensor([650] * (jdx-idx)).to(device), torch.Tensor([1]).to(device))
                        logits.extend(y.cpu().detach().numpy())
                    index_better, best_logit = 0, 0.0
                    # pdb.set_trace()
                    fragment_class = np.argmax(logits, axis=1)
                    max_class = np.max(fragment_class)
                    for idx, f in enumerate(fragment_class):
                        if max_class == f and logits[idx][f] > best_logit:
                            index_better = idx
                            best_logit = logits[idx][f]
                    torch.save(torch.squeeze(exps[index_better].cpu()), f"{path_embeddings}/{paths[0]}.pt")
                    print(f"best rank index!!! {index_better} of {exps.shape[0]}")
            else:
                print(f"COMPUTADO {paths[0]}.pt")


if __name__ == '__main__':
    # create_excerpts_exp(
    #     path_model='results/mikrokosmos_exp/split:21_epoch:20_rep_velocity.pkl',
    #     path_embeddings="henle_excerpts/exp"
    # )
    # create_excerpts(
    #     rep='fng',
    #     path_model='results/mikrokosmos_fng/split:2_epoch:20_rep_velocity.pkl',
    #     path_embeddings="henle_excerpts/fng"
    # )
    create_excerpts(
        rep='nakamura',
        path_model='results/mikrokosmos_nakamura/split:13_epoch:20_rep_velocity.pkl',
        path_embeddings="henle_excerpts/nakamura"
    )
    # create_excerpts(
    #     rep='note',
    #     path_model='results/mikrokosmos_note/split:42_epoch:20_rep_velocity.pkl',
    #     path_embeddings="henle_excerpts/note"
    # )