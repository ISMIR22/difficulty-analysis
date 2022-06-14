import json
import pickle

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, mean_squared_error
import json
import numpy as np


def compute_metrics(model, name_subset, X_subset, y_subset):
    pred = model.predict(X_subset)

    bacc = balanced_accuracy_score(y_pred=pred, y_true=y_subset)

    mse = mean_squared_error(y_pred=pred, y_true=y_subset)

    mask = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    bacc3 = balanced_accuracy_score(y_pred=[mask[yy] for yy in pred], y_true=[mask[yy] for yy in y_subset])

    matches = [1 if pp in [tt - 1, tt, tt + 1] else 0 for tt, pp in zip(y_subset, pred)]
    acc_plusless_1 = sum(matches) / len(matches)

    print(json.dumps(
        {
            f'bacc-{name_subset}': bacc,
            f'3bacc-{name_subset}': bacc3,
            f'acc_plusless_1-{name_subset}': acc_plusless_1,
            f'mse-{name_subset}': mse
        }, indent=4)
    )


def save_json(dictionary, name_file):
    with open(name_file, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)


def load_json(name_file):
    data = None
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data


def save_binary(dictionary, name_file):
    with open(name_file, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_binary(name_file):
    data = None
    with open(name_file, 'rb') as fp:
        data = pickle.load(fp)
    return data