import csv
import glob
import json
import os
import sys

import pandas as pd
import pianoplayer.core

import utils
from utils import save_json


def load_json(name_file):
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data


fixed_scores = [
    "2364046.musicxml",
    "5026044.musicxml",
    "5026820.musicxml",
    "5028687.musicxml",
    "5047635.musicxml",
    "5074210.musicxml",
    "5112058.musicxml",
    "5187218.musicxml",
    "5452009.musicxml",
    "117422.musicxml",
    "583656.musicxml",
    "4180411.musicxml",
    "5526285.musicxml",
    "5541896.musicxml",
    "5639517.musicxml",
    "6165659.musicxml",
    "6266056.musicxml",
    "6303314.musicxml",
    "6331689.musicxml",
    "180705.musicxml",
    "4518566.musicxml",
    "4867475.musicxml",
    "4885820.musicxml",
    "5459604.musicxml",
    "5477551.musicxml",
    "5581440.musicxml",
    "5715607.musicxml",
    "6114842.musicxml",
    "6175077.musicxml",
    "6224476.musicxml"
]


def load_xmls():
    grade, name, xmls = [], [], [],

    for path, g in load_json("henleXmus/index_good.json").items():
        grade.append(g)
        name.append(path)
        xmls.append(f"henleXmus/score_files/{path}.musicxml")
    return zip(grade, name, xmls)


import multiprocessing as mp


def run_loop(args):
    r_h, l_h, xml = args

    try:
        pianoplayer.core.run_annotate(xml, outputfile=r_h, n_measures=10000, depth=9,
                             right_only=True, quiet=False)
        pianoplayer.core.run_annotate(xml, outputfile=l_h, n_measures=10000, depth=9,
                             left_only=True, quiet=False)
    except Exception as e:
        print(e)
    print(f"done {xml}")


def do_fingering():
    num_workers = 6
    print("num_workers", num_workers)
    args = []
    for grade, path, xml in load_xmls():
        r_h_out = '/'.join(["Fingers/henle", os.path.basename(xml[:-4]) + '_rh.txt'])
        l_h_out = '/'.join(["Fingers/henle", os.path.basename(xml[:-4]) + '_lh.txt'])

        if f'{path}.musicxml' in fixed_scores or not os.path.exists(r_h_out) or not os.path.exists(l_h_out):
            print(xml)
            args.append((r_h_out, l_h_out, xml))
        else:
            # print("YA HA SIDO COMPUTADO")
            continue
    p = mp.Pool(processes=num_workers)
    print(f"Faltan: {len(args)}")

    p.map(run_loop, args)


def finger2index(f):
    if f > 0:
        index = int(f) + 4
    elif f < 0:
        index = int(f) - 5
    else:  # == 0
        index = -1000
    return index


def velocity_piece(path, xml):
    errors = 0
    total = 0
    path_alias = 'henle'
    print(path)
    r_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_rh.txt'])
    l_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_lh.txt'])

    intermediate_rep = []
    for path_txt, hand in zip([r_h_cost, l_h_cost], ["right_", "left_"]):
        time_series = []
        with open(path_txt) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for l in read_tsv:
                if int(l[7]) != 0:
                    time_series.append((round(float(l[1]), 2), int(l[7]), abs(float(l[8]))))
                else:
                    errors += 1
                total += 1
        time_series = time_series[:-9]
        intermediate_rep.extend(time_series)

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = intermediate_rep[idx][2]
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = intermediate_rep[j][2]
            j += 1
        idx = j
        # print(t)
        matrix.append(t)
    return matrix, onsets, errors/total


def rep_velocity():
    rep = {}
    for idx, (grade, path, xml) in enumerate(load_xmls()):
        matrix, _, _ = velocity_piece(path, xml)
        rep[path] = {
            'grade': int(grade['henle']) - 1,
            'matrix': matrix
        }
        print(idx)

    save_json(rep, os.path.join('representations', 'henle', 'rep_velocity.json'))
    json2pickle("representations/henle/rep_velocity.json")



def prob_piece(path, xml):
    path_alias = 'henle_Fingering'

    print(path)
    PIG_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-9]) + '.txt'])

    time_series = []
    with open(PIG_cost) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for l in list(read_tsv)[1:]:
            if int(l[7]) != 0:
                time_series.append((round(float(l[1]), 2), int(l[7]), abs(abs(float(l[8])))))
    time_series = time_series[:-3]

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(time_series, key=(lambda a: a[0]))]

    onsets = []
    idx = 0
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = intermediate_rep[idx][2]
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = intermediate_rep[j][2]
            j += 1
        idx = j
        # print(t)
        matrix.append(t)

    return matrix, onsets


def rep_prob():
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = prob_piece(path, xml)

        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }
    save_json(rep, os.path.join('representations', 'henle', 'rep_nakamura.json'))
    json2pickle("representations/henle/rep_nakamura.json")


def notes_piece(path, xml):
    path_alias = 'henle'
    print(path)
    r_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_rh.txt'])
    l_h_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_lh.txt'])

    intermediate_rep = []
    for path_txt, hand in zip([r_h_cost, l_h_cost], ["right_", "left_"]):
        time_series = []
        with open(path_txt) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for l in read_tsv:
                if int(l[7]) != 0:
                    # (onset, note)
                    time_series.append((round(float(l[1]), 2), int(l[3]) - 21))
        intermediate_rep.extend(time_series)

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0.0] * 88
        index = intermediate_rep[idx][1]
        if index >= 88:
            index = 87
            print(f"ERROR: index {index}")
        t[index] = 1.0
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = intermediate_rep[j][1]
            if index >= 88:
                index = 87
                print(f"ERROR: index {index}")
            # print(index)
            t[index] = 1.0
            j += 1
        idx = j
        # print(t)
        matrix.append(t)
    return matrix, onsets


def rep_notes():
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = notes_piece(path, xml)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', 'henle', 'rep_note.json'))
    json2pickle("representations/henle/rep_note.json")



def compute_velocity_errors():
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    rep = {}
    malas = []
    for grade, path, xml in load_xmls():
        _, _, errors = velocity_piece(path, xml)
        rep[path] = {
            'grade': int(grade['henle']) - 1,
            'errors': errors
        }
        if errors >= 0.25:
            malas.append(path)
    print(len(malas), malas)
    pd.DataFrame(rep).T.groupby(['grade'])['errors'].describe().to_csv("errors_percentage.csv")




def json2pickle(path):
    print(path)
    data = utils.load_json(path)
    utils.save_binary(data, path[:-5] + ".pickle")


if __name__ == "__main__":
    # if not os.path.exists('Fingers/henle'):
    #     os.makedirs('Fingers/henle')
    # do_fingering()
    # rep_velocity()
    rep_prob()
    rep_notes()

    # json2pickle("representations/mikrokosmos/rep_velocity.json")
    #
    # compute_velocity_errors()
    # json2pickle('representations/mikrokosmos/rep_note.json')
    # json2pickle('representations/mikrokosmos/rep_nakamura.json')
