#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/4/12 0012 10:20
# @Author : zhe lang
# @Site : 
# @File : sif_model.py
# @Software:


import os
import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
import data_preprocess


SIF_HPARA_A = 1e-3
SIF_HPARA_RMPC = 1
INS_DICT_PATH = './ins_dict_x86.json'
FREQ_DICT_PATH = './freq_dict_x86.json'
EMB_LIST_PATH = './cbow_embeddings_x86.npy'


# 分别生成指令编号和指令频率字典
def get_ins_dict(corpus_path):
    ins_list = data_preprocess.read_asm_ins(corpus_path)
    _, ins_count, ins_dict, _ = data_preprocess.build_asm_dataset(ins_list)

    sum_count = len(ins_list)
    freq_dict = {}
    for i_count in ins_count:
        freq_dict[i_count[0]] = i_count[1] / sum_count

    with open(INS_DICT_PATH, 'w') as i_fp:
        i_fp.write(json.dumps(ins_dict))
    with open(FREQ_DICT_PATH, 'w') as f_fp:
        f_fp.write(json.dumps(freq_dict))

    return ins_dict, freq_dict


# 生成指令嵌入字典
def get_ins_embedding(id_file, emb_file):
    ins_dict = {}
    with open(id_file, 'r') as i_fp:
        ins_dict = json.load(i_fp)
    print(ins_dict)

    emb_dict = {}
    ins_emb_m = np.load(emb_file)
    for key in ins_dict.keys():
        emb_dict[key] = ins_emb_m[ins_dict[key]]

    return emb_dict


# 生成指令权重字典
def get_ins_weight(freq_file):
    freq_dict = {}
    with open(freq_file, 'r') as f_fp:
        freq_dict = json.load(f_fp)
    print(freq_dict)

    weight_dict = {}
    for key in freq_dict.keys():
        weight_dict[key] = round(SIF_HPARA_A / (SIF_HPARA_A + freq_dict[key]), 6)

    print(weight_dict)
    return weight_dict


def compute_pc(X, npc=1):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


# 生成SIF句子嵌入
def get_sent_emb(sent_list, emb_dict, weight_dict, all_sent_m):
    # 计算指令嵌入的加权平均和
    sent_emb_m = np.zeros((len(sent_list), len(emb_dict[' '])))
    for i in range(len(sent_list)):
        weight_list = []
        emb_list = []
        for ins in sent_list[i]:
            # oov
            ins = ins.strip()
            weight_list.append(weight_dict[ins])
            emb_list.append(emb_dict[ins])

        weight_m = np.mat(weight_list)
        emb_m = np.mat(emb_list)
        sent_emb_m[i, :] = weight_m.dot(emb_m) / len(sent_list[i])
    # print(sent_emb_m) #

    cur_sent_emb_m = np.vstack((sent_emb_m, all_sent_m))
    # 为每个句子嵌入移除主成分
    if SIF_HPARA_RMPC > 0:
        cur_sent_emb_m = remove_pc(cur_sent_emb_m, SIF_HPARA_RMPC)

    # print(sent_emb_m) #
    return cur_sent_emb_m[:len(sent_list), :]


# 生成所有句子的嵌入
def get_all_sent_emb(sent_list, emb_dict, weight_dict):
    sent_emb_m = np.zeros((len(sent_list), len(emb_dict[' '])))
    for i in range(len(sent_list)):
        weight_list = []
        emb_list = []
        for ins in sent_list[i]:
            # oov
            ins = ins.strip()
            weight_list.append(weight_dict[ins])
            emb_list.append(emb_dict[ins])

        weight_m = np.mat(weight_list)
        emb_m = np.mat(emb_list)
        sent_emb_m[i, :] = weight_m.dot(emb_m) / len(sent_list[i])

    return sent_emb_m


# 预生成所有句子的指令嵌入加权平均矩阵
def get_sent_emb_matrix(dir_path, emb_dict, weight_dict):
    file_list = os.listdir(dir_path)
    file_list = [os.path.join(dir_path, file_name) for file_name in file_list
                 if os.path.getsize(os.path.join(dir_path, file_name)) > 0]

    sent_list = []
    for file_path in file_list:
        with open(file_path, 'r') as fp:
            ins_list = list(fp)
        sent_list.append(ins_list)
    print(len(sent_list))

    sent_emb = get_all_sent_emb(sent_list, emb_dict, weight_dict)
    return sent_emb


if __name__ == '__main__':
    get_ins_dict('./words/datasets_ppc')

    # ins_emb_dict = get_ins_embedding(INS_DICT_PATH, EMB_LIST_PATH)
    # print(len(ins_emb_dict))
    #
    # ins_weight_dict = get_ins_weight(FREQ_DICT_PATH)
    # print(len(ins_weight_dict))
    #
    # # test_sent_list = [['call <FUN>'], ['call eax']]
    # # sent_emb_list = get_sent_emb(test_sent_list, ins_emb_dict, ins_weight_dict)
    # # print(sent_emb_list)
    #
    # get_sent_emb_matrix('/home/pair', ins_emb_dict, ins_weight_dict)
