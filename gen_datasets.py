#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/1/5 0005 14:58
# @Author : zhe lang
# @Site : 
# @File : gen_datasets.py
# @Software:


import os
import random


def gen_pair(file_path, type):
    ins_list = []
    all_block_list = []
    valid_block_list = []
    pair_list = []

    with open(file_path, 'r') as fp1:
        ins_list = list(fp1)
        ins_list = [ins.strip('\n') for ins in ins_list]
        print(len(ins_list))
        p_index = 0
        for n_index in range(0, len(ins_list)):
            if ins_list[n_index] == '':
                if p_index == 0:
                    all_block_list.append(ins_list[p_index: n_index])
                else:
                    all_block_list.append(ins_list[p_index+1: n_index])
                p_index = n_index
        all_block_list.append(ins_list[p_index + 1: n_index + 1])
    print(len(all_block_list))

    for block in all_block_list:
        if len(block) > 5 and len(block) < 76:
            valid_block_list.append(block)
    print(len(valid_block_list))

    file_path = file_path.replace(".ir", ".json")
    fp_2 = open(file_path, 'w')
    for c_index in range(0, len(valid_block_list)):
        p_index = random.randint(0, len(valid_block_list) - 1)
        while c_index == p_index or valid_block_list[c_index] == valid_block_list[p_index]:
            p_index = random.randint(0, len(valid_block_list) - 1)
        str_elem_0 = '{"y":0, "x1":%s, "x2":%s}' % (valid_block_list[c_index], valid_block_list[p_index])
        # print(str_elem_0)
        fp_2.write(str_elem_0 + '\n')
        if type == 0:
            str_elem_1 = '{"y":1, "x1":%s, "x2":%s}' % (valid_block_list[c_index], valid_block_list[c_index][random.randint(1, 1+1*len(valid_block_list[c_index])//5):])
        elif type == 1:
            str_elem_1 = '{"y":1, "x1":%s, "x2":%s}' % (valid_block_list[c_index], valid_block_list[c_index][: random.randint(4*len(valid_block_list[c_index])//5, len(valid_block_list[c_index])-1)])
        # print(str_elem_1)
        fp_2.write(str_elem_1 + '\n')
    fp_2.close()


def pair_shuffle(file_list):
    json_list = []
    for file_path in file_list:
        with open(file_path, 'r') as fp:
            json_list.extend(list(fp))
    random.shuffle(json_list)
    json_list = [str_json.strip('\n') for str_json in json_list]
    with open('./openssl_train.json', 'w') as fp:
        fp.write('\n'.join(json_list))
    print(len(json_list))


#
# gen_pair is original, get_sd_pair is modified.
#
def algorithm_for_lcs(l_seq, r_seq):
    matrix = [[0 for i in range(len(r_seq) + 1)] for j in range(len(l_seq) + 1)]
    max_length = 0

    for i in range(len(l_seq)):
        for j in range(len(r_seq)):
            if l_seq[i] == r_seq[j]:
                matrix[i + 1][j + 1] = matrix[i][j] + 1
                if matrix[i + 1][j + 1] > max_length:
                    max_length = matrix[i + 1][j + 1]

    return max_length


def is_similar_pair(i_file1, i_file2):
    with open(i_file1, 'r') as fp1:
        i_list1 = list(fp1)
    with open(i_file2, 'r') as fp2:
        i_list2 = list(fp2)
    if len(i_list1) == 0 or len(i_list2) == 0:
        return None
    if len(i_list1) > 20 or len(i_list2) > 20:
        return None

    if i_list1 != i_list2:
        i_list1 = [elem.strip('\n') for elem in i_list1]
        i_list2 = [elem.strip('\n') for elem in i_list2]
        str_elem = '{"y":1, "x1":%s, "x2":%s}' % (str(i_list1), str(i_list2))
        return str_elem
    else:
        return None


def is_dissimilar_pair(i_file, i_list):
    with open(i_file, 'r') as fp1:
        i_list1 = list(fp1)
    if len(i_list1) == 0 or len(i_list1) > 20:
        return None

    l_count = 0
    while True:
        i_file2 = i_list[random.randint(0, len(i_list) - 1)]
        with open(i_file2, 'r') as fp:
            i_list2 = list(fp)
        if len(i_list2) == 0:
            continue
        opt_list1 = [ins_opt.split(' ', 1)[0] for ins_opt in i_list1]
        opt_list2 = [ins_opt.split(' ', 1)[0] for ins_opt in i_list2]
        if 1/3 * len(i_list1) > algorithm_for_lcs(opt_list1, opt_list2):
            i_list1 = [elem.strip('\n') for elem in i_list1]
            i_list2 = [elem.strip('\n') for elem in i_list2]
            str_elem = '{"y":0, "x1":%s, "x2":%s}' % (str(i_list1), str(i_list2))
            return str_elem
        l_count += 1
        # if l_count > 3:
        #     return None


def get_sd_pair(dir_path):
    file_list = os.listdir(dir_path)
    O0_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name)) and '.O0' in file_name]
    O1_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name)) and '.O1' in file_name]
    O2_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name)) and '.O2' in file_name]
    O3_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name)) and '.O3' in file_name]
    pair_list = []

    for file_name in O0_list:
        file_name1 = file_name.replace('O0', 'O1')
        file_name2 = file_name.replace('O0', 'O2')
        file_name3 = file_name.replace('O0', 'O3')
        str_retn = ''

        # 预判断
        if not os.path.isfile(os.path.join(dir_path, file_name)) or not os.path.isfile(os.path.join(dir_path, file_name1)) or not os.path.isfile(os.path.join(dir_path, file_name2)) or not os.path.isfile(os.path.join(dir_path, file_name3)):
            continue
        if os.path.getsize(file_name) == 0 or os.path.getsize(file_name1) == 0 or os.path.getsize(file_name2) == 0 or os.path.getsize(file_name3) == 0:
            continue
        with open(file_name, 'r') as fp1:
            t_list1 = list(fp1)
        with open(file_name3, 'r') as fp2:
            t_list2 = list(fp2)
        if 1/3 * len(t_list1) > len(t_list2):
            continue

        if file_name in O0_list and file_name1 in O1_list:
            str_retn = is_similar_pair(file_name, file_name1)
            if str_retn:
                pair_list.append(str_retn)
                # print(file_name, file_name1)
                # print(str_retn)

        if file_name in O0_list and file_name2 in O2_list:
            str_retn = is_similar_pair(file_name, file_name2)
            if str_retn:
                pair_list.append(str_retn)
                # print(file_name, file_name2)
                # print(str_retn)

        if file_name in O0_list and file_name3 in O3_list:
            str_retn = is_similar_pair(file_name, file_name3)
            if str_retn:
                pair_list.append(str_retn)
                # print(file_name, file_name3)
                # print(str_retn)

        if file_name1 in O1_list and file_name2 in O2_list:
            str_retn = is_similar_pair(file_name1, file_name2)
            if str_retn:
                pair_list.append(str_retn)
                # print(file_name1, file_name2)
                # print(str_retn)

        if file_name1 in O1_list and file_name3 in O3_list:
            str_retn = is_similar_pair(file_name1, file_name3)
            if str_retn:
                pair_list.append(str_retn)
                # print(file_name1, file_name3)
                # print(str_retn)

        if file_name2 in O2_list and file_name3 in O3_list:
            str_retn = is_similar_pair(file_name2, file_name3)
            if str_retn:
                pair_list.append(str_retn)
                # print(file_name2, file_name3)
                # print(str_retn)
    sim_len = len(pair_list)
    print(sim_len)

    # a dissimilar pair
    all_list = []
    all_list.append(O0_list)
    all_list.append(O1_list)
    all_list.append(O2_list)
    all_list.append(O3_list)

    for cur_list in all_list:
        for file_name in cur_list:
            str_retn = is_dissimilar_pair(file_name, O0_list)
            if str_retn:
                pair_list.append(str_retn)
    print(len(pair_list))

    random.shuffle(pair_list)
    # pair_list = [str_json.strip('\n') for str_json in pair_list]
    all_len = len(pair_list)
    print(all_len - sim_len)
    for str_pair in pair_list[:100]:
        print(str_pair)
    with open('./openssl_train.json', 'w') as fp:
        fp.write('\n'.join(pair_list))


if __name__ == '__main__':
    # path_list = ['./openssl-1.0.1a-O0-g.ir', './openssl-1.0.1u-O0-g.ir']
    # gen_pair(path_list[0], 0)
    # gen_pair(path_list[1], 1)

    # pair_shuffle(['/home/patch2vec/sentences/datasets_x86/openssl-1.0.1a-O0-g.json', '/home/patch2vec/sentences/datasets_x86/openssl-1.0.1u-O0-g.json'])
    get_sd_pair('/home/pair')

