#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/10/26 0026 10:39
# @Author : zhe lang
# @Site : 
# @File : data_preprocess.py
# @Software:


import collections
import os
import pathlib

# x86 assembly
# VOCABULARY_SIZE = 1055
# arm assembly
# VOCABULARY_SIZE = 424
# ppc assembly
VOCABULARY_SIZE = 242

def read_asm_ins(dir_path):
    data_list = []

    dp = pathlib.Path(dir_path)
    if not dp.exists():
        print('%s directory does not exist.' % (dir_path))
        return

    file_list = os.listdir(dir_path)
    # 遍历所有预处理后的汇编文件
    path_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name))]
    for file_path in path_list:
        with open(file_path) as fp:
            assembly_list = list(fp)
        data_list.extend(assembly_list)
        data_list.extend(' ')
    data_list = [data.strip('\n') for data in data_list]
    # data_list = [data.strip('\n') for data in data_list if data != '\n']
    print(len(data_list))
    # print(data_list)
    return data_list


def build_asm_dataset(asm_list):
    assembly_count = [['UNK', -1]]
    # 根据汇编语句的出现频率进行排序
    assembly_count.extend(collections.Counter(asm_list).most_common())
    # 根据字典当前长度生成汇编编码字典 key-汇编语句名称 value-汇编编码
    assembly_dictionary = {}
    for assembly_sentence, _ in assembly_count:
        assembly_dictionary[assembly_sentence] = len(assembly_dictionary)
    print(len(assembly_dictionary))
    print(assembly_dictionary)

    # 对数据进行编码
    assembly_encoding_list = []
    unk_count = 0
    for assembly_sentence in asm_list:
        if assembly_sentence in assembly_dictionary:
            assembly_encoding = assembly_dictionary[assembly_sentence]
        else:
            assembly_encoding = 0
            unk_count = unk_count + 1
        assembly_encoding_list.append(assembly_encoding)
    assembly_count[0][1] = unk_count

    # 生成逆汇编编码字典 key-汇编编码 value-汇编语句名称
    reverse_dictionary = dict(zip(assembly_dictionary.values(), assembly_dictionary.keys()))
    return assembly_encoding_list, assembly_count, assembly_dictionary, reverse_dictionary


if __name__ == '__main__':
    asm_ins_list = read_asm_ins('./words/datasets_ppc')
    asm_encoding_list, asm_count_list, dictionary, reverse_dictionary = build_asm_dataset(asm_ins_list)
    print(asm_encoding_list[: 200])
