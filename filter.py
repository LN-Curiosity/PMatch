#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/14 0014 14:08
# @Author : zhe lang
# @Site : 
# @File : filter.py
# @Software:


class ImplFilter:
    def __init__(self):
        self.__label_feature_list = []
        self.__patch_feature_list = []

    def __del__(self):
        self.__label_feature_list.clear()
        self.__patch_feature_list.clear()

    def merge_snippet(self, i_list, i_dict):
        t_list = []
        t_list.extend(i_list)
        for c_index in range(0, len(i_list) - 1):
            if i_list[c_index + 1] - i_list[c_index] == 1:
                continue
            if i_list[c_index + 1] - i_list[c_index] == 2:
                t_list.insert(t_list.index(i_list[c_index]) + 1, i_list[c_index] + 1)

        s_index = 0
        o_list = []
        for c_index in range(0, len(t_list)):
            if c_index == len(t_list) - 1:
                o_list.append(t_list[s_index:c_index + 1])
                break
            if t_list[c_index + 1] - t_list[c_index] > 1:
                o_list.append(t_list[s_index:c_index + 1])
                s_index = c_index + 1

        # divide into basic blocks
        bb_list = []
        for cur_block in o_list:
            pre_id = 0
            for cur_id in range(len(cur_block)):
                if i_dict[cur_block[cur_id]].strip() == '':
                    bb_list.append(cur_block[pre_id: cur_id])
                    pre_id = cur_id + 1
            bb_list.append(cur_block[pre_id:])

        bb_list = [bb for bb in bb_list if len(bb) > 0]
        return bb_list

    def __algorithm_for_lcs(self, l_seq, r_seq):
        matrix = [[0 for i in range(len(r_seq) + 1)] for j in range(len(l_seq) + 1)]
        max_length = 0
        p = 0
        for i in range(len(l_seq)):
            for j in range(len(r_seq)):
                if l_seq[i] == r_seq[j]:
                    matrix[i + 1][j + 1] = matrix[i][j] + 1
                    if matrix[i + 1][j + 1] > max_length:
                        max_length = matrix[i + 1][j + 1]
                        p = i + 1
        return l_seq[p - max_length:p], max_length

    def filter_snippet(self, i_list):
        sim_id = 0
        cur_id = 0
        max_length = 0
        cur_length = 0
        o_list = []
        if len(self.__label_feature_list) > 0:
            for n_index in range(0, len(self.__patch_feature_list)):
                if len(self.__patch_feature_list[n_index]) > 0:
                    cur_list, cur_length = self.__algorithm_for_lcs(self.__label_feature_list, self.__patch_feature_list[n_index])
                    # if cur_length > float(1 / 3 * len(self.__label_feature_list)) and cur_length < 2 * len(self.__label_feature_list):
                    # if cur_length > float(1/4 * len(self.__label_feature_list)) and cur_length < 2 * len(self.__label_feature_list):
                    if cur_length > float(2/5 * len(self.__label_feature_list)) and cur_length < 2 * len(self.__label_feature_list):
                        o_list.append(i_list[n_index])
                    if cur_length > max_length:
                        max_length = cur_length
                        sim_id = cur_id
                    cur_id += 1
            o_list = [c_list for c_list in o_list if len(c_list) > 1]
            return sim_id, o_list
        else:
            o_list.extend(i_list)
            # for n_index in range(0, len(self.__patch_feature_list)):
            #     if len(self.__patch_feature_list[n_index]) == 0:
            #         o_list.append(i_list[n_index])
            o_list = [c_list for c_list in o_list if len(c_list) > 1]
            return None, o_list

    def main_handler(self):
        pass
