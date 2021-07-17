#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/4/16 0016 9:54
# @Author : zhe lang
# @Site : 
# @File : sif_eval.py
# @Software:


import os
import numpy as np
import time
import binary_differ
import filter_x86
import preprocess_assembly
import sif_model


X86_MAX_LENGTH = 150
EMBEDDING_SIZE = 128
SINGLE_BLOCK_THRESHOLD = 0.1 # O3
MULTIPLE_BLOCKS_THRESHOLD = 0.6 # all-opt
# MULTIPLE_BLOCKS_THRESHOLD = 0.45 # cross-opt

def get_binary_label(p, v):
    tgt_ver = os.path.basename(p)[os.path.basename(p).find('-') + 1: os.path.basename(p).find('-O')]
    tgt_sub_ver = tgt_ver[-1]
    if tgt_sub_ver >= 'a' and tgt_sub_ver <= 'z':
        tgt_ver = ord(tgt_sub_ver)
    else:
        tgt_ver = int(tgt_ver.split('.', 1)[0]) * 100 + \
                  int(tgt_ver.split('.', 1)[1].split('.', 1)[0]) * 10 + \
                  int(tgt_ver.split('.', 1)[1].split('.', 1)[1]) * 1

    vul_ver = os.path.basename(v)[os.path.basename(v).find('-') + 1: os.path.basename(v).find('-O')]
    vul_sub_ver = vul_ver[-1]
    if vul_sub_ver >= 'a' and vul_sub_ver <= 'z':
        vul_ver = ord(vul_sub_ver)
    else:
        vul_ver = int(vul_ver.split('.', 1)[0]) * 100 + \
                  int(vul_ver.split('.', 1)[1].split('.', 1)[0]) * 10 + \
                  int(vul_ver.split('.', 1)[1].split('.', 1)[1]) * 1

    if tgt_ver >= vul_ver:
        label = 'p'
    elif tgt_ver < vul_ver:
        label = 'v'
    return label


def rm_label_and_addr(cand_index_list, ins_dict):
    ins_list = []
    for bb_index_list in cand_index_list:
        ins_block = []
        for cur_index in bb_index_list:
            ins_block.append(ins_dict[cur_index].split('|', 1)[1].split('|', 1)[1].strip('\n'))
        ins_list.append(ins_block)
    cand_num = len(ins_list)
    return cand_num, ins_list


def rm_addr(ins_list, l_ins_list):
    ori_len = len(ins_list)
    for bb in l_ins_list:
        ins_block = []
        for ins in bb:
            ins_block.append(ins.split('|', 1)[1].strip('\n'))
        ins_list.append(ins_block)
    return len(ins_list) - ori_len, ins_list


def get_ir_ins(ins_list, w_dict):
    ir_list = []
    asm_factory = preprocess_assembly.AsmFactory()
    for bb in ins_list:
        cur_bb = []
        cur_bb = asm_factory.get_type('X86', None, True, bb).translate()
        # cur_bb = asm_factory.get_type('ARM', None, True, bb).translate()
        # cur_bb = asm_factory.get_type('PPC', None, True, bb).translate()
        cur_bb = oov_handler(cur_bb, w_dict)
        ir_list.append(cur_bb)
    return ir_list


def oov_handler(ins_list, w_dict):
    m_ins_list = []
    for ins in ins_list:
        if not ins in w_dict.keys():
            is_change = False
            print('oov')
            for key in w_dict.keys():
                if ins.split(' ', 1)[0] == key.split(' ', 1)[0]:
                    ins = key
                    is_change = True
                    break
            if is_change:
                m_ins_list.append(ins)
            else:
                m_ins_list.append('UNK')
        else:
            m_ins_list.append(ins)
    return m_ins_list


def calculate(y_vec, x_vec):
    y_vec = np.squeeze(y_vec)
    x_vec = np.squeeze(x_vec)
    return round(np.dot(y_vec, x_vec) / (np.linalg.norm(y_vec) * (np.linalg.norm(x_vec))), 4)


def gen_scores_m(c_m, c_num, l_m, l_num):
    scores_m = np.zeros((l_num, c_num))
    for i in range(l_num):
        for j in range(c_num):
            scores_m[i, j] = calculate(l_m[i, :], c_m[j, :])
    return scores_m


def gen_blocks_w(bb_list, l_num):
    sum_len = 0
    w_list = []
    for n_id in range(l_num):
        sum_len += len(bb_list[n_id])
    for n_id in range(l_num):
        w_list.append(len(bb_list[n_id]) / sum_len)
    return w_list


def is_single_block_predict(p, l, index_list, sim_id, ins_dict):
    if p == l:
        if p == 'p':
            return 'tp'
        elif p == 'v':
            return 'tn'
    else:
        if p == 'p':
            return 'fp'
        elif p == 'v':
            return 'fn'


def is_multiple_blocks_predict(p, l, index_list, sim_dict, ins_dict, l_num):
    if p == l:
        if p == 'p':
            return 'tp'
        elif p == 'v':
            return 'tn'
    else:
        if p == 'p':
            return 'fp'
        elif p == 'v':
            return 'fn'


def gen_roc_data(l_pre, l_lab):

    if not len(l_pre) == len(l_lab):
        print('error.')
        return

    with open('./prediction_result.txt', 'a') as p_fp:
        for pre in l_pre:
            p_fp.write(str(pre)+'\n')
    with open('./label_result.txt', 'a') as l_fp:
        for lab in l_lab:
            l_fp.write(str(lab)+'\n')


def evaluate(dir_path):
    file_list = os.listdir(dir_path)
    vul_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name)) and '.vul' in file_name]
    path_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name)) and '.pat' in file_name]

    ins_emb_dict = sif_model.get_ins_embedding(sif_model.INS_DICT_PATH, sif_model.EMB_LIST_PATH)
    ins_weight_dict = sif_model.get_ins_weight(sif_model.FREQ_DICT_PATH)

    all_sent_emb_m = np.zeros((1033, len(ins_emb_dict[' '])))
    # all_sent_emb_m = np.zeros((966, len(ins_emb_dict[' '])))
    # all_sent_emb_m = np.zeros((1019, len(ins_emb_dict[' '])))
    all_sent_emb_m = sif_model.get_sent_emb_matrix('./sentence', ins_emb_dict, ins_weight_dict)
    print(np.shape(all_sent_emb_m))

    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0
    tgt_num = len(path_list)
    num_count = 0
    aver_time = 0
    pre_list = []
    lab_list = []
    for p_path in path_list:
        s_t = time.time()

        v_path = p_path.replace('.pat', '.vul')
        if not v_path in vul_list:
            for cur_path in vul_list:
                if v_path[v_path.find('CVE'): v_path.find('.vul')] == cur_path[cur_path.find('CVE'): cur_path.find('.vul')]:
                    v_path = cur_path
                    break
        l_path = v_path.replace('.vul', '.asm')

        # 获取标签
        label = get_binary_label(p_path, v_path)
        prediction = ''
        num_count += 1
        print('%d / %d' % (num_count, tgt_num))
        print(p_path)
        print(v_path)
        print('* the label : %s' % label)


        # 获取漏洞版本和目标版本的函数指令序列
        with open(v_path) as v_fp:
            v_list = list(v_fp)
        with open(p_path) as p_fp:
            p_list = list(p_fp)
        with open(l_path) as l_fp:
            l_list = list(l_fp)

        # 获取补丁代码片段
        label_ins_list = []
        label_ins_list = binary_differ.get_block_ins(l_list)
        is_single_fileter = True
        for bb in label_ins_list:
            if len(bb) < 2:
                is_single_fileter = False
                break

        # 1.进行二进制diff
        o_dict = {}
        o_list = []
        o_dict, o_list = binary_differ.asm_diff(v_list, p_list)
        test_filter = filter_x86.X86Filter(l_list, o_list, o_dict)
        candidate_index_list = test_filter.main_handler()
        # 对diff结果中尺寸过大或过小的代码块进行移除
        candidate_index_list = [candidate for candidate in candidate_index_list if len(candidate) <= X86_MAX_LENGTH]
        if is_single_fileter:
            candidate_index_list = [candidate for candidate in candidate_index_list if len(candidate) > 1]
        print('candidate number : %d' % len(candidate_index_list))

        input_ins_list = []
        candidate_num, input_ins_list = rm_label_and_addr(candidate_index_list, o_dict)
        label_num, input_ins_list = rm_addr(input_ins_list, label_ins_list)

        # 规则1：如果diff后无候选预测目标为漏洞版本
        if candidate_num == 0:
            prediction = 'v'
            if prediction == label:
                tn_count += 1
            else:
                fn_count += 1
            e_t = time.time()
            print('time : %f' % (e_t - s_t))
            aver_time += e_t - s_t
            print('* the prediction : %s\n' % prediction)
            continue

        input_ir_list = []
        input_ir_list = get_ir_ins(input_ins_list, ins_weight_dict)

        # 2.生成所有代码片段的embedding
        retn_sent_m = sif_model.get_sent_emb(input_ir_list, ins_emb_dict, ins_weight_dict, all_sent_emb_m)
        candidate_m = retn_sent_m[:candidate_num, :]
        label_m = retn_sent_m[candidate_num:, :]

        # 3.根据补丁代码片段的基本块数量进行预测
        if label_num == 1:
            scores_list = []
            for i in range(candidate_num):
                cur_score = calculate(candidate_m[i, :], label_m[0, :])
                scores_list.append(cur_score)
            sim_index = scores_list.index(max(scores_list))
            # 规则2：加权得分大于阈值则预测为补丁版本否则为漏洞版本
            if scores_list[sim_index] > MULTIPLE_BLOCKS_THRESHOLD:
                prediction = 'p'
            else:
                prediction = 'v'
            res = is_single_block_predict(prediction, label, candidate_index_list, sim_index, o_dict)
            if res == 'tp':
                tp_count += 1
            elif res == 'tn':
                tn_count += 1
            elif res == 'fp':
                fp_count += 1
            elif res == 'fn':
                fn_count += 1

            if label == 'p':
                lab_list.append(1.0)
            elif label == 'v':
                lab_list.append(0.0)
            pre_list.append(scores_list[sim_index])
        elif label_num > 1:
            sim_index = 0
            sim_dict = {}
            average_score = 0
            weight_list = gen_blocks_w(label_ins_list, label_num)
            scores_m = gen_scores_m(candidate_m, candidate_num, label_m, label_num)
            # 寻找best match block
            for i in range(label_num):
                for j in range(candidate_num):
                    if float(scores_m[i, j]) == 1.0 and not j in sim_dict.values():
                        sim_dict[i] = j
                        average_score += weight_list[i] * 1.0
                        break
            # 寻找sim match block
            for i in range(label_num):
                m_count = 0
                if not i in sim_dict.keys():
                    while True:
                        sim_index = list(scores_m[i, :]).index(max(scores_m[i, :]))
                        if not sim_index in sim_dict.values():
                            if scores_m[i, sim_index] > SINGLE_BLOCK_THRESHOLD:
                                sim_dict[i] = sim_index
                                average_score += weight_list[i] * scores_m[i, sim_index]
                                break
                            else:
                                average_score += weight_list[i] * 0
                                break
                        else:
                            scores_m[i, sim_index] = 0
                            m_count += 1
                        if m_count > candidate_num:
                            break
            # 规则2：加权得分大于阈值则预测为补丁版本否则为漏洞版本
            if average_score > MULTIPLE_BLOCKS_THRESHOLD:
                prediction = 'p'
            else:
                prediction = 'v'
            res = is_multiple_blocks_predict(prediction, label, candidate_index_list, sim_dict, o_dict, label_num)
            if res == 'tp':
                tp_count += 1
            elif res == 'tn':
                tn_count += 1
            elif res == 'fp':
                fp_count += 1
            elif res == 'fn':
                fn_count += 1

            if label == 'p':
                lab_list.append(1.0)
            elif label == 'v':
                lab_list.append(0.0)
            pre_list.append(average_score)

        e_t = time.time()
        print('time : %f' % (e_t - s_t))
        aver_time += e_t - s_t
        print('* the prediction : %s\n' % prediction)

    print((tp_count + tn_count) / tgt_num * 100)
    print(aver_time / tgt_num)
    print('tp_count :%d' % tp_count)
    print('tn_count :%d' % tn_count)
    print('fp_count :%d' % fp_count)
    print('fn_count :%d' % fn_count)
    # gen_roc_data(pre_list, lab_list)

    return


if __name__ == '__main__':
    evaluate('./dataset2/O3')
