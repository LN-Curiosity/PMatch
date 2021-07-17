#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/11/26 0026 10:29
# @Author : zhe lang
# @Site :
# @File : binary_differ.py
# @Software:


import os
import difflib
import ssdeep
import preprocess_assembly


def get_block_ins(ins_list):
    block_list = []
    pre_id = 0

    for cur_id in range(len(ins_list)):
        if ins_list[cur_id] == '\n':
            block_list.append(ins_list[pre_id: cur_id])
            pre_id = cur_id + 1
    block_list.append(ins_list[pre_id: ])

    block_list = [bb for bb in block_list if len(bb) > 0]
    return block_list


def get_block_hash(bb_list):
    block_string = []
    block_hash = []
    ins_list = []

    asm_factory = preprocess_assembly.AsmFactory()
    for cur_block in bb_list:
        for str_ins in cur_block:
            ins_list.append(str_ins.split('|', 1)[1].split('|', 1)[1])

        ir_list = []
        ir_list = asm_factory.get_type('X86', None, True, ins_list).translate()
        in_hash = ''
        for c_id in range(len(ir_list)):
            # if ir_list[c_id][0: 4] == 'call':
            #     in_hash += ins_list[c_id].strip('\n')
            # else:
            #     in_hash += ir_list[c_id]
            in_hash += ir_list[c_id]
            in_hash += ' '
        ins_list.clear()

        # print(in_hash)
        # print(ssdeep.hash(in_hash))
        block_string.append(in_hash)
        block_hash.append(ssdeep.hash(in_hash))

    return block_string, block_hash


def match_similar_block(l_hash, r_hash):
    l_remove = []
    r_remove = []

    for i in range(len(r_hash)):
        for j in range(len(l_hash)):
            if ssdeep.compare(r_hash[i], l_hash[j]) == 100 and not j in l_remove:
                l_remove.append(j)
                r_remove.append(i)
                print(j, i)
                break
    if len(l_remove) != len(r_remove):
        print('the best match error')

    return l_remove, r_remove


def remove_similar_block(bb_list, rm_list):
    ins_list = []

    bb_id = 0
    for bb in bb_list:
        if bb_id in rm_list:
            bb_id += 1
            continue
        bb_id += 1
        for ins in bb:
            ins_list.append(ins + '\n')
        ins_list.append('')

    return ins_list


def filter_block(l_list, r_list):
    # 1.obtain the ins list of blocks
    l_block_list = get_block_ins(l_list)
    r_block_list = get_block_ins(r_list)
    # 2.obtain the hash list of blocks
    l_input_list, l_hash_list = get_block_hash(l_block_list)
    r_input_list, r_hash_list = get_block_hash(r_block_list)
    # 3.remove the best match blocks
    l_remove_list, r_remove_list = match_similar_block(l_hash_list, r_hash_list)
    l_ins_list = remove_similar_block(l_block_list, l_remove_list)
    r_ins_list = remove_similar_block(r_block_list, r_remove_list)

    return l_ins_list, r_ins_list


def asm_diff(l_list, r_list):
    l_dict = {}
    for n_line in range(1, len(l_list) + 1):
        l_dict[n_line] = l_list[n_line - 1]
    p_dict = {}
    for n_line in range(1, len(r_list) + 1):
        p_dict[n_line] = r_list[n_line - 1]

    # l_list = [str_asm.split('|', 1)[1] for str_asm in l_list]
    # r_list = [str_asm.split('|', 1)[1] for str_asm in r_list]
    l_in = []
    for str_asm in l_list:
        if str_asm == '' or str_asm == '\n':
            l_in.append('')
        else:
            l_in.append(str_asm.split('|', 1)[1].split('|', 1)[1])
    r_in = []
    for str_asm in r_list:
        if str_asm == '' or str_asm == '\n':
            r_in.append('')
        else:
            r_in.append(str_asm.split('|', 1)[1].split('|', 1)[1])

    p_list = []

    for left, right, changed in difflib._mdiff(fromlines=l_in, tolines=r_in):
        l_no, l_asm = left
        r_no, r_asm = right

        if not changed:
            pass
            # print(l_no, l_dict[l_no].strip('\n'), r_no, p_dict[r_no].strip('\n'))
        else:
            if l_no:
                l_beg = l_asm.strip(b'\x00'.decode()).strip(b'\x01'.decode()).strip('\n')[0]
                if l_beg == '+':
                    l_asm = '+'
                elif l_beg == '-':
                    l_asm = '-'
                else:
                    l_asm = ''
                l_asm += l_dict[l_no]
            else:
                l_asm = ''
            if r_no:
                r_beg = r_asm.strip(b'\x00'.decode()).strip(b'\x01'.decode()).strip('\n')[0]
                if r_beg == '+':
                    r_asm = '+'
                    p_list.append(r_no)
                elif r_beg == '-':
                    r_asm = '-'
                else:
                    r_asm = ''
                r_asm += p_dict[r_no]
            else:
                r_asm = ''
            # print(l_no, l_asm.strip('\n'), r_no, r_asm.strip('\n'))
    return p_dict, p_list


if __name__ == '__main__':
    # v_list = []
    # p_list = []

    # dir_path = '/home/ubuntu/patch-O3'
    # file_list = os.listdir(dir_path)
    # path_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name)) and '.pat' in file_name]
    #
    # n_id = 0
    # for pf_path in path_list:
    #     n_id += 1
    #     print('#%d' % n_id)
    #     vf_path = pf_path.replace('.pat', '.vul')
    #     with open(vf_path) as v_fp:
    #         v_list = list(v_fp)
    #     with open(pf_path) as p_fp:
    #         p_list = list(p_fp)
    #
    #     print(pf_path)
    #     print(vf_path)
    #
    #     p_index = 0
    #     o_dict = {}
    #     o_list = []
    #     v_list, p_list = filter_block(v_list, p_list)
    #     o_dict, o_list = asm_diff(v_list, p_list)
    #     print('out:')
    #     for c_index in o_list:
    #         if c_index - p_index > 1:
    #             print()
    #         print(c_index, o_dict[c_index].strip('\n'))
    #         p_index = c_index


    vf_path = '/home/security-patch-assembly-datasets/openssl/X86/gcc-5.4.0/O3/openssl-1.0.1h-O3-g-CVE-2014-0195-dtls1_reassemble_fragment.vul'
    pf_path = '/home/security-patch-assembly-datasets/openssl/X86/gcc-5.4.0/O3/openssl-1.0.1h-O3-g-CVE-2014-0195-dtls1_reassemble_fragment.pat'
    with open(vf_path) as v_fp:
        v_list = list(v_fp)
    with open(pf_path) as p_fp:
        p_list = list(p_fp)

    print(pf_path)
    print(vf_path)

    p_index = 0
    o_dict = {}
    o_list = []
    v_list, p_list = filter_block(v_list, p_list)
    o_dict, o_list = asm_diff(v_list, p_list)
    print('out:')
    for c_index in o_list:
        if c_index - p_index > 1:
            print()
        print(c_index, o_dict[c_index].strip('\n'))
        p_index = c_index

    import filter_x86
    test_filter = filter_x86.X86Filter([], o_list, o_dict)
    candidate_index_list = test_filter.main_handler()
    candidate_index_list = [candidate for candidate in candidate_index_list if len(candidate) > 1]
    print()
    print(candidate_index_list)
    print('len : %d' % (len(candidate_index_list)))
    print()
    for bb in candidate_index_list:
        for index in bb:
            print(o_dict[index].strip('\n').split('|', 1)[1].split('|', 1)[1])
        print()
