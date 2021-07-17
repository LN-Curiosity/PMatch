#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/12/14 0014 14:09
# @Author : zhe lang
# @Site : 
# @File : filter_x86.py
# @Software:


import os
import filter
import binary_differ


CALL_INS_FEATURE_SET = ['call']
# list[1] : >, >=, <, <=, <=, <, >=, >,
#           >, >=, <, <=, <=, <, >=, >,
#           =, !=, 0, !0, -, !-, e, e, o.
PATCH_CONSTRAINT_FEATURE_SET = [['cmp', 'test'],
                                ['ja', 'jae', 'jb', 'jbe', 'jna', 'jane', 'jnb', 'jnbe',
                                 'jg', 'jge', 'jl', 'jle', 'jng', 'jnge', 'jnl', 'jnle',
                                 'je', 'jne', 'jz', 'jnz', 'js', 'jns', 'jp', 'jpe', 'jpo'],
                                ['jmp'],
                                ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp']]


class X86Filter(filter.ImplFilter):
    def __init__(self, l_list, p_list, p_dict):
        filter.ImplFilter.__init__(self)
        self.__label_list = l_list
        self.__patch_list = p_list
        self.__patch_dict = p_dict


    def __del__(self):
        filter.ImplFilter.__del__(self)
        self.__label_list.clear()
        self.__patch_list.clear()
        self.__patch_dict.clear()


    def generate_feature_sequence(self, i_list):
        label_feature_list = []
        n_index = 0
        operand_list = []
        for asm_ins in i_list:
            asm_ins = asm_ins[: asm_ins.find(';')].lower().strip('\n')
            if ' ' in asm_ins:
                asm_opcode = asm_ins.split(' ', 1)[0].strip()
                asm_operand = asm_ins.split(' ', 1)[1].strip()
            else:
                asm_opcode = asm_ins
                asm_operand = ''
            if asm_opcode in CALL_INS_FEATURE_SET:
                label_feature_list.append(asm_ins)
                n_index += 1
                continue
            if asm_opcode in PATCH_CONSTRAINT_FEATURE_SET[0]:
                if ',' in asm_operand:
                    operand_list.append(asm_operand.split(',', 1)[0].strip())
                    operand_list.append(asm_operand.split(',', 1)[1].strip())
                else:
                    operand_list.append(asm_operand)
                asm_ins = asm_opcode
                for c_index in range(len(operand_list)):
                    if operand_list[c_index] in PATCH_CONSTRAINT_FEATURE_SET[3]:
                        asm_ins += ' <reg>'
                    elif operand_list[c_index][-1] == ']':
                        asm_ins += ' <mem>'
                    else:
                        asm_ins += ' '
                        asm_ins += operand_list[c_index]
                operand_list.clear()
                label_feature_list.append(asm_ins)
                n_index += 1
                continue
            if asm_opcode in PATCH_CONSTRAINT_FEATURE_SET[1] or asm_opcode in PATCH_CONSTRAINT_FEATURE_SET[2]:
                asm_ins = asm_opcode + ' <loc>'
                label_feature_list.append(asm_ins)
                n_index += 1
                continue
            n_index += 1
        return label_feature_list


    def main_handler(self):
        o_patch_list = filter.ImplFilter.merge_snippet(self, self.__patch_list, self.__patch_dict)
        # print(self.__patch_list)
        # print(o_patch_list)

        # label_feature_list = []
        # label_feature_list = self.generate_feature_sequence(self.__label_list)
        # self._ImplFilter__label_feature_list.extend(label_feature_list)
        # print('label feature:')
        # print(self._ImplFilter__label_feature_list)
        #
        # patch_list = []
        # for sub_patch_list in o_patch_list:
        #     patch_list = [self.__patch_dict[n_line].split('|', 1)[1] for n_line in sub_patch_list]
        #     label_feature_list = self.generate_feature_sequence(patch_list)
        #     self._ImplFilter__patch_feature_list.append(label_feature_list)
        # print('patch feature:')
        # print(self._ImplFilter__patch_feature_list)
        #
        # _, o_patch_list = filter.ImplFilter.filter_snippet(self, o_patch_list)
        # print('filter result list:')
        # print(o_patch_list)
        return o_patch_list


if __name__ == '__main__':
    v_list = []
    p_list = []

    dir_path = '/home/ubuntu/patch-O3'
    file_list = os.listdir(dir_path)
    path_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name)) and '.pat' in file_name]

    n_id = 0
    for pf_path in path_list:
        n_id += 1
        print('#%d' % n_id)
        vf_path = pf_path.replace('.pat', '.vul')
        lf_path = pf_path.replace('.pat', '.asm')
        with open(vf_path) as v_fp:
            v_list = list(v_fp)
        with open(pf_path) as p_fp:
            p_list = list(p_fp)
        with open(pf_path) as l_fp:
            l_list = list(l_fp)

        print(pf_path)
        print(vf_path)

        p_index = 0
        o_dict = {}
        o_list = []
        v_list, p_list = binary_differ.filter_block(v_list, p_list)
        o_dict, o_list = binary_differ.asm_diff(v_list, p_list)
        print('out:')
        for c_index in o_list:
            if c_index - p_index > 1:
                print()
            print(c_index, o_dict[c_index].strip('\n'))
            p_index = c_index

        test_filter = X86Filter(l_list, o_list, o_dict)
        test_filter.main_handler()
        print('')
