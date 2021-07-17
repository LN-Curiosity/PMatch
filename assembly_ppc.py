#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/17 0017 9:43
# @Author : zhe lang
# @Site : 
# @File : assembly_ppc.py
# @Software:

#
# ppc
#

import assembly

class PPCAsm(assembly.ImplAsm):
    def __init__(self, file_path, b_online, i_list):
        assembly.ImplAsm.__init__(self, file_path, b_online, i_list)
        self.__reg_name_list = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
                                'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27', 'r28', 'r29', 'r30', 'r31',
                                'lr', 'ctr', 'xer', 'msr', 'cr0', 'cr1', 'cr2', 'cr3', 'cr4', 'cr5', 'cr6', 'cr7', 'sp_0', 'rtoc']
        self.__call_ins_list = ['bl']
        self.__jump_ins_list = ['beq', 'bne', 'bls', 'ble', 'blt', 'bge', 'bgt']
        self.__jump_fmt1_ins_list = ['beq+', 'bne+', 'bls+', 'ble+', 'blt+', 'bge+', 'bgt+']
        self.__jump_fmt2_ins_list = ['beq-', 'bne-', 'bls-', 'ble-', 'blt-', 'bge-', 'bgt-']
        self.__load_ins_list = []
        self.__other_ins_list = ['nop']
        self.__str_opcode = ''
        self.__str_operand = ''
        self.__operand_list = []

    def __del__(self):
        assembly.ImplAsm.__del__(self)

    def translate(self):
        for str_asm in self.asm_list:
            if '#' in str_asm:
                str_asm = str_asm[: str_asm.find('#')].lower()
            else:
                str_asm = str_asm.lower().strip()
            if len(str_asm) == 0:
                self.out_list.append('')
                continue

            if ' ' in str_asm:
                self.__str_opcode = str_asm.split(' ', 1)[0].strip()
                if self.__str_opcode in self.__jump_fmt1_ins_list or self.__str_opcode in self.__jump_fmt2_ins_list:
                    self.__str_opcode = self.__str_opcode[0: 3]
                if self.__str_opcode[-1] == '.':
                    self.__str_opcode = self.__str_opcode[: len(self.__str_opcode)]
                self.__str_operand = str_asm.split(' ', 1)[1].strip()
            self.__operand_list.clear()
            if ', ' in self.__str_operand:
                self.__operand_list.append(self.__str_operand.split(',', 1)[0].strip())
                if ', ' in self.__str_operand.split(',', 1)[1].strip():
                    self.__operand_list.append(self.__str_operand.split(',', 1)[1].split(',', 1)[0].strip())
                    self.__operand_list.append(self.__str_operand.split(',', 1)[1].split(',', 1)[1].strip())
                else:
                    self.__operand_list.append(self.__str_operand.split(',', 1)[1].strip())
            else:
                self.__operand_list.append(self.__str_operand)

            # print(str_asm)
            self.__str_operand = ''
            if str_asm in self.__other_ins_list:
                self.out_list.append(str_asm.strip(' '))
                continue
            for n_index in range(len(self.__operand_list)):
                # replace register
                if self.__operand_list[n_index] in self.__reg_name_list:
                    self.__str_operand += '<REG>'
                    self.__str_operand += ' '
                    continue
                # replace memory access
                if '(' in self.__operand_list[n_index] and ')' in self.__operand_list[n_index]:
                    self.__str_operand += '<MEM>'
                    self.__str_operand += ' '
                    continue
                # replace immediate operand
                if self.__operand_list[n_index][0] == '-' or (self.__operand_list[n_index][0] >= '0' and self.__operand_list[n_index][0] <= '9'):
                    self.__str_operand += '<NUM>'
                    self.__str_operand += ' '
                    continue

                # replace function name
                if self.__str_opcode in self.__call_ins_list:
                    self.__str_operand += '<FUN>'
                    self.__str_operand += ' '
                # replace jump loc
                elif self.__str_opcode in self.__jump_ins_list and self.__operand_list[n_index][0: 4] == 'loc_':
                    self.__str_operand += '<ADR>'
                    self.__str_operand += ' '
                else:
                    self.__str_operand += '<STR>'
                    self.__str_operand += ' '
            # print(self.__str_opcode + ' ' + self.__str_operand.strip(' ') + '\n')
            self.out_list.append(self.__str_opcode + ' ' + self.__str_operand.strip(' '))
        if self.online:
            return self.out_list
        else:
            # print(self.out_list)
            return
