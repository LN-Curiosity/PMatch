#
# arm
#

import assembly

class ArmAsm(assembly.ImplAsm):
    def __init__(self, file_path, b_online, i_list):
        assembly.ImplAsm.__init__(self, file_path, b_online, i_list)
        self.__reg_name_list = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'sp', 'pc', 'lr']
        self.__call_ins_list = ['bl', 'blx']
        self.__jump_ins_list = ['beq', 'bne', 'bls', 'ble', 'blt', 'bge', 'bgt', 'bcc', 'bcs', 'bhi', 'bmi', 'bpl', 'bvc', 'bvs']
        self.__load_ins_list = ['ldr', 'str', 'ldrb', 'strb']
        self.__other_ins_list = ['nop']
        self.__str_opcode = ''
        self.__str_operand = ''
        self.__operand_list = []

    def __del__(self):
        assembly.ImplAsm.__del__(self)

    def translate(self):
        for str_asm in self.asm_list:
            if ';' in str_asm:
                str_asm = str_asm[: str_asm.find(';')].lower()
            else:
                str_asm = str_asm.lower().strip()
            if len(str_asm) == 0:
                self.out_list.append('')
                continue
            if ' ' in str_asm:
                self.__str_opcode = str_asm.split(' ', 1)[0].strip()
                self.__str_operand = str_asm.split(' ', 1)[1].strip()
            self.__operand_list.clear()
            if ', ' in self.__str_operand:
                self.__operand_list.append(self.__str_operand.split(',', 1)[0].strip())
                if ',' in self.__str_operand.split(',', 1)[1].strip() and not '[' in self.__str_operand.split(',', 1)[1].strip():
                    self.__operand_list.append(self.__str_operand.split(',', 1)[1].split(',', 1)[0].strip())
                    self.__operand_list.append(self.__str_operand.split(',', 1)[1].split(',', 1)[1].strip())
                else:
                    self.__operand_list.append(self.__str_operand.split(',', 1)[1].strip())
            else:
                self.__operand_list.append(self.__str_operand)

            # smull
            if self.__str_opcode == 'smull':
                self.__operand_list.clear()
                self.__operand_list.append(self.__str_operand.split(',', 1)[0].strip())
                self.__operand_list.append(self.__str_operand.split(',', 1)[1].split(',', 1)[0].strip())
                self.__operand_list.append(self.__str_operand.split(',', 1)[1].split(',', 1)[1].split(',', 1)[0].strip())
                self.__operand_list.append(self.__str_operand.split(',', 1)[1].split(',', 1)[1].split(',', 1)[1].strip())

            # print(str_asm)
            self.__str_operand = ''
            if str_asm in self.__other_ins_list:
                # print(str_asm + '\n')
                self.out_list.append(str_asm.strip(' '))
                continue
            for n_index in range(len(self.__operand_list)):
                # ldr & str
                if self.__str_opcode in self.__load_ins_list:
                    if not self.__operand_list[n_index] in self.__reg_name_list:
                        self.__str_operand += '<ADR>'
                        self.__str_operand += ' '
                    else:
                        self.__str_operand += '<REG>'
                        # self.__str_operand += self.__operand_list[n_index]
                        self.__str_operand += ' '
                    continue
                # replace register
                if self.__operand_list[n_index] in self.__reg_name_list:
                    self.__str_operand += '<REG>'
                    # self.__str_operand += self.__operand_list[n_index]
                    self.__str_operand += ' '
                    continue
                # replace memory access
                if self.__operand_list[n_index][-1] == ']':
                    self.__str_operand += '<MEM>'
                    self.__str_operand += ' '
                    continue
                # replace immediate operand
                if self.__operand_list[n_index][0] == '#' and self.__operand_list[n_index][1] >= '0' and self.__operand_list[n_index][1] <= '9':
                    self.__str_operand += '<NUM>'
                    self.__str_operand += ' '
                    continue

                # replace function name
                if self.__str_opcode in self.__call_ins_list:
                    self.__str_operand += '<FUN>'
                    self.__str_operand += ' '
                # replace jump loc
                elif self.__str_opcode in self.__jump_ins_list:
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
