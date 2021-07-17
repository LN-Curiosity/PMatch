#
# x86
#

import assembly

class X86Asm(assembly.ImplAsm):
    def __init__(self, file_path, b_online, i_list):
        assembly.ImplAsm.__init__(self, file_path, b_online, i_list)
        self.__reg_name_list = ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp', 'eip']
        self.__call_ins_list = ['call', 'callfi', 'callni']
        self.__jump_ins_list = ['ja', 'jae', 'jb', 'jbe', 'jc', 'jcxz', 'jecxz', 'jrcxz', 'je', 'jg', 'jge', 'jl',
                                'jle', 'jna', 'jnae', 'jnb', 'jnbe', 'jnc', 'jne', 'jng', 'jnge', 'jnl', 'jnle', 'jno',
                                'jnp', 'jns', 'jnz', 'jo', 'jp', 'jpe', 'jpo', 'js', 'jz', 'jmp', 'jmpfi', 'jmpni', 'jmpshort']
        self.__load_ins_list = ['lea', 'leavew', 'leave', 'leaved', 'leaveq', 'lds', 'les', 'lfs', 'lgs', 'lss']
        self.__other_ins_list = ['nop', 'popf', 'pushf', 'retn', 'cdq', 'fldz', 'leave']
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
            if ',' in self.__str_operand:
                self.__operand_list.append(self.__str_operand.split(',', 1)[0].strip())
                self.__operand_list.append(self.__str_operand.split(',', 1)[1].strip())
            else:
                self.__operand_list.append(self.__str_operand)

            # print(str_asm)
            self.__str_operand = ''
            if str_asm in self.__other_ins_list:
                # print(str_asm + '\n')
                self.out_list.append(str_asm.strip(' '))
                continue
            for n_index in range(len(self.__operand_list)):
                # replace register
                if self.__operand_list[n_index] in self.__reg_name_list:
                    self.__str_operand += self.__operand_list[n_index]
                    self.__str_operand += ' '
                    continue
                # replace memory access
                if self.__operand_list[n_index][-1] == ']':
                    if self.__str_opcode in self.__load_ins_list:
                        self.__str_operand += '<ADR>'
                        self.__str_operand += ' '
                        continue
                    else:
                        self.__str_operand += '<MEM>'
                        self.__str_operand += ' '
                        continue
                # replace immediate operand
                if self.__operand_list[n_index][0] >= '0' and self.__operand_list[n_index][0] <= '9':
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
                # replace other string
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
