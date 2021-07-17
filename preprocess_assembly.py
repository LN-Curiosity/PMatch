import assembly_x86
import assembly_arm
import assembly_mips
import assembly_ppc
import os


class AsmFactory:
    def get_type(self, asm_type, input_path, b_online=False, i_list=[]):
        if asm_type == 'X86':
            return assembly_x86.X86Asm(input_path, b_online, i_list)
        elif asm_type == 'ARM':
            return assembly_arm.ArmAsm(input_path, b_online, i_list)
        elif asm_type == 'PPC':
            return assembly_ppc.PPCAsm(input_path, b_online, i_list)
        elif asm_type == 'MIPS':
            return assembly_mips.MipsAsm(input_path, b_online, i_list)
        else:
            return

def generate_batch_ir(dir_path):
    file_list = os.listdir(dir_path)
    path_list = [os.path.join(dir_path, file_name) for file_name in file_list if os.path.isfile(os.path.join(dir_path, file_name))]
    asm_factory = AsmFactory()
    for file_path in path_list:
        print(file_path)
        if os.path.getsize(file_path) > 0:
            # asm_factory.get_type('ARM', file_path).translate()
            asm_factory.get_type('PPC', file_path).translate()
        else:
            print('rm')
            os.remove(file_path)


if __name__ == '__main__':
    # input_list = ['sub esp, 0Ch\n', 'push [esp+148h+s]; s\n', 'call dtls1_max_handshake_message_len\n', 'add esp, 10h\n', 'cmp eax, [esp+13Ch+frag_len]\n', 'jb loc_80CD5DB\n', 'nop\n', 'jmp short err\n']
    # asm_factory = AsmFactory()
    # print(asm_factory.get_type('X86', None, True, input_list).translate())

    # asm_factory = AsmFactory()
    # asm_factory.get_type('X86', './openssl-1.0.1a-O0-g.asm').translate()

    # generate_batch_ir('../output-binutils')
    generate_batch_ir('/home/ubuntu/output')
