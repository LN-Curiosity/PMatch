#
import os

class ImplAsm:
    def __init__(self, file_path, b_online, i_list):
        self.asm_list = []
        self.out_list = []
        self.asm_fp = None
        self.out_fp = None
        self.asm_path = file_path
        self.out_path = None
        self.online = b_online

        if self.online:
            self.asm_list = i_list
        else:
            with open(self.asm_path, 'r') as self.asm_fp:
                self.asm_list = list(self.asm_fp)
            self.out_path = file_path.replace(".asm", ".ir")
            self.out_fp = open(self.out_path, 'w')
        # print(self.asm_list)

    def __del__(self):
        if self.out_fp and not self.online:
            self.out_fp.write('\n'.join(self.out_list))
            self.out_fp.close()

    def translate(self):
        pass
