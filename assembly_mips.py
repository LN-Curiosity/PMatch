#
# mips
#

import assembly

class MipsAsm(assembly.ImplAsm):
    def __init__(self, file_path):
        assembly.ImplAsm.__init__(self, file_path)

    def __del__(self):
        assembly.ImplAsm.__del__(self)

    def translate(self):
        print('mips')
        return
