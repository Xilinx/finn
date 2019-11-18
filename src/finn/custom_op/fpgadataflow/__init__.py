from abc import abstractmethod
from finn.custom_op import CustomOp


class HLSCustomOp(CustomOp):
    def __init__(self):
        super().__init__()


    @abstractmethod
    def code_generation(self, node):
        pass
