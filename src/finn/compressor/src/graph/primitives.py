from ctypes import c_uint64, c_uint32
from .nodes import BlackboxInput, BlackboxOutput, Blackbox, LUT, Constant
from .nodes import BlackboxInputVec, BlackboxOutputVec

class LUT2(LUT):
    @classmethod
    def fromPred(self, predO2, desired_name = "lut2"):
        res = 0
        for i in range(32):
            inputs = [bool(i & (1 << shmt)) for shmt in range(2)]
            res = res | (int(predO2(*inputs)) << i)
            res = res & 0xF
        init_str = f"""4'h{"{:_x}".format(c_uint32(res).value)}"""
        return LUT2(init_str, desired_name)
    
    def __init__(self, init_code: str, desired_name):
        in_ports = [BlackboxInput(f"I{el}") for el in range(2)]
        out_ports = [BlackboxOutput("O")]
        super().__init__("LUT2", init_code, in_ports, out_ports,
                         desired_name=desired_name, size=0.5)

class LUT5(LUT):
    @classmethod
    def fromPred(self, predO5, desired_name = "lut5"):
        res = 0
        for i in range(32):
            inputs = [bool(i & (1 << shmt)) for shmt in range(5)]
            res = res | (int(predO5(*inputs)) << i)
        init_str = f"""32'h{"{:_x}".format(c_uint32(res).value)}"""
        return LUT5(init_str, desired_name)
    
    def __init__(self, init_code: str, desired_name):
        in_ports = [BlackboxInput(f"I{el}") for el in range(5)]
        out_ports = [BlackboxOutput("O")]
        super().__init__("LUT5", init_code, in_ports, out_ports,
                         desired_name=desired_name, size=0.5)

class LUT6(LUT):
    @classmethod
    def fromPred(self, predO6, desired_name = "lut6"):
        res = 0
        for i in range(64):
            inputs = [bool(i & (1 << shmt)) for shmt in range(6)]
            res = res | (int(predO6(*inputs)) << i)
        init_str = f"""64'h{"{:_x}".format(c_uint64(res).value)}"""
        return LUT6(init_str, desired_name)
    
    def __init__(self, init_code: str, desired_name):
        in_ports = [BlackboxInput(f"I{el}") for el in range(6)]
        out_ports = [BlackboxOutput("O")]
        super().__init__("LUT6", init_code, in_ports, out_ports,
                         desired_name=desired_name, size=1)

def split_lut_from_pred(predO5, predO6):
    res = 0
    for i in range(32, 64):
        inputs = [bool(i & (1 << shmt)) for shmt in range(6)]
        res = res | (int(predO5(*inputs)) << (i-32)) | (int(predO6(*inputs)) << (i))
        init_str = f"""64'h{"{:_x}".format(c_uint64(res).value)}"""
    return init_str

class LUT6_2(LUT):
    @classmethod
    def fromPred(self, predO5, predO6, desired_name = "lut6_2"):
        return LUT6_2(split_lut_from_pred(predO5, predO6), desired_name)

    def __init__(self, init_code: str, desired_name):
        in_ports = [BlackboxInput(f"I{el}") for el in range(6)] 
        out_ports = [BlackboxOutput("O6"), BlackboxOutput("O5")]
        super().__init__("LUT6_2", init_code, in_ports, out_ports,
                         desired_name=desired_name, size=1)
        Constant("1'b1").connect_to(self.I5)

class LUT6CY(LUT):
    @classmethod
    def fromPred(self, predO51, predO52, desired_name = "lut6cy"):
        return LUT6CY(split_lut_from_pred(predO51, predO52), desired_name)

    def __init__(self, init_code: str, desired_name):
        in_ports = [BlackboxInput(f"I{el}") for el in range(5)]
        out_ports = [BlackboxOutput(f"O5{el+1}") for el in range(2)]
        out_ports.append(BlackboxOutput("PROP"))
        super().__init__("LUT6CY", init_code, in_ports, out_ports,
                         desired_name=desired_name, size=1)

class LOOKAHEAD8(Blackbox):
    def __init__(self):
        c_in_ports_str = ["CIN", "CYA", "CYB", "CYC", "CYD", "CYE", "CYF", "CYG", "CYH"]
        p_in_ports_str = ["PROPA", "PROPB", "PROPC", "PROPD", "PROPE", "PROPF", "PROPG",
                          "PROPH"]
        out_ports_str = ["COUTB", "COUTD", "COUTF", "COUTH"]
        
        self.c_in_ports = [BlackboxInput(el) for el in c_in_ports_str]
        self.p_in_ports = [BlackboxInput(el) for el in p_in_ports_str]
        out_ports = [BlackboxOutput(el) for el in out_ports_str]
        super().__init__("LOOKAHEAD8", self.c_in_ports + self.p_in_ports, out_ports,
                         {"LOOKB" : "\"TRUE\"", "LOOKD" : "\"TRUE\"",
                          "LOOKF" : "\"TRUE\"", "LOOKH" : "\"TRUE\""})

class CARRY4(Blackbox):
    def __init__(self):
        in_ports = [BlackboxInputVec("DI", 4), BlackboxInputVec("S", 4),
                    BlackboxInput("CI")]
        out_ports = [BlackboxOutputVec("O", 4), BlackboxOutputVec("CO", 4)]
        super().__init__("CARRY4", in_ports, out_ports, {})