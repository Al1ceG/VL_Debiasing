import re
import numpy as np

log_path = "/home/s2142414/VL_Debiasing/LIC_details.out"
def processe_section(name, text):
    accs = [float(a) for a in re.findall(r"Accuracy for iteration \d+: ([\d.]+)", text)]
    if not accs:
        return
    
    # the first 10 are always HUMAN, the next 10 are MODEL
    human_accs = accs[:10]
    model_accs = accs[10:20]
    