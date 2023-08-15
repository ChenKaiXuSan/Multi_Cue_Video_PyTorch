import sys 

try:
    from data_loader import *
except:
    sys.path.append('/workspace/Multi_Cue_Video_PyTorch/project/dataloader')
    from data_loader import *