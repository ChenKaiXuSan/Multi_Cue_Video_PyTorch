import sys 

try:
    from make_model import *
    from optical_flow import *
    from yolov8 import *
    from preprocess import *
except:
    sys.path.append('/workspace/Multi_Cue_Video_PyTorch/project/models')

    from make_model import *
    from optical_flow import *
    from yolov8 import *
    from preprocess import *