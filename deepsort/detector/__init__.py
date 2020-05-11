from .YOLOv3 import YOLOv3

CFG = "./deepsort/detector/YOLOv3/cfg/yolo_v3.cfg"
WEIGHT = "weights/yolov3/yolov3.weights"
CLASS_NAMES = "./deepsort/detector/YOLOv3/cfg/coco.names"

SCORE_THRESH = 0.5
NMS_THRESH = 0.4


def build_detector(use_cuda=True):
    return YOLOv3(CFG, WEIGHT, CLASS_NAMES, score_thresh=SCORE_THRESH, nms_thresh=NMS_THRESH, is_xywh=True)
