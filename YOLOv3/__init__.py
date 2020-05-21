from YOLOv3.detector import YOLOv3


__all__ = ['YOLOv3', 'build_detector']


CFG = "YOLOv3/cfg/yolo_v3.cfg"
WEIGHT = "weights/YOLOv3/yolov3.weights"
CLASS_NAMES = "YOLOv3/cfg/coco.names"

SCORE_THRESH = 0.5
NMS_THRESH = 0.4


def build_detector():
    return YOLOv3(CFG, WEIGHT, CLASS_NAMES, score_thresh=SCORE_THRESH, nms_thresh=NMS_THRESH, is_xywh=True)