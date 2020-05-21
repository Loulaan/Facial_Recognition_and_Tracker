from deepsort.deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


REID_CKPT = "weights/deepsort/ckpt.t7"
MAX_DIST = 0.2
MIN_CONFIDENCE = 0.3
NMS_MAX_OVERLAP = 0.5
MAX_IOU_DISTANCE = 0.7
MAX_AGE = 70
N_INIT = 3
NN_BUDGET = 100


def build_tracker():
    return DeepSort(REID_CKPT, max_dist=MAX_DIST, min_confidence=MIN_CONFIDENCE,
                    nms_max_overlap=NMS_MAX_OVERLAP, max_iou_distance=MAX_IOU_DISTANCE,
                    max_age=MAX_AGE, n_init=N_INIT, nn_budget=NN_BUDGET)