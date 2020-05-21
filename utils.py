import cv2
from torchvision import transforms as trans
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def convert_bbox_2_img_area(img, bbox, toTensor=True):
    face_area = cv2.resize(img[bbox[1]:bbox[3], bbox[0]:bbox[2]], (112, 112), interpolation=cv2.INTER_NEAREST)
    normilized_face_area = test_transform(face_area)
    # TODO refactor converting to torch.Tensor
    if toTensor:
        return torch.tensor(normilized_face_area).to(device).type('torch.cuda.FloatTensor').unsqueeze(0)
    else:
        return np.array(normilized_face_area)


def draw_boxes_and_names(bbox, name, score, frame):
    x_min, y_min, x_max, y_max = bbox
    name = 'Undefined' if name == '-1' else name
    frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(255, 0, 0))
    frame = cv2.putText(frame, f'{name}_{round(float(score), 3)}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 1)
    return frame


def draw_results(frame, bbox, person, score):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'{person}_{round(float(score), 3)}', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return frame


def intersection_over_union(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def is_proper_predictions(pred):
    return False if pred is None else True
