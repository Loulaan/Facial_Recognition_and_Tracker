import cv2
import torch
import numpy as np
from torchvision import transforms as trans


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def convert_bbox_2_img_area(img, bbox, toTensor=True):
    x1, y1, x2, y2 = [max(0, i) for i in bbox]
    # face_area = cv2.resize(img[bbox[1]:bbox[3], bbox[0]:bbox[2]], (112, 112), interpolation=cv2.INTER_NEAREST)
    face_area = cv2.resize(img[y1:y2, x1:x2], (112, 112), interpolation=cv2.INTER_NEAREST)
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
    if person is not 'Undefined':
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{person}_{round(float(score), 3)}', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return frame


def map_face_and_person(face, person):
    face_center = [(face[2] + face[0])/2, (face[3] + face[1])/2]
    # Is face center inside person's bbox, y-coord
    if person[1] < face_center[1] < person[3]:
        # Is face center inside person's bbox, x-coord
        if person[0] < face_center[0] < person[2]:
            return True
    return False


def is_proper_predictions(pred):
    return False if pred is None else True
