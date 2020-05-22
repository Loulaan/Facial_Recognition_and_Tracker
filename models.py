from pathlib import Path

import torch
import cv2
import numpy as np

from YOLOv3 import build_detector
from deepsort import build_tracker
from retinaface import RetinaFace
from facenet import InceptionResnetV1
from utils import convert_bbox_2_img_area, device


torch.set_grad_enabled(False)


class FaceDetector:
    def __init__(self):
        self.face_detector = RetinaFace(gpu_id=0, network='resnet50')  # Backbone: resnet50 or mobilenet
        self.threshold = 0.8

    def filter_detections(self, faces):
        filtered_faces = []
        if faces is not None:
            for face in faces:
                bbox, landmarks, score = face
                if score < self.threshold:
                    continue
                bbox = bbox.astype(np.int) + [-5, -5, 5, 5]  # broadcast bboxes with 5 px
                filtered_faces.append(bbox)
        return filtered_faces

    def __call__(self, frame):
        return self.filter_detections(self.face_detector.detect(frame))


class FaceIdentificator:

    def __init__(self, update_facebank=False, dataset=None):
        """
        :param update_facebank: True for generating new facebank, False for loading the existing
        :param dataset: Path to data for computing, saving and loading facebank
        """
        self.update_facebank = update_facebank
        self.facenet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
        self.face_detector = FaceDetector()
        self.threshold = 1.1
        self.dataset = dataset
        self.saved_embeddings = {}
        self.prepare_facebank()  # Create facebank

    def prepare_facebank(self):
        if self.update_facebank:
            for path in Path(self.dataset).iterdir():
                if path.is_file():
                    continue
                embs = []
                for file in path.iterdir():
                    if not file.is_file() or file.parts[-1][-4:] != ".jpg":
                        continue
                    img = cv2.imread(str(file))
                    faces = self.face_detector(img)
                    if faces is not None:
                        box = faces[0]
                        embs.append(self.calculate_embeddings(img, box, True))
                if len(embs) == 0:
                    continue

                self.saved_embeddings[path.parts[-1]] = torch.cat(embs).mean(0)  # 1x1x512

            torch.save(self.saved_embeddings, f"{str(self.dataset)}/facebank.pth")
            print(f'embeddings are calculated and saved at {str(self.dataset)}')
        else:
            self.saved_embeddings = torch.load(f"{str(self.dataset)}/facebank.pth")
            print(f'embeddings are loaded from {str(self.dataset)}')

    def calculate_embeddings(self, img, box, for_facebank=False):
        face_area = convert_bbox_2_img_area(img, box)
        return self.facenet(face_area).unsqueeze(0) if for_facebank else self.facenet(face_area)

    def __call__(self, frame):
        """
        Face identification
        :param frame: Captured frame from video stream
        :return: Areas with faces, idxs in facebank, confidence (l2 distance)
        """
        faces = self.face_detector(frame)

        if len(faces) == 0:
            return None, None, None

        target_embeddings = torch.cat(list(self.saved_embeddings.values()))
        names = list(self.saved_embeddings.keys())
        names.append('-1')  # for undefined persons

        source_embeddings = []
        for bbox in faces:
            source_embeddings.append(self.calculate_embeddings(frame, bbox))

        # Compute distance and get identification information
        diff = torch.cat(source_embeddings).unsqueeze(-1) - \
               target_embeddings.transpose(1, 0).unsqueeze(0)  # #detected_faces x 512 x #persons_in_facebank
        dist = torch.sum(torch.pow(diff, 2), dim=1)  # #detected_faces x #persons_in_facebank
        min_score, min_idx = torch.min(dist, dim=1)
        min_idx[min_score > self.threshold] = -1  # filtering preds
        min_score[min_score > self.threshold] = 10
        return faces, min_idx.to('cpu').numpy(), min_score.to('cpu').numpy()


class Tracker:
    def __init__(self):
        self.detector = build_detector()  # YOLOv3
        self.tracker = build_tracker()  # DeepSort
        self.threshold = 0.7

    def __call__(self, frame):
        bbox_xywh, cls_conf, cls_ids = self.detector(frame)

        if len(bbox_xywh) == 0:
            return None, None

        # Filter YOLOv3 predictions
        # select person class
        mask = cls_ids == 0
        bbox_xywh = bbox_xywh[mask]
        cls_conf = cls_conf[mask]

        # select high conf bboxes
        mask = cls_conf > self.threshold
        bbox_xywh = bbox_xywh[mask]
        cls_conf = cls_conf[mask]

        outputs = self.tracker.update(bbox_xywh, cls_conf, frame)
        if len(outputs) == 0:
            return None, None
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        return bbox_xyxy, identities
