import os
from datetime import datetime
from pathlib import Path
from collections import Counter

import torch
import cv2
import numpy as np

from face_detection import RetinaFace
from arcface.Learner import face_learner
from facenet import InceptionResnetV1
from torchvision import transforms as trans

# TODO clean deepsort folder
from deepsort.deep_sort import build_tracker
from deepsort.detector import build_detector


from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO move all models into their classes in models.py

working_dir = os.getcwd()

torch.set_grad_enabled(False)
color = (0, 255, 0)

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# TODO refactor Pipeline (make separate class for face processing)
class FaceDetector:

    def __init__(self, cap_name='0', update_facebank=False):

        self.working_dir = os.getcwd()
        self.active_face_verificator = 'facenet'
        self.update_facebank = update_facebank
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.face_detector = RetinaFace(gpu_id=0, network='resnet50')  # resnet50 or mobilenet

        self.show_frames = True
        self.threshold = 1.54
        self.dataset = f"{self.working_dir}/data/dataset2"

        self.cap = \
            cv2.VideoCapture(f'{self.dataset}/{cap_name}') if cap_name != '0' else cv2.VideoCapture(0)
        self.saved_embeddings = {}
        self.embeddings_calculator = None

        self.arcface = face_learner(True, 'ir_50')  # ir_50 or mobilefacenet
        self.arcface.model.load_state_dict(torch.load(f'{self.working_dir}/weights/arcface/model_ir_se50.pth'))
        self.arcface.threshold = self.threshold
        self.arcface.model.eval()

        self.facenet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()

        self.prepare_facebank()  # Create database imitation

    @staticmethod
    def draw_boxes_and_names(bbox, name, score, frame):
        x_min, y_min, x_max, y_max = bbox
        name = 'Undefined' if name == '-1' else name
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(255, 0, 0))
        frame = cv2.putText(frame, f'{name}_{round(float(score), 3)}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color, 1)
        return frame

    @staticmethod
    def convert_2_gray(img):
        """
        Convert image to grayscale duplicated every channel (3x112x112)
        :param img: face area (3x112x112)
        :return: grayscale image with shape 3x112x112
        """

        face_area_gray = np.zeros_like(img)
        face_area = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face_area_gray[:, :, 0] = face_area
        face_area_gray[:, :, 1] = face_area
        face_area_gray[:, :, 2] = face_area
        return face_area_gray

    def convert_bbox_2_img_area(self, img, bbox, toTensor=True):
        face_area = cv2.resize(img[bbox[1]:bbox[3], bbox[0]:bbox[2]], (112, 112), interpolation=cv2.INTER_NEAREST)
        # face_area = self.convert_2_gray(face_area)
        normilized_face_area = test_transform(face_area)
        # TODO refactor converting to torch.Tensor
        if toTensor:
            return torch.tensor(normilized_face_area).to(self.device).type('torch.cuda.FloatTensor').unsqueeze(0)
        else:
            return np.array(normilized_face_area)

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
                    faces = self.detect_faces(img)
                    if faces is not None:
                        box, landmarks, score = faces[0]
                        embs.append(self.calculate_embeddings(img, box, True))

                if len(embs) == 0:
                    continue

                self.saved_embeddings[path.parts[-1]] = torch.cat(embs).mean(0)  # 1x1x512

            torch.save(self.saved_embeddings, f"{str(self.dataset)}/facebank.pth")
            print('embeddings are calculated')
        else:
            self.saved_embeddings = torch.load(f"{str(self.dataset)}/facebank.pth")
            print('embeddings are loaded')

    def calculate_embeddings(self, img, box, for_facebank=False):
        face_area = self.convert_bbox_2_img_area(img, box)
        if self.active_face_verificator == 'arcface':
            return self.arcface.model(face_area).unsqueeze(0) if for_facebank else self.arcface.model(face_area)
        if self.active_face_verificator == 'facenet':
            return self.facenet(face_area).unsqueeze(0) if for_facebank else self.facenet(face_area)

    def verify_faces(self, frame):
        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return frame

        target_embeddings = torch.cat(list(self.saved_embeddings.values()))
        names = list(self.saved_embeddings.keys())
        names.append('-1')  # for undefined persons

        source_embeddings = []
        for bbox in faces:
            source_embeddings.append(self.calculate_embeddings(frame, bbox))

        diff = torch.cat(source_embeddings).unsqueeze(-1) - \
               target_embeddings.transpose(1, 0).unsqueeze(0)  # #detected_faces x 512 x #persons_in_facebank
        dist = torch.sum(torch.pow(diff, 2), dim=1)  # #detected_faces x #persons_in_facebank (dataset2 = 1x5)
        min_score, min_idx = torch.min(dist, dim=1)
        min_idx[min_score > self.threshold] = -1  # filtering preds
        min_score[min_score > self.threshold] = 10
        return faces, min_idx.to('cpu').numpy(), min_score.to('cpu').numpy()

    def detect_faces(self, frame):
        faces = self.face_detector.detect(frame)
        filtered_face = []
        if faces is not None:
            for face in faces:
                bbox, landmarks, score = face
                if score < 0.8:
                    continue
                bbox = bbox.astype(np.int) + [-5, -5, 5, 5]  # broadcast bboxes with 5 px
                filtered_face.append(bbox)

        return filtered_face

    def start(self):
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(f'{self.dataset}/processed_video3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # frame = np.rot90(frame, -1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.verify_faces(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if frame is not None:
                    cv2.imshow("Verified", frame)
                    # frame = np.rot90(frame, 1)
                    # out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()


class Tracker:
    def __init__(self, show_frames=False):
        self.deepsort = build_tracker(use_cuda=True)
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        self.show_frames = show_frames
        self.detector = build_detector(True)

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)

    def infer(self, frame):
        bbox_xywh, cls_conf, cls_ids = self.detector(frame)
        if len(bbox_xywh) == 0:
            return

        # select person class
        mask = cls_ids == 0

        bbox_xywh = bbox_xywh[mask]
        cls_conf = cls_conf[mask]

        # select high conf bboxes
        mask = cls_conf > 0.7
        bbox_xywh = bbox_xywh[mask]
        cls_conf = cls_conf[mask]

        outputs = self.deepsort.update(bbox_xywh, cls_conf, frame)
        if len(outputs) == 0:
            return
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]

        if self.show_frames:
            for i, box in enumerate(bbox_xyxy):
                x1, y1, x2, y2 = box
                # box text and bar
                id = int(identities[i]) if identities is not None else 0
                color = self.compute_color_for_labels(id)
                label = '{}{:d}'.format("", id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return bbox_xyxy, identities


class Pipeline:
    def __init__(self):

        self.face_identificator = FaceDetector('data/dataset2/video3.mp4')
        self.tracker = Tracker()
        self.names_identity = list(self.face_identificator.saved_embeddings.keys())
        self.saved_scores = dict(zip(self.names_identity, [10 for name in self.names_identity]))
        self.mapped_tracks = {}
        self.update_mapped_tracks = True

    def map_faces_with_persons(self, bboxes, faces, idx, identities):
        verified_bboxes = []
        verified_identities = []

        for i, bbox in enumerate(bboxes):
            for j, face in enumerate(faces):
                # TODO fix this
                IoU = intersection_over_union(face, bbox)
                if IoU != 0:
                    # print('IoU: ', IoU)
                    verified_bboxes.append(bbox)
                    verified_identities.append([identities[i], idx[j]])
                    break
        # print('Len bboxes', len(verified_bboxes), ' | Len identities', len(verified_identities))
        return verified_bboxes, verified_identities

    def compare_scores(self, identities, new_scores):
        compared_identities = []
        # print(new_scores)
        for i, identities in enumerate(identities):
            new_track_name = self.names_identity[identities[1]]
            old_track_name = self.mapped_tracks[identities[0]][0]
            try:
                # TODO find out what is the wrong case
                score = new_scores[i]
            except IndexError:
                print("Error")
            if self.mapped_tracks[identities[0]][1] > score:
                # Если расстояние до старого имени больше, обновляем расстояние и имя
                self.mapped_tracks[identities[0]][1] = score
                self.mapped_tracks[identities[0]][0] = new_track_name
                compared_identities.append(new_track_name)
            else:
                # В противном случае считаем, что идентификация было ложно положительной
                compared_identities.append(old_track_name)
        return compared_identities

    def draw_results(self, frame, bbox, person, score):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{person}_{round(float(score), 3)}', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        return frame


    def infer(self):

        cap = cv2.VideoCapture('data/dataset2/video3.mp4')
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # out = cv2.VideoWriter(f'data/dataset2/processed_video_test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    faces, idx, verif_scores = self.face_identificator.verify_faces(frame)
                except ValueError:
                    faces = None

                try:
                    bboxes, identities = self.tracker.infer(frame)
                except TypeError:
                    continue

                if len(bboxes) == 0:
                    continue

                # TODO пофиксить снижение устойчивости треков
                # # show tracked bboxes
                # for i, bbox in enumerate(bboxes):
                #     frame = self.draw_results(frame, bbox, identities[i], '1')
                # cv2.imshow('Frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # continue

                for ident in identities:
                    if ident not in list(self.mapped_tracks.keys()):
                        self.mapped_tracks[ident] = ['Undefined', 10.0]

                if faces is not None:
                    mapped_bboxes, identities = self.map_faces_with_persons(bboxes, faces, idx, identities)
                    # print('Len bboxes', len(bboxes), ' | Len identities', len(identities))
                    # if self.update_mapped_tracks:
                    #     self.update_mapped_tracks = False
                    #     for i, ident in enumerate(identities):
                    #         # Начальный маппинг
                    #         self.mapped_tracks[ident[0]] = [self.names_identity[ident[1]], 10.0]
                    # Обновление маппинга
                    self.compare_scores(identities, verif_scores)

                    # Отрисовка результатов
                    for i, box in enumerate(bboxes):
                        frame = self.draw_results(frame, box, self.mapped_tracks[i + 1][0], self.mapped_tracks[i + 1][1])

                # elif self.update_mapped_tracks:
                #     for i, box in enumerate(bboxes):
                #         frame = self.draw_results(frame, box, 'Undefined', '1.0')

                else:
                    for i, box in enumerate(bboxes):
                        frame = self.draw_results(frame, box, self.mapped_tracks[i + 1][0], self.mapped_tracks[i + 1][1])

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if frame is not None:
                    cv2.imshow("Verified", frame)
                    # out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        # out.release()
        cv2.destroyAllWindows()


pipe = Pipeline()
pipe.infer()