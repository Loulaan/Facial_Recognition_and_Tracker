from datetime import datetime
from pathlib import Path

import torch
import cv2
import numpy as np

from face_detection import RetinaFace
from arcface.Learner import face_learner
from arcface.config import get_config
from facenet import InceptionResnetV1

torch.set_grad_enabled(False)
color = (0, 255, 0)


class Pipeline:

    def __init__(self, cap_name='0', update_facebank=False):

        self.active_face_verificator = 'arcface'

        self.update_facebank = update_facebank
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.face_detector = RetinaFace(gpu_id=0, network='resnet50')  # resnet50 or mobilenet

        self.show_frames = True
        self.threshold = 1.54
        self.data_for_embeddings = "/home/loulaan/Documents/diploma/data/dataset2"

        self.cap = \
            cv2.VideoCapture(f'{self.data_for_embeddings}/{cap_name}') if cap_name != '0' else cv2.VideoCapture(0)
        self.saved_embeddings = {}
        self.embeddings_calculator = None

        # TODO refactor config approach
        self.face_verification_conf = get_config(False)

        self.face_learner = face_learner(self.face_verification_conf, True)
        self.face_learner.model.load_state_dict(torch.load('arcface/weights/model_ir_se50.pth'))
        self.face_learner.threshold = self.threshold
        self.face_learner.model.eval()

        self.inception_resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()

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
        normilized_face_area = self.face_verification_conf.test_transform(face_area)
        if toTensor:
            return torch.tensor(normilized_face_area).to(self.device).type('torch.cuda.FloatTensor').unsqueeze(0)
        else:
            return np.array(normilized_face_area)

    def prepare_facebank(self):
        if self.update_facebank:
            for path in Path(self.data_for_embeddings).iterdir():
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
                        embs.append(self.calculate_embeddings(img, box, train_learner=True))

                if len(embs) == 0:
                    continue

                self.saved_embeddings[path.parts[-1]] = torch.cat(embs).mean(0)  # 1x1x512

            torch.save(self.saved_embeddings, f"{str(self.data_for_embeddings)}/facebank.pth")
            print('embeddings are calculated')
        else:
            self.saved_embeddings = torch.load(f"{str(self.data_for_embeddings)}/facebank.pth")
            print('embeddings are loaded')

    def calculate_embeddings(self, img, box, train_learner=False):
        face_area = self.convert_bbox_2_img_area(img, box)

        if self.active_face_verificator == 'arcface' and train_learner:
            return self.face_learner.model(face_area).unsqueeze(0)
        if self.active_face_verificator == 'arcface':
            return self.face_learner.model(face_area)
        if self.active_face_verificator == 'facenet':
            return self.inception_resnet(face_area)

    def verify_faces(self, frame):
        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return frame

        target_embeddings = torch.cat(list(self.saved_embeddings.values()))
        names = list(self.saved_embeddings.keys())
        names.append('-1')  # for undefined

        source_embeddings = []
        for face in faces:
            bbox, _, _ = face
            source_embeddings.append(self.calculate_embeddings(frame, bbox))

        diff = torch.cat(source_embeddings).unsqueeze(-1) - \
               target_embeddings.transpose(1, 0).unsqueeze(0)  # #detected_faces x 512 x #persons_in_facebank
        dist = torch.sum(torch.pow(diff, 2), dim=1)  # #detected_faces x #persons_in_facebank (dataset2 = 1x5)
        min_score, min_idx = torch.min(dist, dim=1)
        min_idx[min_score > self.threshold] = -1  # filtering preds
        for idx, face in enumerate(faces):
            frame = self.draw_boxes_and_names(face[0], names[min_idx[idx]], min_score[idx], frame)
        return frame

    def detect_faces(self, frame):
        faces = self.face_detector.detect(frame)
        filtered_face = []
        if faces is not None:
            for face in faces:
                bbox, landmarks, score = face
                if score < 0.6:
                    continue
                bbox = bbox.astype(np.int) + [-5, -5, 5, 5]  # broadcast bboxes with 5 px
                filtered_face.append([bbox, landmarks, score])

        return filtered_face

    def start(self):
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(f'{self.data_for_embeddings}/processed_video3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              size)

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
                    out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()


pipeline = Pipeline(cap_name='video3.mp4', update_facebank=True)
pipeline.start()
