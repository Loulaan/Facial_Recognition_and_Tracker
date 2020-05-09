import os
from datetime import datetime
from pathlib import Path

import torch
import cv2
import numpy as np
from torch.autograd import Variable

from face_detection import RetinaFace
from arcface.Learner import face_learner
from facenet import InceptionResnetV1
from torchvision import transforms as trans

from yolov3.util import *
from yolov3.darknet import Darknet
import pickle as pkl


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
class Pipeline:

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
        for face in faces:
            bbox, _, _ = face
            source_embeddings.append(self.calculate_embeddings(frame, bbox))

        diff = torch.cat(source_embeddings).unsqueeze(-1) - \
               target_embeddings.transpose(1, 0).unsqueeze(0)  # #detected_faces x 512 x #persons_in_facebank
        dist = torch.sum(torch.pow(diff, 2), dim=1)  # #detected_faces x #persons_in_facebank (dataset2 = 1x5)
        min_score, min_idx = torch.min(dist, dim=1)
        min_idx[min_score > self.threshold] = -1  # filtering preds
        for idx, face in enumerate(faces):
            frame = self.draw_boxes_and_names(face[0], names[int(min_idx[idx])], min_score[idx], frame)
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


class YoloV3:

    def __init__(self, show_frames=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.show_frames = show_frames
        self.cfgfile = "yolov3/cfg/yolov3.cfg"
        self.weightsfile = "weights/yolov3/yolov3.weights"

        self.num_classes = 80
        self.confidence = 0.25
        self.nms_thesh = 0.4
        self.bbox_attrs = 5 + self.num_classes

        self.model = Darknet(self.cfgfile).to(device)
        self.model.net_info["height"] = 320

        self.inp_dim = int(self.model.net_info["height"])

        self.classes = load_classes('yolov3/data/coco.names')
        self.colors = pkl.load(open("yolov3/pallete", "rb"))

        self.model.load_weights(self.weightsfile)

    @staticmethod
    def prep_frame(frame, inp_dim):
        """
        Prepare image for inputting to the neural network.

        :return: scaled_img, orig_img and dim (w, h)
        """
        orig_im = frame
        dim = orig_im.shape[1], orig_im.shape[0]
        img = cv2.resize(orig_im, (inp_dim, inp_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim

    def infer(self, frame):
        """
        :param frame: input frame
        :return: list of coords (x1, y1, x2, y2) of localized persons stored on cpu
        """
        bboxes = []
        img, orig_im, dim = self.prep_frame(frame, self.inp_dim)
        output = self.model(Variable(img).to(device))
        output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(self.inp_dim)) / self.inp_dim
        output[:, [1, 3]] *= orig_im.shape[1]
        output[:, [2, 4]] *= orig_im.shape[0]

        for detection in output:
            label = f"{self.classes[int(detection[-1])]}"
            if label != "person":
                continue
            c1 = tuple(detection[1:3].int().to('cpu'))
            c2 = tuple(detection[3:5].int().to('cpu'))
            bboxes.append([c1[0], c1[1], c2[0], c2[1]])

        if self.show_frames:
            for bbox in bboxes:
                cv2.rectangle(orig_im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.imshow('img', orig_im)
            cv2.waitKey()

        return bboxes


yolo = YoloV3(show_frames=True)
print(yolo.infer(cv2.imread('data/parni.jpg')))
