import os
from datetime import datetime

import cv2

from utils import draw_results, map_face_and_person, is_proper_predictions
from models import FaceIdentificator, Tracker


class Pipeline:

    def __init__(self, update_facebank=False, show_results=True, dataset_path=None, save=False):
        """
        :param update_facebank: Calculate new facebank or use old one
        :param show_results: Draw bboxes and ident info at frame
        :param videoPath: Path for processing the video
        :param save: Save processed video
        """
        self.dataset_path = dataset_path
        self.video_path = f'{self.dataset_path}/{self.find_video()}'
        self.face_identificator = FaceIdentificator(update_facebank, f'{os.getcwd()}/{self.dataset_path}')
        self.tracker = Tracker()
        self.unique_facebank_names = list(self.face_identificator.saved_embeddings.keys())
        # TODO Create smth more smarter for scores initialization
        self.saved_scores = dict(
            zip(
                self.unique_facebank_names,
                [self.face_identificator.threshold for name in self.unique_facebank_names]
            )
        )
        self.mapped_tracks = {}
        self.show_results = show_results
        self.treshold = 0.4
        self.save = save

    def find_video(self):
        file_list = os.listdir(self.dataset_path)
        for file in file_list:
            if file[-4:] == '.mp4' and file[:9] != "processed":
                return file
        raise FileNotFoundError("Can't find a video for processing")

    def map_faces_with_persons(self, bboxes, faces, idx, identities, scores):
        verified_bboxes = []
        verified_identities = []
        verified_scores = []

        # TODO O(n^2) is bad, optimize this
        for i, bbox in enumerate(bboxes):
            for j, face in enumerate(faces):

                # TODO fix this
                if map_face_and_person(face, bbox):
                    verified_bboxes.append(bbox)
                    verified_identities.append([identities[i], idx[j]])
                    verified_scores.append(scores[j])
                    break
        return verified_bboxes, verified_identities, verified_scores

    def update_relations(self, identities, new_scores):
        """
        :param identities: mapped track's and names idxs
        :param new_scores: computed scores of facial recognition
        :return: Nothing
        """
        for i, ident in enumerate(identities):
            new_track_name = 'Undefined' if ident[1] == -1 else self.unique_facebank_names[ident[1]]
            try:
                # TODO find out what is the wrong case
                score = new_scores[i]
            except IndexError:
                print("Error")
                if len(new_scores) == 1:
                    score = new_scores
            if self.mapped_tracks[ident[0]][1] > score:
                # If distance for old name is bigger than new then update name and score
                self.mapped_tracks[ident[0]][1] = score
                self.mapped_tracks[ident[0]][0] = new_track_name

    def infer(self):
        cap = cv2.VideoCapture(self.video_path)
        if self.save:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(
                f'{self.dataset_path}/processed_video.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                size)

        # [face detection and recognition, person detection and tracking, mapping faces with persons]
        time = [[], [], []]

        initial_time = datetime.now()
        frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Get error while trying to retrieve new frame or video has ended.")
                break

            frames += 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = datetime.now().microsecond
            faces, idx, verif_scores = self.face_identificator(frame)
            delta = datetime.now().microsecond - start_time
            if delta > 0:
                time[0].append(delta)

            start_time = datetime.now().second
            bboxes, identities = self.tracker(frame)
            delta = datetime.now().microsecond - start_time
            if delta > 0:
                time[1].append(delta)


            # TODO пофиксить снижение устойчивости треков
            # # show tracked bboxes
            # for i, bbox in enumerate(bboxes):
            #     frame = draw_results(frame, bbox, identities[i], '1')
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # continue

            # Create ident info for new tracks
            if is_proper_predictions(bboxes):
                for ident in identities:
                    if ident not in list(self.mapped_tracks.keys()):
                        self.mapped_tracks[ident] = ['Undefined', self.face_identificator.threshold]
                if is_proper_predictions(faces):
                    start_time = datetime.now().microsecond
                    mapped_bboxes, identities, scores = self.map_faces_with_persons(
                        bboxes,
                        faces,
                        idx,
                        identities,
                        verif_scores
                    )
                    # Update mapping
                    self.update_relations(identities, scores)
                    delta = datetime.now().microsecond - start_time
                    if delta > 0:
                        time[2].append(delta)

                if self.show_results:
                    tracks_numbers = list(self.mapped_tracks.keys())
                    if len(tracks_numbers) != 0:
                        for i, box in enumerate(bboxes):
                            frame = draw_results(
                                frame,
                                box,
                                self.mapped_tracks[tracks_numbers[i]][0],
                                self.mapped_tracks[tracks_numbers[i]][1]
                            )
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Verified", frame)

            if self.save:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



        cap.release()

        if self.save:
            out.release()

        cv2.destroyAllWindows()

        finish_time = float(frames)/(datetime.now() - initial_time).seconds

        print('\n\nRetinaFace + FaceNet: ', int(average(time[0])),
              '| YOLOv3 + DeepSort: ', int(average(time[1])),
              '| Update Relations: ', int(average(time[2])),
              '\n')

        print('Average FPS: ', finish_time)


def average(items):
    return float(sum(items))/len(items)


pipe = Pipeline(
    update_facebank=False,
    show_results=True,
    dataset_path='data/dataset4',
    save=False
)

pipe.infer()
