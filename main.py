import os
from datetime import datetime

import torch
import cv2

from utils import draw_results, intersection_over_union, is_proper_predictions
from models import FaceIdentificator, Tracker

torch.set_grad_enabled(False)


class Pipeline:

    def __init__(self, update_facebank=False, show_results=True, video_path=None, save=False):
        """
        :param update_facebank: Calculate new facebank or use old one
        :param show_results: Draw bboxes and ident info at frame
        :param videoPath: Path for processing the video
        :param save: Save processed video
        """
        self.video_path = video_path
        self.face_identificator = FaceIdentificator(update_facebank, f'{os.getcwd()}/data/dataset2')
        self.tracker = Tracker()
        self.names_identity = list(self.face_identificator.saved_embeddings.keys())
        # TODO Create smth more smarter for scores initialization
        self.saved_scores = dict(zip(self.names_identity, [10 for name in self.names_identity]))
        self.mapped_tracks = {}
        self.update_mapped_tracks = True
        self.show_results = show_results
        self.save = save

    def map_faces_with_persons(self, bboxes, faces, idx, identities):
        verified_bboxes = []
        verified_identities = []

        # TODO O(n^2) is bad, optimize this
        for i, bbox in enumerate(bboxes):
            for j, face in enumerate(faces):
                # TODO fix this
                IoU = intersection_over_union(face, bbox)
                if IoU != 0:
                    verified_bboxes.append(bbox)
                    verified_identities.append([identities[i], idx[j]])
                    break
        return verified_bboxes, verified_identities

    def update_relations(self, identities, new_scores):
        compared_identities = []
        for i, identities in enumerate(identities):
            new_track_name = self.names_identity[identities[1]]
            old_track_name = self.mapped_tracks[identities[0]][0]
            try:
                # TODO find out what is the wrong case
                score = new_scores[i]
            except IndexError:
                print("Error")
            if self.mapped_tracks[identities[0]][1] > score:
                # If distance for old name is bigger than new then update name and score
                self.mapped_tracks[identities[0]][1] = score
                self.mapped_tracks[identities[0]][0] = new_track_name
                compared_identities.append(new_track_name)
            else:
                # Otherwise suppose that false-positive identification
                compared_identities.append(old_track_name)
        return compared_identities

    def infer(self):
        cap = cv2.VideoCapture(self.video_path)
        if self.save:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(f'data/dataset2/processed_video_test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Get error while trying to retrieve new frame")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces, idx, verif_scores = self.face_identificator(frame)
            if not is_proper_predictions(faces):
                continue

            bboxes, identities = self.tracker(frame)
            if not is_proper_predictions(bboxes):
                continue

            # TODO пофиксить снижение устойчивости треков
            # # show tracked bboxes
            # for i, bbox in enumerate(bboxes):
            #     frame = draw_results(frame, bbox, identities[i], '1')
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # continue

            # Create ident info for new tracks
            for ident in identities:
                if ident not in list(self.mapped_tracks.keys()):
                    self.mapped_tracks[ident] = ['Undefined', 10.0]

            if faces is not None:
                mapped_bboxes, identities = self.map_faces_with_persons(bboxes, faces, idx, identities)
                # Update mapping
                self.update_relations(identities, verif_scores)

            if self.show_results:
                for i, box in enumerate(bboxes):
                    frame = draw_results(frame, box, self.mapped_tracks[i + 1][0], self.mapped_tracks[i + 1][1])
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


pipe = Pipeline(False, True, f'{os.getcwd()}/data/dataset2/video3.mp4', False)
pipe.infer()
