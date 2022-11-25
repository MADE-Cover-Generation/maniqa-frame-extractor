import cv2
import random

class FrameExtractor():

    def __init__(self, step=30):
        self.step = step
        
    def get_frames(self, path):
        result = []
        frame_count = self.get_frames_count(path)
        capture = cv2.VideoCapture(path)
        current_frame = 0
        success = True
        while success and current_frame < frame_count:
            current_frame = current_frame + self.step
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            success, image = capture.read()
            if success:
                result.append(image)
        capture.release()
        return result

    def get_frames_in_groups(self, path, n_groups = 4):
        result = []
        frame_count = self.get_frames_count(path)
        frame_count_in_group = frame_count // n_groups
        print(f"frames in group = {frame_count}")
        capture = cv2.VideoCapture(path)
        for i in range(0, n_groups):
            current_frames = self.__get_frames_in_range(capture, frame_count_in_group * i, frame_count_in_group * (i + 1))
            result.append(current_frames)
        capture.release()
        return result

    def __get_frames_in_range(self, capture, start, end):
        result = []
        current_frame = start
        success = True
        while success and current_frame < end:
            current_frame = current_frame + self.step
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            success, image = capture.read()
            if success:
                result.append(image)
        
        return result

    
    def get_random_frames_in_groups(self, path, n_groups = 4):
        result = []
        current_frame = 0
        frame_count = self.get_frames_count(path)
        frame_count_in_group = frame_count // n_groups
        capture = cv2.VideoCapture(path)
        success = True
        for i in range(0, n_groups):
            frame = random.randrange(frame_count_in_group * i, frame_count_in_group * (i + 1))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, image = capture.read()
            result.append(image)
        
        return result

    def get_step(self) -> int:
        return self.step

    def get_fps(self, path) -> float:
        capture = cv2.VideoCapture(path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        return fps

    def get_frames_count(self, path):
        capture = cv2.VideoCapture(path)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        return frame_count

    def get_random_frame(self, path):
        frame_count = self.get_frames_count(path)
        capture = cv2.VideoCapture(path)
        frame = random.randrange(0, frame_count)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = capture.read()
        capture.release()
        return image 
        
    def get_frame(self, path: str, index: int):
        capture = cv2.VideoCapture(path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        _, image = capture.read()
        capture.release()
        return image
