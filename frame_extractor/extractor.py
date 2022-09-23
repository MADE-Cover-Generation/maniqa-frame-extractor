import cv2


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

    def get_frames_count(self, path):
        video = cv2.VideoCapture(path)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        return frame_count
