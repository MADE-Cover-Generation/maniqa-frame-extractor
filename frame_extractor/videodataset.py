import glob
import os.path


class VideoDataset():
    def __init__(self, dir_path):
        self.path = dir_path
        self.files = glob.glob(os.path.join(dir_path, '*.webm'))
        print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]
