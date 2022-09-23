import os
import argparse
import torch
import numpy as np
import random
import cv2
from tqdm import tqdm

from torchvision import transforms
from utils.inference_process import ToTensor, Normalize, sort_file
from frame_extractor.videodataset import VideoDataset
from frame_extractor.extractor import FrameExtractor


crop_size = 224


#workaround for https://github.com/facebookresearch/fairseq/issues/2510
class GELU(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(input)
torch.nn.modules.activation.GELU = GELU


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model')
    parser.add_argument('--step', help='distance between frames', type=int, default=30)
    parser.add_argument('--output', help='output folder')
    parser.add_argument('--input', help='input folder')
    args = parser.parse_args()
    return args


def prepare(img):
    d_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    d_img = np.array(d_img).astype('float32') / 255
    d_img = np.transpose(d_img, (2, 0, 1))
    sample = {
        'd_img_org': d_img,
        'd_name': ""
    }
    transform = transforms.Compose([Normalize(0.5, 0.5), ToTensor()])
    sample = transform(sample)
    return sample

def random_crop(d_img):
    c, h, w = d_img.shape
    top = np.random.randint(0, h - crop_size)
    left = np.random.randint(0, w - crop_size)
    d_img_org = crop_image(top, left, crop_size, img=d_img)
    d_img_org = d_img_org[None, :]
    return d_img_org


def crop_image(top, left, patch_size, img=None):
    tmp_img = img[:, top:top + patch_size, left:left + patch_size]
    return tmp_img


if __name__ == '__main__':
    args = parse_args()
    data = VideoDataset(args.input)
    extractor = FrameExtractor(args.step)
    net = torch.load(args.model)
    net.eval()
    net = net.cuda()

    for video in data:
        frames = extractor.get_frames(video)
        max = 0
        best_frame = None
        for frame in frames:
            prepared = prepare(frame)
            img = prepared['d_img_org'].cuda()
            img = random_crop(img)
            res = net(img)
            if res > max:
                max = res
                best_frame = frame
        output_path = args.output + "/" + os.path.basename(video) + ".jpg"
        cv2.imwrite(output_path, best_frame)