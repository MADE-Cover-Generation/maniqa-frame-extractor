import os
import argparse
import torch
import numpy as np
import random
import cv2
from tqdm import tqdm
from models.maniqa import MANIQA as Maniqa
import gc
import piq
from PIL import Image

from torchvision import transforms
from utils.inference_process import ToTensor, Normalize
from frame_extractor.videodataset import VideoDataset
from frame_extractor.extractor import FrameExtractor

import imagehash


crop_size = 224
VERBOSE = False

#workaround for https://github.com/facebookresearch/fairseq/issues/2510
from torch import Tensor
import torch.nn.functional as F

class GELU(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

torch.nn.modules.activation.GELU = GELU


FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model')
    parser.add_argument('--step', help='distance between frames', type=int, default=30)
    parser.add_argument('--output', help='output folder')
    parser.add_argument('--input', help='input folder')
    parser.add_argument('--random', help='random frame mode', action="store_true")
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
    

def brisque_filter(frames, device):
    result = []
    previous_hash = None
    for frame in frames[0 : len(frames)]:
        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_)
        current_hash = imagehash.average_hash(image)
        if previous_hash is not None and previous_hash - current_hash < 5:
            previous_hash = current_hash
            continue
        previous_hash = current_hash
        shot = (
            torch.tensor(frame_).permute(2, 0, 1) / 255.0
        )
        try:
            brisque_score = piq.brisque(
                shot[None, ...].to(device), data_range=1.0, reduction="none"
            ).item()
        except AssertionError:
            continue
        result.append((frame, brisque_score))
    
    result.sort(key=lambda tup: tup[1])
    frames = [i[0] for i in result]
    scores = [i[1] for i in result]
    print(scores)
    gc.collect()
    return frames


def inference(net, frames, device):
    current = 0
    scored_frames = []
    print("----- \n number of frames", len(frames))
    for frame in frames:
        print("----- \n current frame = ", current + 1)
        prepared = prepare(frame)
        img = prepared['d_img_org'].to(device)
        img = random_crop(img)

        res = net(img)
        scored_frames.append((frame, res[0].item()))
        current += 1
        gc.collect()
        torch.cuda.empty_cache()
    
    scored_frames.sort(key=lambda tup: tup[1], reverse=True)
    return scored_frames[0][0]


def inference_in_groups(net, video_dataset, extractor, device, number_of_frames = 4):
    for video in video_dataset:
        print(video)
        frames_groups = extractor.get_frames_in_groups(video)
        result = []
        print(len(frames_groups[0]))
        for i in range(0, len(frames_groups)):
            result.append(inference(net, frames_groups[i], device))
        
        output_pathes = []

        for i in range(0, number_of_frames):
            image = result[i]
            output_path = args.output + "/" + os.path.basename(video) + f"_{i}" + "_maniqa" + ".jpg"
            output_pathes.append(output_path)
            cv2.imwrite(output_path, image)

        images = []
        for i in range(0, len(output_pathes)):
            images.append(cv2.imread(output_pathes[i]))

        im_h_first = cv2.hconcat(images[0 : int(number_of_frames / 2)])
        im_h_second = cv2.hconcat(images[int(number_of_frames / 2) : len(images) + 1])
        im_v = cv2.vconcat((im_h_first, im_h_second))
        if VERBOSE:
            im_v = cv2.putText(im_v, "maniqa", (10,450), FONT, 3, (0, 255, 0), 2, cv2.LINE_AA)
        output_path = args.output + "/" + os.path.basename(video) + "_maniqa" + ".jpg"
        for path in output_pathes:
            os.remove(path)
        cv2.imwrite(output_path, im_v)


def generate_random_frames(video_dataset, extractor, number_of_frames = 4):
    for video in video_dataset:
        output_pathes = []
        result = extractor.get_random_frames_in_groups(video)
        for i in range(0, len(result)):
            frame = result[i]
            output_path = args.output + "/" + os.path.basename(video) + f"_{i}" + "_random" + ".jpg"
            output_pathes.append(output_path)
            cv2.imwrite(output_path, frame)
        
        images = []
        for i in range(0, len(output_pathes)):
            images.append(cv2.imread(output_pathes[i]))

        im_h_first = cv2.hconcat(images[0 : int(number_of_frames / 2)])
        im_h_second = cv2.hconcat(images[int(number_of_frames / 2) : len(images) + 1])
        im_v = cv2.vconcat((im_h_first, im_h_second))
        if VERBOSE:
            im_v = cv2.putText(im_v, "random", (10,450), FONT, 3, (0, 255, 0), 2, cv2.LINE_AA)
        output_path = args.output + "/" + os.path.basename(video) + "_random" + ".jpg"
        for path in output_pathes:
            print(path)
            os.remove(path)
        cv2.imwrite(output_path, im_v)
        gc.collect()


model_path = "/home/kirill/Documents/final_project_dataset/ckpt_valid"


if __name__ == '__main__':
    args = parse_args()
    data = VideoDataset(args.input)
    if not args.random:
        extractor = FrameExtractor(step=10)
        device = torch.device('cuda')
        net = torch.load(model_path)
        net = net.cuda()
        net.eval()
        inference_in_groups(net, data, extractor, device)
    else:
        extractor = FrameExtractor(step = 10)
        generate_random_frames(data, extractor)
