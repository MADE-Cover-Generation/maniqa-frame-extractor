import os
import argparse
import torch
import numpy as np
import random
import cv2
from tqdm import tqdm
from models.maniqa import MANIQA as Maniqa
import gc

from torchvision import transforms
from utils.inference_process import ToTensor, Normalize
from frame_extractor.videodataset import VideoDataset
from frame_extractor.extractor import FrameExtractor


crop_size = 224

#workaround for https://github.com/facebookresearch/fairseq/issues/2510
from torch import Tensor
import torch.nn.functional as F

class GELU(torch.nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


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
    
    
def inference(net, video_dataset, extractor, number_of_frames = 4):
    for video in video_dataset:
        print(video)
        frames = extractor.get_frames(video)
        step = extractor.get_step()
        current = 0
        scored_frames = []
        for frame in frames:
            print("----- \n current frame = ", current + 1)
            prepared = prepare(frame)
            img = prepared['d_img_org']
            img = random_crop(img)
            res = net(img)
            scored_frames.append((current * step, res[0].item()))
            current += 1
            gc.collect()
            torch.cuda.empty_cache()
        
        scored_frames.sort(key=lambda tup: tup[1], reverse=True)
        print(scored_frames)
        for i in range(0, number_of_frames):
            output_path = args.output + "/" + os.path.basename(video) + f"_{i}" + "_maniqa" + ".jpg"
            cv2.imwrite(output_path, extractor.get_frame(video, scored_frames[i][0]))


# def inference(net, video_dataset, extractor):
#     for video in video_dataset:
#         print(video)
#         frames = extractor.get_frames(video)
#         max = 0
#         current = 0
#         best_frame = None
#         for frame in frames:
#             print("----- \n current frame = ", current + 1)
#             prepared = prepare(frame)
#             img = prepared['d_img_org']
#             img = random_crop(img)
#             res = net(img)
#             if res > max:
#                 max = res
#                 best_frame = frame

#             gc.collect()
#             torch.cuda.empty_cache()
#             current += 1

#         output_path = args.output + "/" + os.path.basename(video)  + "_maniqa" + ".jpg"
#         cv2.imwrite(output_path, best_frame)



def generate_random_frames(video_dataset, extractor):
    for video in video_dataset:
        frame = extractor.get_random_frame(video)
        output_path = args.output + "/" + os.path.basename(video) + "_random" + ".jpg"
        cv2.imwrite(output_path, frame)

if __name__ == '__main__':
    args = parse_args()
    data = VideoDataset(args.input)
    extractor = FrameExtractor()
    if not args.random:
        device = torch.device('cpu')
        net = torch.nn.DataParallel(Maniqa())
        net.load_state_dict(torch.load(args.model, device), strict=False)
        net = net.module.to(device)
        net.eval()
        inference(net, data, extractor)
    else:
        generate_random_frames(data, extractor)
