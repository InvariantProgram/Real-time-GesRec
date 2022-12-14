import numpy as np
import pandas as pd 
from torch.utils.data import Dataset
import random
import torch
import torch.nn.functional as F
import os
import cv2

class retrainSet(Dataset):
    def __init__(self, input_frames):
        #[ccw, cw, down, left, none, pull, push, right, up]
        self.num_classes = 9

        self.samples = input_frames

        rootpath = "./Retrain_vids/"
        self.files = []
        self.classes = {}
        for i, dir in enumerate(os.listdir(rootpath)):
            self.classes[i] = dir
            if not os.path.isfile(rootpath + dir):
                for file in os.listdir(rootpath + dir):
                    self.files.append(("{0}/{1}/{2}".format(rootpath, dir, file), i)) #Form (file_path, class_num)
    
    def __len__(self):
        return len(self.files)    
    
    def __getitem__(self, idx):
        path, class_val = self.files[idx]
        
        vidcap = cv2.VideoCapture(path)
        totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = list()
        for iloc in sorted(random.sample(range(totalFrames), self.samples)):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, iloc)
            rval, frame = vidcap.read()
            if rval:
                frames.append(torch.FloatTensor(frame))
            else:
                print("Error reading video frame")
        
        training_input = torch.unsqueeze(torch.stack(frames, 0).permute(3, 0, 1, 2), 0)      
        label = torch.tensor([class_val])

        return training_input, label