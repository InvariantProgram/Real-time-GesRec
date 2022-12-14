import os
import sys
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import cv2
from utils import *

from models import mobilenet

if __name__ == '__main__':
    #gestures = open("annotation_Jester/categories.txt").readlines()
    gestures = open("finetune/matchfile.txt").readlines()
    gestures = [gesture.split()[-1] for gesture in gestures]

    #pretrained = torch.load("results/jester_mobilenet_0.5x_RGB_16_best.pth")
    #dataset_architect_[width_mult]x_modality_sampleDuration 

    ##Finetuned version
    pretrained = torch.load("finetune/save_25.pth")
    
    model = mobilenet.get_model(num_classes = 27, width_mult = 0.5)
    model = nn.DataParallel(model, device_ids=None)
    
    #model.module.classifier = nn.Sequential(
    #                            nn.Dropout(0.5),
    #                            nn.Linear(model.module.classifier[1].in_features, 27)
    #)
    
    ##Finetuned Version
    model.module.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.module.classifier[1].in_features, 9)
    )
    model.load_state_dict(pretrained["state_dict"])
    model = model.cuda()
    model.eval()


    print("Starting Video Process")
    cv2.namedWindow("preview")

    q = list()
    max_size = 16 # Number of frames stored
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    
    last_input = None
    while rval:
        if len(q) >= max_size:
            test = q.pop() #numpy array (480, 640, 3)
        q.insert(0, torch.Tensor(frame))

        max_ind = None
        if len(q) == max_size:
            with torch.no_grad():
                input = torch.unsqueeze(torch.stack(q, 0).permute(3, 0, 1, 2), 0)
                '''
                if last_input is not None:
                    diff = torch.sum(input - last_input).item()
                    n = 1
                    for dim in input.size():
                        n *= dim
                    mean = diff / n
                    print("Differences:", diff, "Average:", mean)
                last_input = input.detach()
                '''
                output = model(input) #torch.size([1, 27])
                output = F.softmax(output, dim=1)
                max_ind = torch.topk(output, 3)

        rval, frame = vc.read()
        if max_ind is not None:
            for i in range(max_ind[1].size()[1]):
                gest_str = "%d. %s: %.4f" % (i+1, gestures[max_ind[1][0][i]], max_ind[0][0][i])
                cv2.putText(frame, gest_str, (0, 20 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imshow("preview", frame)

        key = cv2.waitKey(20)
        if key == 27:
            break

    cv2.destroyWindow("preview")
    vc.release()
    