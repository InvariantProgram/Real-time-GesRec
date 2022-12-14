import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from models import mobilenet
from utils import *
from retrainLoader import retrainSet


def finetune_epoch(epoch, data_loader, model, criterion, optimizer):
    print('train at epoch {}'.format(epoch))

    model.train()

    start_time = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        inputs = Variable(inputs)
        target = Variable(target)
        output = model(inputs)

        if torch.cuda.is_available():
            target = target.cuda()
        
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {elapsed_time:.3f}\t'
                  'Loss {loss:.4f}\t'.format(
                      epoch,
                      i,
                      len(data_loader),
                      elapsed_time=end_time - start_time,
                      loss=loss,
                      lr=optimizer.param_groups[0]['lr']))

if __name__ == "__main__":
    pretrained = torch.load("results/jester_mobilenet_0.5x_RGB_16_best.pth")
    #dataset_architect_[width_mult]x_modality_sampleDuration 
    
    model = mobilenet.get_model(num_classes = 27, width_mult = 0.5)
    model = nn.DataParallel(model, device_ids=None)
    model.module.classifier = nn.Sequential(
                                nn.Dropout(0.5),
                                nn.Linear(model.module.classifier[1].in_features, 27))
    model.load_state_dict(pretrained["state_dict"])
    #Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
    #Reset class differentiator for training
    model.module.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.module.classifier[1].in_features, 9)
    )
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    data_loader = retrainSet(input_frames=16) #Input Frame num dim
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)
    checkpointing = 5

    begin_epochs = 0
    end_epochs = 25
    for i in range(begin_epochs, end_epochs+1):
        finetune_epoch(i, data_loader, model, criterion, optimizer)

        if i % checkpointing == 0:
            save_file_path = os.path.join(os.getcwd(), "finetune",
                                        'save_{}.pth'.format(i))
            states = {
                'epoch': i + 1,
                'arch': 'tunedMobileNet',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)