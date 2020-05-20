#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import numpy
import pdb
import torch
import glob
from tuneThreshold import tuneThresholdfromScore
from SpeakerNet import SpeakerNet
from DatasetLoader import DatasetLoader

parser = argparse.ArgumentParser(description = "SpeakerNet");

## Data loader
parser.add_argument('--max_frames', type=int, default=200,  help='Input length to the network');
parser.add_argument('--batch_size', type=int, default=200,  help='Batch size');
parser.add_argument('--max_seg_per_spk', type=int, default=100, help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=5, help='Number of loader threads');

## Training details
parser.add_argument('--test_interval', type=int, default=10, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int, default=500, help='Maximum number of epochs');
parser.add_argument('--trainfunc', type=str, default="",    help='Loss function');
parser.add_argument('--optimizer', type=str, default="adam", help='sgd or adam');

## Learning rates
parser.add_argument('--lr', type=float, default=0.001,      help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');

## Loss functions
parser.add_argument("--hard_prob", type=float, default=0.5, help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank", type=int, default=10,    help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin', type=float,  default=1,     help='Loss margin, only for some loss functions');
parser.add_argument('--scale', type=float,   default=15,    help='Loss scale, only for some loss functions');
parser.add_argument('--nSpeakers', type=int, default=5994,  help='Number of speakers in the softmax layer for softmax-based losses, utterances per speaker per iteration for other losses');

## Load and save
parser.add_argument('--initial_model',  type=str, default="", help='Initial model weights');
parser.add_argument('--save_path',      type=str, default="./data/exp1", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list', type=str, default="",   help='Train list');
parser.add_argument('--test_list',  type=str, default="",   help='Evaluation list');
parser.add_argument('--train_path', type=str, default="voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',  type=str, default="voxceleb1", help='Absolute path to the test set');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

## Model definition
parser.add_argument('--model', type=str,        default="",     help='Name of model definition');
parser.add_argument('--encoder_type', type=str, default="SAP",  help='Type of encoder');
parser.add_argument('--nOut', type=int,         default=512,    help='Embedding size in the last FC layer');

args = parser.parse_args();

## Initialise directories
model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"
feat_save_path      = ""

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)
        
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

## Load models
s = SpeakerNet(**vars(args));

it          = 1;
prevloss    = float("inf");
sumloss     = 0;

## Load model weights
modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1]);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

for ii in range(0,it-1):
    if ii % args.test_interval == 0:
        clr = s.updateLearningRate(args.lr_decay) 

## Evaluation code
if args.eval == True:
        
    sc, lab = s.evaluateFromListSave(args.test_list, print_interval=100, feat_dir=feat_save_path, test_path=args.test_path)
    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
    print('EER %2.4f'%result[1])

    quit();

## Write args to scorefile
scorefile = open(result_save_path+"/scores.txt", "a+");

for items in vars(args):
    print(items, vars(args)[items]);
    scorefile.write('%s %s\n'%(items, vars(args)[items]));
scorefile.flush()

## Assertion
gsize_dict  = {'proto':args.nSpeakers, 'triplet':2, 'contrastive':2, 'softmax':1, 'amsoftmax':1, 'aamsoftmax':1, 'ge2e':args.nSpeakers, 'angleproto':args.nSpeakers}

assert args.trainfunc in gsize_dict
assert gsize_dict[args.trainfunc] <= 100

## Initialise data loader
trainLoader = DatasetLoader(args.train_list, gSize=gsize_dict[args.trainfunc], **vars(args));

clr = s.updateLearningRate(1)

while(1):   
    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with LR %f..."%(args.model,max(clr)));

    ## Train network
    loss, traineer = s.train_network(loader=trainLoader);

    ## Validate and save
    if it % args.test_interval == 0:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");

        sc, lab = s.evaluateFromListSave(args.test_list, print_interval=100, feat_dir=feat_save_path, test_path=args.test_path)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/T1 %2.2f, TLOSS %f, VEER %2.4f"%( max(clr), traineer, loss, result[1]));
        scorefile.write("IT %d, LR %f, TEER/T1 %2.2f, TLOSS %f, VEER %2.4f\n"%(it, max(clr), traineer, loss, result[1]));

        scorefile.flush()

        clr = s.updateLearningRate(args.lr_decay) 

        s.saveParameters(model_save_path+"/model%09d.model"%it);
        
        eerfile = open(model_save_path+"/model%09d.eer"%it, 'w')
        eerfile.write('%.4f'%result[1])
        eerfile.close()

    else:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER %2.2f, TLOSS %f"%( max(clr), traineer, loss));
        scorefile.write("IT %d, LR %f, TEER %2.2f, TLOSS %f\n"%(it, max(clr), traineer, loss));

        scorefile.flush()

    if it >= args.max_epoch:
        quit();

    it+=1;
    print("");

scorefile.close();





