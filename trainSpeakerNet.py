#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import yaml
import numpy
import pdb
import torch
import glob
import zipfile
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import get_data_loader
import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description = "SpeakerNet");

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=1,      help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=15,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="./data/exp1", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list',     type=str,   default="",     help='Train list');
parser.add_argument('--test_list',      type=str,   default="",     help='Evaluation list');
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set');
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set');
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text');
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')

args = parser.parse_args();

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Trainer script
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    ## Load models
    s = SpeakerNet(**vars(args));

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU %d'%args.gpu)

    else:
        s = WrappedModel(s).cuda(args.gpu)

    prevloss    = float("inf");
    sumloss     = 0;
    min_eer     = [100];
    it          = 1

    ## Write args to scorefile
    scorefile = open(args.result_save_path+"/scores.txt", "a+");

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(args.train_list, **vars(args));
    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1]);
        print("Model %s loaded from previous state!"%modelfiles[-1]);
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    elif(args.initial_model != ""):
        trainer.loadParameters(args.initial_model);
        print("Model %s loaded!"%args.initial_model);

    for ii in range(1,it):
        trainer.__scheduler__.step()
    
    ## Evaluation code - must run on single GPU
    if args.eval == True:

        assert args.distributed == False
            
        sc, lab, _ = trainer.evaluateFromList(args.test_list, print_interval=100, test_path=args.test_path, eval_frames=args.eval_frames)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        print(result[1])
        quit();

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)

    ## Core training script
    for it in range(it,args.max_epoch+1):

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training epoch %d on GPU %d with LR %f "%(it,args.gpu,max(clr)));

        loss, traineer = trainer.train_network(trainLoader, verbose=(args.gpu == 0));

        if it % args.test_interval == 0 and args.gpu == 0:

            trainer.saveParameters(args.model_save_path+"/model%09d.model"%it);

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "TEER/TAcc %2.2f, TLOSS %f"%( traineer, loss));
        scorefile.write("IT %d, TEER/TAcc %2.2f, TLOSS %f\n"%(it, traineer, loss));

        scorefile.flush()

    scorefile.close();


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    if not(os.path.exists(args.model_save_path)):
        os.makedirs(args.model_save_path)
            
    if not(os.path.exists(args.result_save_path)):
        os.makedirs(args.result_save_path)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()