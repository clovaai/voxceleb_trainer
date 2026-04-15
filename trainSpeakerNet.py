
import argparse
import datetime
import logging
import os
import sys
import warnings
import zipfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from DatasetLoader import train_dataset_loader, train_dataset_sampler, worker_init_fn
from SpeakerNet import ModelTrainer, SpeakerNet, WrappedModel
from tuneThreshold import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore

warnings.simplefilter("ignore")
logging.basicConfig(
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    format='[%(levelname)s] :: %(asctime)s :: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',        dest='augment', action='store_true', help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      dest='log_input', action='store_true', help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
parser.add_argument('--gpu_id',         type=int,   default=0,      help='GPU index for single GPU training')

args = parser.parse_args()

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
            if typ is not None:
                args.__dict__[k] = typ(v)
            else:
                args.__dict__[k] = v
        else:
            sys.stderr.write(f"Ignored unknown parameter {k} in yaml.\n")


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    logger = logging.getLogger('SpeakerNet')

    if args.gpu == 0:
        file_handler = logging.FileHandler(Path(args.result_save_path) / "scores.txt", mode="a+")
        file_handler.setFormatter(logging.Formatter('[%(levelname)s] :: %(asctime)s :: %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(file_handler)

    ## Load models
    s = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        if args.gpu == 0:
            logger.info(f'Loaded the model on GPU {args.gpu:d}')

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    eers = [100]

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))

    train_sampler = train_dataset_sampler(train_dataset, **vars(args))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = sorted(Path(args.model_save_path).glob('model0*.model'))

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        if args.gpu == 0:
            logger.info(f"Model {args.initial_model} loaded!")
    elif len(modelfiles) >= 1:
        epoch = trainer.loadParameters(str(modelfiles[-1]))
        if args.gpu == 0:
            logger.info(f"Model {modelfiles[-1]} loaded from previous state!")
        if epoch is not None:
            ## New checkpoint format: optimizer/scheduler already restored
            it = epoch + 1
        else:
            ## Legacy format: step scheduler manually to restore LR
            it = int(modelfiles[-1].stem[5:]) + 1
            for ii in range(1, it):
                trainer.__scheduler__.step()

    ## Evaluation code - must run on single GPU
    if args.eval == True:

        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        if args.gpu == 0:
            logger.info(f'Total parameters: {pytorch_total_params:d}')
            logger.info(f'Test list {args.test_list}')

        sc, lab, _ = trainer.evaluateFromList(**vars(args))

        if args.gpu == 0:

            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

            logger.info(f"VEER {result[1]:2.4f} MinDCF {mindcf:2.5f}")

        return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = list(Path('.').glob('*.py'))
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(Path(args.result_save_path) / f'run{strtime}.zip', 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(Path(args.result_save_path) / f'run{strtime}.cmd', 'w') as f:
            f.write(f'{args}')

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            logger.info(f"Epoch {it:d}, TEER/TAcc {traineer:2.2f}, TLOSS {loss:f}, LR {max(clr):f}")

        if it % args.test_interval == 0:

            sc, lab, _ = trainer.evaluateFromList(**vars(args))

            if args.gpu == 0:
                
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                logger.info(f"Epoch {it:d}, VEER {result[1]:2.4f}, MinDCF {mindcf:2.5f}")

                trainer.saveParameters(str(Path(args.model_save_path) / f"model{it:09d}.model"), epoch=it)

                with open(Path(args.model_save_path) / f"model{it:09d}.eer", 'w') as eerfile:
                    eerfile.write(f'{result[1]:2.4f}')


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    logger = logging.getLogger('SpeakerNet')

    args.model_save_path     = str(Path(args.save_path) / "model")
    args.result_save_path    = str(Path(args.save_path) / "result")
    args.feat_save_path      = ""

    Path(args.model_save_path).mkdir(parents=True, exist_ok=True)
    Path(args.result_save_path).mkdir(parents=True, exist_ok=True)

    if not args.distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    n_gpus = torch.cuda.device_count()

    logger.info(f'Python Version: {sys.version}')
    logger.info(f'PyTorch Version: {torch.__version__}')
    logger.info(f'Number of GPUs: {torch.cuda.device_count()}')
    logger.info(f'Save path: {args.save_path}')

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()