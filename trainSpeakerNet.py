#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
warnings.simplefilter("ignore")

# Try to import new dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    print("Loaded TensorBoard SummaryWriter.")
except ImportError:
    print("TensorBoard not found. Please run 'pip install tensorboard' to enable logging.")
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
    print("Loaded Matplotlib.")
except ImportError:
    print("Matplotlib not found. Please run 'pip install matplotlib' to enable ROC curve plotting.")
    plt = None


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
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')
parser.add_argument('--patience',       type=int,   default=10,     help='Number of test intervals to wait for EER improvement before early stopping (0 to disable)')

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
parser.add_argument('--nClasses',       type=int,   default=5991,   help='Number of speakers in the softmax layer, only for softmax-based losses')

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
parser.add_argument('--musan_path',     type=str,   default="", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
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
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write(f"Ignored unknown parameter {k} in yaml.\n")


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    # Add performance optimization
    torch.backends.cudnn.benchmark = True

    ## Load models
    s = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print(f'Loaded the model on GPU {args.gpu}')

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    eers = [100]
    
    # Define variables for early stopping
    best_eer = float('inf')
    epochs_since_improvement = 0
    best_model_path = os.path.join(args.model_save_path, "model_best.model")
    best_eer_path = os.path.join(args.model_save_path, "model_best.eer")
    best_threshold_path = os.path.join(args.model_save_path, "model_best.threshold")
    best_roc_curve_path = os.path.join(args.result_save_path, "roc_curve_best.png")
    
    # Initialize TensorBoard writer
    writer = None
    if args.gpu == 0 and SummaryWriter is not None:
        log_dir = os.path.join(args.save_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging enabled. Log directory: {log_dir}")
    

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile_path = os.path.join(args.result_save_path, "scores.txt")
        scorefile   = open(scorefile_path, "a+")
        print(f"Score file opened at: {scorefile_path}")
        
        # Check if a best EER file already exists (for resuming)
        if os.path.exists(best_eer_path):
            try:
                with open(best_eer_path, 'r') as f:
                    best_eer = float(f.readline().strip())
                print(f"Resuming training, best EER so far: {best_eer:2.4f}%")
            except:
                print(f"Could not read {best_eer_path}, starting EER from infinity.")
                best_eer = float('inf')


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
    # Find all model0*.model files, sort them, and remove 'model_best.model' if it's caught
    modelfiles = glob.glob(os.path.join(args.model_save_path, 'model0*.model'))
    modelfiles.sort()
    if best_model_path in modelfiles:
        modelfiles.remove(best_model_path)

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print(f"Model {args.initial_model} loaded!")
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print(f"Model {modelfiles[-1]} loaded from previous state!")
        # Get epoch number from filename (e.g., model000000010.model -> 10)
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Evaluation code - must run on single GPU
    if args.eval == True:

        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        print(f'Total parameters: {pytorch_total_params}')
        print(f'Test list: {args.test_list}')
        
        sc, lab, _ = trainer.evaluateFromList(**vars(args))

        if args.gpu == 0:

            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

            print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")}, VEER {result[1]:2.4f}, MinDCF {mindcf:2.5f}, Threshold {result[2]:f}')

        return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(os.path.join(args.result_save_path, f'run{strtime}.zip'), 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(os.path.join(args.result_save_path, f'run{strtime}.cmd'), 'w') as f:
            f.write(f'{args}')

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Epoch {it}, TEER/TAcc {traineer:2.2f}, TLOSS {loss:f}, LR {max(clr):f}')
            scorefile.write(f"Epoch {it}, TEER/TAcc {traineer:2.2f}, TLOSS {loss:f}, LR {max(clr):f} \n")
            
            # Log training stats to TensorBoard
            if writer is not None:
                writer.add_scalar('Train/Loss', loss, it)
                writer.add_scalar('Train/EER', traineer, it)
                writer.add_scalar('Train/LR', max(clr), it)


        if it % args.test_interval == 0:

            sc, lab, _ = trainer.evaluateFromList(**vars(args))

            if args.gpu == 0:
                
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])
                current_eer = result[1]
                current_threshold = result[2] # <-- Capture threshold

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
                
                eers.append(current_eer)
                
                print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Epoch {it}, VEER {current_eer:2.4f}, MinDCF {mindcf:2.5f}, Threshold {current_threshold:f}')
                scorefile.write(f"Epoch {it}, VEER {current_eer:2.4f}, MinDCF {mindcf:2.5f}, Threshold {current_threshold:f}\n")

                # Log validation stats to Tensorboard
                if writer is not None:
                    writer.add_scalar('Val/EER', current_eer, it)
                    writer.add_scalar('Val/MinDCF', mindcf, it)

                # --- NEW BEST MODEL & EARLY STOPPING LOGIC ---
                
                if current_eer < best_eer:
                    print(f'🎉 New best EER: {current_eer:2.4f}% (was {best_eer:2.4f}%)')
                    best_eer = current_eer
                    epochs_since_improvement = 0 # Reset patience
                    
                    # Save the best model
                    trainer.saveParameters(best_model_path)
                    with open(best_eer_path, 'w') as eerfile:
                        eerfile.write(f'{best_eer:2.4f}')
                    
                    # Save the best threshold
                    with open(best_threshold_path, 'w') as f:
                        f.write(f'{current_threshold:f}')

                    print(f'SAVING BEST MODEL (Epoch {it}) to {best_model_path}')
                    print(f'SAVING BEST THRESHOLD ({current_threshold:f}) to {best_threshold_path}')

                    # Plot and save ROC curve
                    if plt is not None:
                        try:
                            tprs = 1 - fnrs
                            fig = plt.figure()
                            plt.plot(fprs, tprs, label=f'ROC (EER = {current_eer:2.2f}%)')
                            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
                            
                            # Find and plot the EER point
                            eer_fpr = fprs[numpy.nanargmin(numpy.abs(fnrs - fprs))]
                            eer_tpr = tprs[numpy.nanargmin(numpy.abs(fnrs - fprs))]
                            plt.plot(eer_fpr, eer_tpr, 'ro', label=f'EER Point ({eer_fpr:.2f}, {eer_tpr:.2f})')

                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate (1 - FNR)')
                            plt.title(f'ROC Curve - Epoch {it}')
                            plt.legend()
                            plt.grid(True)
                            
                            # Save to file
                            plt.savefig(best_roc_curve_path)
                            print(f"Saved new best ROC curve to {best_roc_curve_path}")

                            # Add to TensorBoard
                            if writer is not None:
                                writer.add_figure('Val/ROC_Curve', fig, global_step=it)
                            
                            plt.close(fig) # Close figure to free memory
                        
                        except Exception as e:
                            print(f"Failed to plot or save ROC curve: {e}")

                else:
                    epochs_since_improvement += 1
                    print(f'EER did not improve: {current_eer:2.4f}% (Best is {best_eer:2.4f}%)')
                
                # --- ORIGINAL CHECKPOINTING (for resuming) ---
                # Always save the latest interval checkpoint
                latest_model_path = os.path.join(args.model_save_path, f"model{it:09d}.model")
                trainer.saveParameters(latest_model_path)
                
                latest_eer_path = os.path.join(args.model_save_path, f"model{it:09d}.eer")
                with open(latest_eer_path, 'w') as eerfile:
                    eerfile.write(f'{current_eer:2.4f}')
                
                print(f"Saved interval checkpoint to {latest_model_path}")
                
                scorefile.flush()

                # --- CHECK FOR EARLY STOPPING ---
                if args.patience > 0 and epochs_since_improvement >= args.patience:
                    print(f'\nNo EER improvement for {epochs_since_improvement} test intervals (patience={args.patience}). EARLY STOPPING.')
                    scorefile.write(f"\nEarly stopping at epoch {it}.\n")
                    break # Exit the main training loop
        
        # Check if the loop was broken by early stopping
        if args.patience > 0 and epochs_since_improvement >= args.patience:
            break

    if args.gpu == 0:
        scorefile.close()
        if writer is not None:
            writer.close()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path     = os.path.join(args.save_path, "model")
    args.result_save_path    = os.path.join(args.save_path, "result")
    args.feat_save_path      = "" # Not used in this script, but kept for compatibility

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print(f'Number of GPUs: {n_gpus}')
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, n_procs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()