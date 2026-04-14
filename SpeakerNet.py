
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import itertools
import random

logger = logging.getLogger(__name__)

import numpy
from tqdm import tqdm

from DatasetLoader import test_dataset_loader
from torch.amp import autocast, GradScaler


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super().__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).to(next(self.parameters()).device)
        outp = self.__S__.forward(data)

        if label is None:
            return outp

        else:

            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1


class ModelTrainer:
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler("cuda")

        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train()

        counter = 0
        loss = 0
        top1 = 0

        with tqdm(loader, unit="batch", disable=not verbose) as tepoch:

            for data, data_label in tepoch:

                data = data.transpose(1, 0)

                self.__model__.zero_grad()

                label = torch.LongTensor(data_label).to(self.device)

                if self.mixedprec:
                    with autocast("cuda"):
                        nloss, prec1 = self.__model__(data, label)
                    self.scaler.scale(nloss).backward()
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update()
                else:
                    nloss, prec1 = self.__model__(data, label)
                    nloss.backward()
                    self.__optimizer__.step()

                loss += nloss.detach().cpu().item()
                top1 += prec1.detach().cpu().item()
                counter += 1

                # Print statistics to progress bar
                tepoch.set_postfix(loss=loss/counter)

                if self.lr_step == "iteration":
                    self.__scheduler__.step()

            if self.lr_step == "epoch":
                self.__scheduler__.step()

        return (loss / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, num_eval=10, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        verbose = (rank == 0)

        self.__model__.eval()

        lines = []
        files = []
        feats = {}

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        with tqdm(test_loader, unit="utt", desc="Extracting", disable=not verbose) as tepoch:
            for data in tepoch:
                inp1 = data[0][0].to(self.device)
                with torch.no_grad():
                    ref_feat = self.__model__(inp1).detach().cpu()
                feats[data[1][0]] = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            with tqdm(lines, unit="trial", desc="Scoring", disable=not verbose) as tepoch:
                for line in tepoch:

                    data = line.split()

                    ## Append random label if missing
                    if len(data) == 2:
                        data = [random.randint(0, 1)] + data

                    ref_feat = feats[data[1]].to(self.device)
                    com_feat = feats[data[2]].to(self.device)

                    if self.__model__.module.__L__.test_normalize:
                        ref_feat = F.normalize(ref_feat, p=2, dim=1)
                        com_feat = F.normalize(com_feat, p=2, dim=1)

                    dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

                    score = -1 * numpy.mean(dist)

                    all_scores.append(score)
                    all_labels.append(int(data[0]))
                    all_trials.append(data[1] + " " + data[2])

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path, epoch):

        torch.save({
            "model_state_dict": self.__model__.module.state_dict(),
            "optimizer_state_dict": self.__optimizer__.state_dict(),
            "scheduler_state_dict": self.__scheduler__.state_dict(),
            "epoch": epoch,
        }, path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        ## New checkpoint format
        if "model_state_dict" in checkpoint:
            loaded_state = checkpoint["model_state_dict"]
            self.__optimizer__.load_state_dict(checkpoint["optimizer_state_dict"])
            self.__scheduler__.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint["epoch"]
        else:
            ## Legacy format: bare state_dict or {"model": state_dict}
            loaded_state = checkpoint
            epoch = None

        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    logger.warning(f"{origname} is not in the model.")
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                logger.warning(f"Wrong parameter length: {origname}, model: {self_state[name].size()}, loaded: {loaded_state[origname].size()}")
                continue

            self_state[name].copy_(param)

        return epoch
