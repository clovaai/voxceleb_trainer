# VoxCeleb trainer

This repository contains the framework for training speaker recognition models described in the paper '_In defence of metric learning for speaker recognition_' and '_Pushing the limits of raw waveform speaker recognition_'.

### Dependencies
```
pip install -r requirements.txt
```

### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training.

```
python ./dataprep.py --save_path data --download --user USERNAME --password PASSWORD 
python ./dataprep.py --save_path data --extract
python ./dataprep.py --save_path data --convert
```
In order to use data augmentation, also run:

```
python ./dataprep.py --save_path data --augment
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

### Training examples

- ResNetSE34L with AM-Softmax:
```
python ./trainSpeakerNet.py --config ./configs/ResNetSE34L_AM.yaml
```

- RawNet3 with AAM-Softmax
```
python ./trainSpeakerNet.py --config ./configs/RawNet3_AAM.yaml
```

- ResNetSE34L with Angular prototypical:
```
python ./trainSpeakerNet.py --config ./configs/ResNetSE34L_AP.yaml
```

You can pass individual arguments that are defined in trainSpeakerNet.py by `--{ARG_NAME} {VALUE}`.
Note that the configuration file overrides the arguments passed via command line.

### Pretrained models

A pretrained model, described in [1], can be downloaded from [here](http://www.robots.ox.ac.uk/~joon/data/baseline_lite_ap.model).

You can check that the following script returns: `EER 2.1792`. You will be given an option to save the scores.

```
python ./trainSpeakerNet.py --eval --model ResNetSE34L --log_input True --trainfunc angleproto --save_path exps/test --eval_frames 400 --initial_model baseline_lite_ap.model
```

A larger model trained with online data augmentation, described in [2], can be downloaded from [here](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_smproto.model). 

The following script should return: `EER 1.0180`.

```
python ./trainSpeakerNet.py --eval --model ResNetSE34V2 --log_input True --encoder_type ASP --n_mels 64 --trainfunc softmaxproto --save_path exps/test --eval_frames 400  --initial_model baseline_v2_smproto.model
```

Pretrained RawNet3, described in [3], can be downloaded via `git submodule update --init --recursive`.

The following script should return `EER 0.8932`.

```
python ./trainSpeakerNet.py --eval --config ./configs/RawNet3_AAM.yaml --initial_model models/weights/RawNet3/model.pt
```



### Implemented loss functions
```
Softmax (softmax)
AM-Softmax (amsoftmax)
AAM-Softmax (aamsoftmax)
GE2E (ge2e)
Prototypical (proto)
Triplet (triplet)
Angular Prototypical (angleproto)
```

### Implemented models and encoders
```
ResNetSE34L (SAP, ASP)
ResNetSE34V2 (SAP, ASP)
VGGVox40 (SAP, TAP, MAX)
```

### Data augmentation

`--augment True` enables online data augmentation, described in [2].

### Adding new models and loss functions

You can add new models and loss functions to `models` and `loss` directories respectively. See the existing definitions for examples.

### Accelerating training

- Use `--mixedprec` flag to enable mixed precision training. This is recommended for Tesla V100, GeForce RTX 20 series or later models.

- Use `--distributed` flag to enable distributed training.

  - GPU indices should be set before training using the command `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

  - If you are running more than one distributed training session, you need to change the `--port` argument.

### Data

The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets are used for these experiments.

The train list should contain the identity and the file path, one line per utterance, as follows:
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```

The train list for VoxCeleb2 can be download from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt). The
test lists for VoxCeleb1 can be downloaded from [here](https://mm.kaist.ac.kr/datasets/voxceleb/index.html#testlist). 

### Replicating the results from the paper

1. Model definitions
  - `VGG-M-40` in [1] is `VGGVox` in the repository.
  - `Thin ResNet-34` in [1] is `ResNetSE34` in the repository.
  - `Fast ResNet-34` in [1] is `ResNetSE34L` in the repository.
  - `H / ASP` in [2] is `ResNetSE34V2` in the repository.

2. For metric learning objectives, the batch size in the paper is `nPerSpeaker` multiplied by `batch_size` in the code. For the batch size of 800 in the paper, use `--nPerSpeaker 2 --batch_size 400`, `--nPerSpeaker 3 --batch_size 266`, etc.

3. The models have been trained with `--max_frames 200` and evaluated with `--max_frames 400`.

4. You can get a good balance between speed and performance using the configuration below.

```
python ./trainSpeakerNet.py --model ResNetSE34L --trainfunc angleproto --batch_size 400 --nPerSpeaker 2 
```

### Citation

Please cite [1] if you make use of the code. Please see [here](References.md) for the full list of methods used in this trainer.

[1] _In defence of metric learning for speaker recognition_
```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Proc. Interspeech},
  year={2020}
}
```

[2] _The ins and outs of speaker recognition: lessons from VoxSRC 2020_
```
@inproceedings{kwon2021ins,
  title={The ins and outs of speaker recognition: lessons from {VoxSRC} 2020},
  author={Kwon, Yoohwan and Heo, Hee Soo and Lee, Bong-Jin and Chung, Joon Son},
  booktitle={Proc. ICASSP},
  year={2021}
}
```

[3] _Pushing the limits of raw waveform speaker recognition_
```
@inproceedings{jung2022pushing,
  title={Pushing the limits of raw waveform speaker recognition},
  author={Jung, Jee-weon and Kim, You Jin and Heo, Hee-Soo and Lee, Bong-Jin and Kwon, Youngki and Chung, Joon Son},
  booktitle={Proc. Interspeech},
  year={2022}
}
```

### License
```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
