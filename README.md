## VoxCeleb trainer

This repository contains the training code for 'In defence of metric learning for speaker recognition.'

#### Dependencies
```
pip install -r requirements.txt
```

#### Data preparation

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

#### Training examples

- AM-Softmax:
```
python ./trainSpeakerNet.py --model ResNetSE34L --log_input True --encoder SAP --trainfunc amsoftmax --save_path exps/exp1 --nClasses 5994 --batch_size 200 --scale 30 --margin 0.3 --train_list train_list.txt --test_list test_list.txt
```

- Angular prototypical:
```
python ./trainSpeakerNet.py --model ResNetSE34L --log_input True --encoder SAP --trainfunc angleproto --save_path exps/exp2 --nPerSpeaker 2 --batch_size 200 --train_list train_list.txt --test_list test_list.txt
```

The arguments can also be passed as `--config path_to_config.yaml`. Note that the configuration file overrides the arguments passed via command line.

#### Pretrained models

A pretrained model can be downloaded from [here](http://www.robots.ox.ac.uk/~joon/data/baseline_lite_ap.model).

You can check that the following script returns: `EER 2.1792`. You will be given an option to save the scores.

```
python ./trainSpeakerNet.py --eval --model ResNetSE34L --log_input True --trainfunc angleproto --save_path exps/test --eval_frames 400 --test_list test_list.txt --initial_model baseline_lite_ap.model
```

A larger model trained with data augmentation can be downloaded from [here](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model). 

The following script should return: `EER 1.1771`. 

```
python ./trainSpeakerNet.py --eval --model ResNetSE34V2 --log_input True --encoder_type ASP --n_mels 64 --trainfunc softmaxproto --save_path exps/test --eval_frames 400 --test_list test_list.txt --initial_model baseline_v2_ap.model
```

#### Implemented loss functions
```
Softmax (softmax)
AM-Softmax (amsoftmax)
AAM-Softmax (aamsoftmax)
GE2E (ge2e)
Prototypical (proto)
Triplet (triplet)
Angular Prototypical (angleproto)
```

#### Implemented models and encoders
```
ResNetSE34 (SAP)
ResNetSE34L (SAP, ASP)
ResNetSE34V2 (SAP, ASP)
VGGVox40 (SAP, TAP, MAX)
```

#### Adding new models and loss functions

You can add new models and loss functions to `models` and `loss` directories respectively. See the existing definitions for examples.

#### Data

The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets are used for these experiments.

The train list should contain the identity and the file path, one line per utterance, as follows:
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```

The train list for VoxCeleb2 can be download from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt) and the
test list for VoxCeleb1 from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt).

#### Replicating the results from the paper

1. Model definitions
  - `VGG-M-40` in the paper is `VGGVox` in the code.
  - `Thin ResNet-34` is in the paper `ResNetSE34` in the code.
  - `Fast ResNet-34` is in the paper `ResNetSE34L` in the code.

2. For metric learning objectives, the batch size in the paper is `nPerSpeaker` multiplied by `batch_size` in the code. For the batch size of 800 in the paper, use `--nPerSpeaker 2 --batch_size 400`, `--nPerSpeaker 3 --batch_size 266`, etc.

3. The models have been trained with `--max_frames 200` and evaluated with `--max_frames 400`.

4. You can get a good balance between speed and performance using the configuration below.

```
python ./trainSpeakerNet.py --model ResNetSE34L --trainfunc angleproto --batch_size 400 --nPerSpeaker 2 --train_list train_list.txt --test_list test_list.txt 
```

#### Citation

Please cite the following if you make use of the code. Please see [here](References.md) for the full list of methods used in this trainer.

```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Interspeech},
  year={2020}
}
```

#### License
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
