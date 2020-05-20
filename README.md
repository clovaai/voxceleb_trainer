## VoxCeleb trainer

This repository contains the training code for 'In defence of metric learning for speaker recognition.'

#### Dependencies
```
pip install -r requirements.txt
```

#### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training.

```
python ./dataprep.py --save_path /home/joon/voxceleb --download --user USERNAME --password PASSWORD 
python ./dataprep.py --save_path /home/joon/voxceleb --extract
python ./dataprep.py --save_path /home/joon/voxceleb --convert
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

#### Training example

```
python ./trainSpeakerNet.py --model ResNetSE34 --encoder SAP --trainfunc amsoftmax --optimizer adam --save_path data/exp1 --batch_size 200 --max_frames 200 --scale 30 --margin 0.3 --train_list /home/joon/voxceleb/train_list.txt --test_list /home/joon/voxceleb/test_list.txt --train_path /home/joon/voxceleb/voxceleb2 --test_path /home/joon/voxceleb/voxceleb1
```

#### Pretrained model

A pretrained model can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/models/baseline_lite_ap.model).

You can check that the following script returns: `EER 2.2322`.

```
python ./trainSpeakerNet.py --eval --model ResNetSE34L --trainfunc angleproto --save_path data/test --max_frames 300 --test_list /home/joon/voxceleb/test_list.txt --test_path /home/joon/voxceleb/voxceleb1 --initial_model baseline_lite_ap.model
```

#### Implemented loss functions
```
Softmax (softmax)
AM-Softmax (amsoftmax)
AAM-Softmax (aamsoftmax)
GE2E (ge2e)
Prototypical (proto)
Triplet (triplet)
Contrastive (contrastive)
Angular Prototypical (angleproto)
```

#### Implemented models and encoders
```
ResNetSE34 (SAP)
ResNetSE34L (SAP)
VGGVox40 (SAP, TAP, MAX)
```

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

2. For metric learning objectives, the batch size in the paper is `nSpeakers` multiplied by `batch_size` in the code. For the batch size of 800 in the paper, use `--nSpeakers 2 --batch_size 400`, `--nSpeakers 3 --batch_size 266`, etc.

3. The models have been trained with `--max_frames 200` and evaluated with `--max_frames 400`.

4. You can get a good balance between speed and performance using the configuration below.

```
python ./trainSpeakerNet.py --model ResNetSE34L --trainfunc angleproto --batch_size 400 --nSpeakers 2 --train_list /home/joon/voxceleb/train_list.txt --test_list /home/joon/voxceleb/test_list.txt --train_path /home/joon/voxceleb/voxceleb2 --test_path /home/joon/voxceleb/voxceleb1
```

#### Citation

Please cite the following if you make use of the code.

```
@article{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  journal={arXiv preprint arXiv:2003.11982},
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
