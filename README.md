This repository contains the code for 'In defence of metric learning for speaker recognition.'

#### Dependencies
```
pip install -r requirements.txt
```

#### Example

```
python ./trainSpeakerNet.py --model ResNetSE34 --encoder SAP --trainfunc amsoftmax --optimizer adam --save_path data/exp1 --batch_size 200 --max_frames 200 --scale 30 --margin 0.3 --train_list /home/joon/train_list.txt --test_list /home/joon/test_list.txt --train_path /home/joon/voxceleb2 --test_path /home/joon/voxceleb1
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
