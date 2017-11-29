# relationNet
This repo contains the code for [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/pdf/1711.06025.pdf) in Pytorch

-------------------------------------
## Usage

### datasets
Please contact the authors for dataset (either `miniImageNet` from Vinyals or `new miniImageNet` from Ravi)

### training
```
python demo.py --gpuid 0
```
feel free to modify `config.py` to change arguments

### weights

pretrained weights for miniImageNet on 5-way 1-shot is [here]()

### todos
* add ResNet version.
* merge training and testing

### results
|settings|5w1s acc|
|----|----|
|84x84 conv w padding| 45.8%
|80x80 randCrop| 49.0%

