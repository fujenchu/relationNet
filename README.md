# matchingNet
This repo contains the code for [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080) in Pytorch

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

pretrained weights for miniImageNet on 5-way 1-shot is [here](https://drive.google.com/file/d/1HPPLkSbPGgyzVfMUr3fyilhr0I3koZkK/view?usp=sharing)

### todos
* add FCE from my Tensorflow implementation
* merge training and testing

