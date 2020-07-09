# Unsupervised_Text-to-Image_Synthesis
 Implementation of our PR 2020 paper:Unsupervised Text-to-Image Synthesis 
 
In this paper, we proposed to train one text-to-image synthesis model in one unsupervised manner, without resorting to any pairwise image-text data. To the best of our knowledge, this is the first attempt to tackle such an unsupervised text-to-image synthesis task. 

##Getting Started
Python 3.6+, Pytorch 1.2, torchvision 0.4, cuda10.0, at least 3.8GB GPU memory and other requirements. All codes are tested on Linux Distributions (centos 7), and other platforms have not been tested yet.

## Download resources.
1. Download `pretrains` from [OneDrive]() or [BaiduPan]() and then move the pretrains.zip to the assets directory and unzip this file.
2. Download `data` from [OneDrive]() or [BaiduPan]()  and then move the `data` to the assets directory.
3. Download `MSCOCO`  from the [COCO  site]() and extract the train2014.zip and  val2014.zip to `assets/data/coco/images`.

## Trainging
If you want to reproduce our model, the following pipeline is your need.
1. Train Concept-to-Sentence model.
```bash
sh scripts/con2sen_train.sh
```
2. Pseudo Image-Text pair construction.  
```bash
sh scripts/con2sen_infer.sh
```
3. Train DAMSM model.  
```bash
sh scripts/DAMSM.sh
```
4. Train Stage-I ut2i model(VCD).  
```bash
sh scripts/vcd.sh
```
5. Train Stage-II ut2i model(GSC).  
```bash
sh scripts/gsc.sh
```

## Evaluation
Our model adopts Evaluation [code]() in [ObjGAN](https://github.com/jamesli1618/Obj-GAN) 



