# Pybrook : A tool to skull strip MRI images :skull: :brain: :skull:
<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/pybrook_logo.jpg?raw=true" alt="example0">
</p>


**Pybrook** is a python package designed for medical MRI preprocessing. Specifically Pybrook is designed to automatically extract the brain from MRI images. This package aims to fill the lack of (modern) tools to respond to this problem.By using several models (Resnet, Efficient net) Pybrook achieves an IOU score of **0.98**. 

## Data 
The models are trained on the **Neurofeedback Skull-stripped (NFBS) repository**.
*It's a database of 125 T1-weighted anatomical MRI scans that are manually skull-stripped. In addition to aiding in the processing and analysis of the NFB dataset, NFBS provides researchers with gold standard training and testing data for developing machine learning algorithms.*

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/NFBS.png?raw=true" alt="=NFBS"
</p>

In order to be able to train the model it was necessary to extract from the MRIs all possible slices of the brain: **Axial, coronal and sagittal**. By doing this, our model is **trained on 2D images** and not directly on the 3D MRI. The script **data_cleaning.py** ensures this processing.

## Training

Our brain extractor is a set of several **Unet** models pre-trained on imagenet. Each model is trained with a **BCE with logit loss**.
And each model has a different encoder.

| Model | Encoder              | Pretrained data | Validation IOU | Source                               |
|-------|----------------------|-----------------|----------------|--------------------------------------|
| Unet  | resnet18             | imagenet        | 0.95           | https://arxiv.org/pdf/1512.03385.pdf |
| Unet  | timm-efficientnet-b4 | imagenet        | 0.965          | https://arxiv.org/pdf/1905.11946.pdf |
| Unet  | timm-resnest50d      | imagenet        | 0.976          | https://arxiv.org/pdf/2004.08955.pdf |

## Pre-processing and post processing

MRI segmentation may require pre-processing and post-processing. Indeed, the variability between the measuring devices and the type of MRI can lead to a huge variability in the pixel distribution. This results in poor segmentation (artefacts, holes). So in some cases it is advisable to use this preprocessing.

### Pre-processing

Preprocessing consists of **histogram matching**. The idea is that a new MRI image can have a pixel distribution that is very different from the distribution of the training set. That's why the idea is to match the distribution of the new image to a reference image drawn in the training set.


<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/matchgraph (1).png?raw=true" alt="example_matching">
</p>

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/histo_matching.png?raw=true" alt="example_matching">
</p>

### Post-processing

As said before, a different distribution can lead to many segmentation flaws. Two major defects: **Artifacts** and **holes**. To overcome this, **morphological operators** are applied on the binary image (the mask). Morphological operators are basic studies in the treatment of binary images and allow to ensure a lot of stain despite their simplicity.

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/morphoskullstrip.png?raw=true" alt="morphological">
</p>

## Skull stripping blend of SOTA models : 

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/example0.gif?raw=true" alt="example0">
</p>

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/example1.gif?raw=true" alt="example1">
</p>

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/example2.gif?raw=true" alt="example2">
</p>

### How to install ?
```
git clone
pip install -e .
```


### How to use it ?

```python
print(issou)
```
