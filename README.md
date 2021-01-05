# Pybrook : A tool to skull strip MRI images :skull: :brain: :skull:
<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/pybrook_logo.jpg?raw=true" alt="example0">
</p>


**Pybrook** is a python package designed for medical MRI preprocessing. Specifically Pybrook is designed to automatically extract the brain from MRI images. This package aims to fill the lack of (modern) tools to respond to this problem.By using several models (Resnet, Efficient net) Pybrook achieves an IOU score of **0.98** in cross-validation. 

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

| Model | Encoder              | Pretrained data | Cross-validation IOU | Source                               | Available |
|-------|----------------------|-----------------|----------------------|--------------------------------------|-----------|
| Unet  | resnet18             | imagenet        | 0.95 ± 0.0141        | https://arxiv.org/pdf/1512.03385.pdf | yes       |
| Unet  | timm-resnest50d      | imagenet        | 0.965 ± 0.0116              | https://arxiv.org/pdf/1905.11946.pdf | no        |
| Unet  | timm-efficientnet-b4 | imagenet        | 0.976 ± 0.0263            | https://arxiv.org/pdf/2004.08955.pdf | yes       |

<p align="center">
  <img height="600px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/stack_model.png?raw=true" alt="stack">
</p>
## Pre-processing and post processing

MRI segmentation may require pre-processing and post-processing. Indeed, the variability between the measuring devices and the type of MRI can lead to a huge variability in the pixel distribution. This results in poor segmentation (artefacts, holes). So in some cases it is advisable to use this preprocessing.

### Pre-processing

Preprocessing consists of **histogram matching**. The idea is that a new MRI image can have a pixel distribution that is very different from the distribution of the training set. That's why the idea is to match the distribution of the new image to a reference image drawn in the training set.


<p align="center">
  <img height="250px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/matchgraph (1).png?raw=true" alt="example_matching">
</p>

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/histo_matching.png?raw=true" alt="example_matching">
</p>

### Post-processing

As said before, a different distribution can lead to many segmentation flaws. Two major defects: **Artifacts** and **holes**. To overcome this, **morphological operators** are applied on the binary image (the mask). Morphological operators are basic studies in the treatment of binary images and allow to ensure a lot of stain despite their simplicity.

<p align="center">
  <img height="260px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/matchgraph (2).png?raw=true" alt="morphological">
</p>

## Results: 

### Coronal slice
<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/example0.gif?raw=true" alt="example0">
</p>

### Axial slice
<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/example1.gif?raw=true" alt="example1">
</p>

### Sagitale slice
<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/example2.gif?raw=true" alt="example2">
</p>

### How to install ?
```
git clone https://github.com/MehdiZouitine/pybrook
cd pybrook
pip install -e .
```


### How to use it ?

```python
import nibabel as nib # IRM image package
from stripper.inference import Brook

mask_path = "../example/sub-A00028185_ses-NFB3_T1w_brainmask.nii.gz" # path to IRM ground truth
data_path = "../example/sub-A00028185_ses-NFB3_T1w.nii.gz" # path to IRM data (skull + brain)

data = nib.load(data_path).get_fdata() # Convert IRM format to numpy 3D tensor
label = nib.load(mask_path).get_fdata() # Convert IRM format to numpy 3D tensor

stripper = Brook() # Create an instance of brook

example_slice = data[120,:,:] # Get on slice of tensor 

brain,mask = stripper.strip(example_slice,pre_process=False,post_process=False) # Extract brain from the skull
```
