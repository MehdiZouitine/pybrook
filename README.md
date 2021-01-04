# Pybrook : A tool to skull strip MRI images
<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/pybrook_logo.jpg?raw=true" alt="example0">
</p>


**Pybrook** is a python package designed for medical MRI preprocessing. Specifically Pybrook is designed to automatically extract the brain from MRI images. This package has a lack of (modern) tools to address this problem. By using several models (Resnet, Efficient net) Pybrook achieves an IOU score of **0.98**. 

## Data and training
The models are trained on the **Neurofeedback Skull-stripped (NFBS) repository**.
*It's a database of 125 T1-weighted anatomical MRI scans that are manually skull-stripped. In addition to aiding in the processing and analysis of the NFB dataset, NFBS provides researchers with gold standard training and testing data for developing machine learning algorithms.*

<p align="center">
  <img height="320px" src="https://github.com/MehdiZouitine/pybrook/blob/main/image/NFBS.png?raw=true" alt="=NFBS"
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
