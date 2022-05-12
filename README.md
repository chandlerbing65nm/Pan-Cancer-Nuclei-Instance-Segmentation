<p align="center">
  <img 
    src="https://raw.githubusercontent.com/chandlerbing65nm/Pan-Cancer-Nuclei-Instance-Segmentation/main/docs/banner.png?raw=true"
  >
</p>

<div align="center">

  <a href="">![last commit](https://img.shields.io/github/last-commit/chandlerbing65nm/Pan-Cancer-Nuclei-Instance-Segmentation)</a>
  <a href="">![repo size](https://img.shields.io/github/repo-size/chandlerbing65nm/Pan-Cancer-Nuclei-Instance-Segmentation)</a>
  <a href="">![watchers](https://img.shields.io/github/watchers/chandlerbing65nm/Pan-Cancer-Nuclei-Instance-Segmentation?style=social)</a>

</div>

<h4 align="center">Instance Segmentation of Different Types of Cancer Cells.</h4>


<p align="center">
  <img 
    src="https://raw.githubusercontent.com/chandlerbing65nm/Pan-Cancer-Nuclei-Instance-Segmentation/main/docs/demo.gif?raw=true"
  >
</p>

---

# About the Project
![personal](https://img.shields.io/badge/project-chandlertimmdoloriel-red?style=for-the-badge&logo=appveyor)

## Dataset

Thie dataset that I used is known as PanNuke (Pan-Cancer Nuclei), it contains semi automatically generated nuclei instance segmentation and classification images with exhaustive nuclei labels across 19 different tissue types. The dataset consists of 481 visual fields, of which 312 are randomly sampled from more than 20K whole slide images at different magnifications, from multiple data sources.

In total the dataset contains 205,343 labeled nuclei, each with an instance segmentation mask. Models trained on PanNuke can aid in whole slide image tissue type segmentation, and generalize to new tissues.

<p align="center">
  <img 
    src="https://github.com/chandlerbing65nm/Pan-Cancer-Nuclei-Instance-Segmentation/blob/main/docs/pannuke-dataset.png?raw=true"
  >
</p>

As can be seen from the image above, the 19 tissue names and their corresponding distribution to the whole dataset are shown below:

<p align="center">
  <img 
    src="https://github.com/chandlerbing65nm/Pan-Cancer-Nuclei-Instance-Segmentation/blob/main/docs/distri.jpg?raw=true"
  >
</p>

There are five nuclei types for each tissue type in PanNuke dataset, these are:

```
1. neoplastic
2. inflammatory
3. softtissue
4. dead
5. epithelial
```

The part-1 of the dataset that I used is hosted in kaggle, it is available [here](https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification). Also, the notebook that I used to analyze the dataset is available [here](https://www.kaggle.com/code/chandlertimm/pan-cancer-nuclei-data-analysis).

## Model
