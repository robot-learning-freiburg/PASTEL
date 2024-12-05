# PASTEL
[**arXiv**](https://arxiv.org/abs/2405.19035) | [**IEEE Xplore**](https://ieeexplore.ieee.org/document/10766616) | [**Website**](http://pastel.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/fC4vapiJTb8)

This repository is the official implementation of the paper:

> **A Good Foundation is Worth Many Labels: Label-Efficient Panoptic Segmentation**
>
> [Niclas Vödisch](https://vniclas.github.io/)&ast;, [Kürsat Petek](http://www2.informatik.uni-freiburg.de/~petek/)&ast;, [Markus Käppeler](https://rl.uni-freiburg.de/people/kaeppelm)&ast;, [Abhinav Valada](https://rl.uni-freiburg.de/people/valada), and [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/). <br>
> &ast;Equal contribution. <br> 
> 
> *IEEE Robotics and Automation Letters*, vol. 10, issue 1, pp. 216-223, January 2025

<p align="center">
  <img src="./assets/pastel_teaser.png" alt="Overview of PASTEL approach" width="800" />
</p>

If you find our work useful, please consider citing our paper:
```
@article{voedisch2025pastel,
  author={Vödisch, Niclas and Petek, Kürsat and Käppeler, Markus and Valada, Abhinav and Burgard, Wolfram},
  journal={IEEE Robotics and Automation Letters}, 
  title={A Good Foundation is Worth Many Labels: Label-Efficient Panoptic Segmentation}, 
  year={2025},
  volume={10},
  number={1},
  pages={216-223},
}
```

**Make sure to also check out our previous work on this topic:** [**SPINO**](https://github.com/robot-learning-freiburg/SPINO).

## 📔 Abstract

A key challenge for the widespread application of learning-based models for robotic perception is to significantly reduce the required amount of annotated training data while achieving accurate predictions. This is essential not only to decrease operating costs but also to speed up deployment time. In this work, we address this challenge for PAnoptic SegmenTation with fEw Labels (PASTEL) by exploiting the groundwork paved by visual foundation models. We leverage descriptive image features from such a model to train two lightweight network heads for semantic segmentation and object boundary detection, using very few annotated training samples. We then merge their predictions via a novel fusion module that yields panoptic maps based on normalized cut. To further enhance the performance, we utilize self-training on unlabeled images selected by a feature-driven similarity scheme. We underline the relevance of our approach by employing PASTEL to important robot perception use cases from autonomous driving and agricultural robotics. In extensive experiments, we demonstrate that PASTEL significantly outperforms previous methods for label-efficient segmentation even when using fewer annotation.

## 👩‍💻 Code

### 🏗 Setup

#### ⚙️ Installation

1. Create conda environment: `conda create --name pastel python=3.8`
2. Activate environment: `conda activate pastel`
3. Install dependencies: `pip install -r requirements.txt`
4. Install torch, torchvision and cuda: `pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html`

#### 💻 Development

1. Install [pre-commit](https://pre-commit.com/) githook scripts: `pre-commit install`
2. Upgrade isort to 5.12.0: `pip install isort`
3. Update [pre-commit]: `pre-commit autoupdate`
Linter ([pylint](https://pypi.org/project/pylint/)) and formatter ([yapf](https://github.com/google/yapf), [iSort](https://github.com/PyCQA/isort)) settings can be set in [pyproject.toml](pyproject.toml).

### 🎨 Running PASTEL

Generating pseudo-labels with PASTEL involves three steps:
1. Train the semantic segmentation module.
2. Train the boundary estimation module.
3. Generate pseudo-labels using the fusion module.

For Cityscapes, an exemplary execution would look like this:
```bash
conda activate pastel
python semantic_fine_tuning.py fit --trainer.devices [0] --config configs/cityscapes_semantics.yaml
python boundary_fine_tuning.py fit --trainer.devices [0] --config configs/cityscapes_boundary.yaml
python instance_clustering.py test --trainer.devices [0,1,2,3] --config configs/cityscapes_instance_ncut.yaml
```

We provide configuration files for each step of all datasets in the `configs` folder. Please make sure to double-check the paths to the datasets and the pretrained weights.

### 🏋️ Pre-trained weights

We provide the following pre-trained weights:
- Cityscapes:
  - [Semantic segmentation](http://pastel.cs.uni-freiburg.de/downloads/semantic_cityscapes.ckpt)
  - [Boundary estimation](http://pastel.cs.uni-freiburg.de/downloads/boundary_cityscapes.ckpt)
  - [Semantic segmentation - ViT-L/14](http://pastel.cs.uni-freiburg.de/downloads/semantic_cityscapes_vitl_100.ckpt)
  - [Boundary estimation - ViT-L/14](http://pastel.cs.uni-freiburg.de/downloads/boundary_cityscapes_vitl_100.ckpt)
- PASCAL VOC:
  - [Semantic segmentation](http://pastel.cs.uni-freiburg.de/downloads/semantic_pascalvoc.ckpt)
  - [Boundary estimation](http://pastel.cs.uni-freiburg.de/downloads/boundary_pascalvoc.ckpt)
  - [Semantic segmentation - ViT-L/14](http://pastel.cs.uni-freiburg.de/downloads/semantic_pascalvoc_vitl_92.ckpt)
  - [Boundary estimation - ViT-L/14](http://pastel.cs.uni-freiburg.de/downloads/boundary_pascalvoc_vitl_92.ckpt)
- PhenoBench:
  - [Semantic segmentation](http://pastel.cs.uni-freiburg.de/downloads/semantic_phenobench.ckpt)
  - [Boundary estimation](http://pastel.cs.uni-freiburg.de/downloads/boundary_phenobench.ckpt)

> &#x26a0;&#xfe0f; If your browser blocks the download, right-click on the link and copy the address to download the file manually.

### 💾 Datasets

#### [Cityscapes](https://www.cityscapes-dataset.com/)

Download the following files:
- leftImg8bit_sequence_trainvaltest.zip (324GB)
- gtFine_trainvaltest.zip (241MB)
- camera_trainvaltest.zip (2MB)

After extraction, one should obtain the following file structure:
```
── data/cityscapes
   ├── camera
   │    └── ...
   ├── gtFine
   │    └── ...
   └── leftImg8bit_sequence
        └── ...
```

#### [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

- We use the 2012 challenge plus the SBD extension.
- Upon execution, the files should be automatically downloaded from [torchvision](https://pytorch.org/vision/main/datasets.html).

Afterward, one should obtain the following file structure:
```
── data/pascal_voc
   ├── SBD
   │    └── ...
   └── VOCdevkit/VOC2012
        └── ...
```

#### [PhenoBench](https://www.phenobench.org/)

- We use the leaf instance segmentation challenge.
- Please download the dataset from the official website.

After extraction, one should obtain the following file structure:
```
── data/phenobench
   ├── test
   │    └── images
   ├── train
   │    ├── images
   │    ├── leaf_instances
   │    ├── leaf_visibility
   │    ├── plant_instances
   │    ├── plant_visibility
   │    └── semantics
   └── val
        ├── images
        ├── leaf_instances
        ├── leaf_visibility
        ├── plant_instances
        ├── plant_visibility
        └── semantics
```


## 👩‍⚖️  License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

This work was funded by the German Research Foundation (DFG) Emmy Noether Program grant No 468878300.
<br><br>
<p float="left">
  <a href="https://www.dfg.de/en/research_funding/programmes/individual/emmy_noether/index.html"><img src="./assets/dfg_logo.png" alt="DFG logo" height="100"/></a>
</p>