<p align="center">
  <a href="" rel="noopener">
  <img height=280px src="https://i.imgur.com/3XmtaFJ.png" alt="Project logo"></a>
</p>

<h1 align="center">A novel ordinal deep learning classification framework for lung ultrasound Covid-19 ranking</br><sub></sub></h1>

<div align="center">
   
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorflow)
![Tensorflow](https://img.shields.io/badge/tensorflow-2.11.0-orange)
![UNIVPM](https://img.shields.io/badge/organization-UNIVPM-red)
![License](https://img.shields.io/github/license/edoardo-conti/iclus-ordinal-classification?color=green)
</div>

---

## 📝 Table of Contents
- [Abstract](#abstract)
- [Nominal architecture: ResNet18](#resnet18)
- [Ordinal architecture #1: CLM](#clm)
- [Ordinal architecture #2: OBD](#obd)
- [Results](#results)

## 📋 Abstract <a name="abstract"></a>
In the context of medical diagnosis, especially in the Covid-19 pneumonia, medical imaging plays an important role. Lung ultrasound (LUS) comes out as a valuable diagnostic technology for the early detection of pulmonary pathologies. This research focuses on how Deep Learning (DL) can contribute to the automation of medical diagnosis, with a specific emphasis on classifying LUS frames using Convolutional Neural Networks (CNN). The University of Trento provides the ICLUS-DB, a lung ultrasound dataset. It includes a 4-level scoring system reflecting the severity ranking of Covid-19 pneumonia, highlighting the intrinsic ordinal nature of LUS data. This aspect has sparked interest in investigating the possibility of achieving more accurate results by leveraging the ordinal nature of the data through the implementation of specific methodologies to optimize the classification of LUS frames. In contrast, the State-of-the-art is oriented towards solving this problem with nominal classification methods, not penalizing errors between distant classes, a relevant aspect for medical implications. In order to fill this gap in the literature, the main goal is to compare a baseline approach, represented by the ResNet18 CNN, with two distinct ordinal approaches: the Ordinal Binary Decomposition (OBD), based on the decomposition of the ordinal problem into a set of binary tasks, and the Cumulative Link Model
(CLM), a probabilistic method to predict the probabilities of groups of contiguous categories. Both ordinal approaches use ResNet18 as a feature extractor and exploit dedicated loss functions, including Quadratic Weighted Kappa (QWK). Furthermore, as traditional stratified holdout methods proved insufficient for obtaining reliable results, a robust evaluation framework was implemented. The design of a dedicated Cross-Validation (CV) procedure for ICLUS-DB is a valuable contribution, addressing the challenges of its nature and yielding generalized and robust results. The results confirm the positive contribution of ordinal approaches, especially in ordinal metrics, highlighting a significant increase in average values across all aspects, particularly in Accuracy 1-Off (up to 96.6%), QWK index (up to 73.1%), and Spearman’s coefficient (up to 74.4%). The analysis of ROC curves, confusion matrices, and saliency maps underline the advantage of ordinal approaches in capturing pathological details in LUS frames. The ablation study, conducted on all components of the Deep Neural Network (DNN), provides further insights, demonstrating the effectiveness of the proposed components.


## 📊 ResNet18 <a name="resnet18"></a>
![resnet18](https://i.imgur.com/RabTt3M.png)

## 📊 CLM <a name="clm"></a>
![clm](https://i.imgur.com/c28AAPE.png)

## 📊 OBD <a name="obd"></a>
![obd](https://i.imgur.com/mE0XUbo.png)

## 🔖 Results <a name = "results"></a>
<img src="https://i.imgur.com/FdzSzKo.png" />

