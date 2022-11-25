# DAM-Net
This repository is the official implementation of DAM-Net: Densely Attention Mechanism-Based Network for COVID-19 Detection in chest X-rays.

## Introduction
**DAM-Net** adaptively extracts spatial features of COVID-19 from the infected regions with various appearances and scales. It is composed of dense layers, channel attention layers, adaptive downsampling layer and label smoothing regularization loss function. Dense layers extract the patial features and the channel attention approach adaptively builds up the weights of major feature channels and suppresses the redundant feature representations. DAM-Net was trained using cross-entropy loss function along with label smoothing to limit hte effect of interclass similarity.

<p align="center">
  <img width="460" height="500" src="https://github.com/Zahid672/DAM-Net-Densely-Attention-Mechanism-Based-Network-for-COVID-19-detection/blob/main/images/model.PNG">
</p>

## Results on COVIDX dataset
The results achieved on covid-x dataset are shown below, we have also compared our results with other other existing techniques.

<p align="center">
  <img width="600" height="300" src=https://github.com/Zahid672/DAM-Net-Densely-Attention-Mechanism-Based-Network-for-COVID-19-detection/blob/main/images/comp_table.PNG>
</p>

## Ablation results
To show how each part of our model effects the results we have performed ablation study by performing experiments by removing specific parts of our model, these results are shown below.

<p align="center">
  <img width="300" height="300" src=https://github.com/Zahid672/DAM-Net-Densely-Attention-Mechanism-Based-Network-for-COVID-19-detection/blob/main/images/ablation.PNG>
</p>

