# DAM-Net
This repository is the official implementation of DAM-Net: Densely Attention Mechanism-Based Network for COVID-19 Detection in chest X-rays.

## Introduction
**DAM-Net** adaptively extracts spatial features of COVID-19 from the infected regions with various appearances and scales. It is composed of dense layers, channel attention layers, adaptive downsampling layer and label smoothing regularization loss function. Dense layers extract the patial features and the channel attention approach adaptively builds up the weights of major feature channels and suppresses the redundant feature representations. DAM-Net was trained using cross-entropy loss function along with label smoothing to limit hte effect of interclass similarity.

<p align="center">
  <img width="460" height="500" src="https://github.com/Zahid672/DAM-Net-Densely-Attention-Mechanism-Based-Network-for-COVID-19-detection/blob/main/images/model.PNG">
</p>
