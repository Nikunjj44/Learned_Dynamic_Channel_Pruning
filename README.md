# Learned Dynamic Channel Pruning

Deep generative models like Stable Diffusion perform exceptionally for image generation tasks. However, the computational costs require high memory consumption. Hence Learned Dynamic Channel Pruning is proposed that aims to reduce the per-step resource computation costs of the Stable Diffusion U-Net. This technique is based on the fact that the importance of channels vary widely such that some channels are redundant and can be removed. A good thing to note here is that the implementation performs dynamic channel pruning not static channel pruning where the channels are permanently removed. Whereas, the proposed approach aims to learn which channels are important and then dynamically turn pruning on or off during inference stage. Hence, this can be broken down into:  

1. Learned: Learning channel importance scores.  
2. Dynamic: Ability to turn pruning on or off during inference.
3. Channel Pruning: Using the channel importance scores, the channels in the U-Net’s convolution layers are pruned.  

In this implementation Learned Dynamic Channel Pruning is applied to **Stable Diffusion 1.5** model and the aim is to reduce peak memory consumption while at the same time preserving the image quality and speed of generation.

## Dataset Used  

The dataset used for the experimentation is **FFHQ (Flickr-Faces-HQ)** Dataset. It is a high-quality image dataset of human faces, maintained by NVIDIA and originally created as a benchmark for generative adversarial networks (GAN). This dataset contains 70,000 high quality images with a resolution of 1024 x 1024. This dataset is widely used in face generation and face editing applications. Furthermore, the dataset has a wide range of variation of age, gender and ethnicity in human faces. In addition to this, the dataset also has good coverage in terms of accessories like sunglasses, hats, jewellery etc. Hence, making it a good choice for our proposed experiments.

## Methodology  

3 key components are used:

1. **Channel Importance Scoring Module**  

This module provides each channel with an importance score. This score determines whether the channel will be enabled or disabled during inference. This is done with the help of sigmoid activation and binary thresholding. Firstly, the sigmoid activation helps to keep the importance scores within the range of [0, 1] and binary threshold is used with threshold value of 0.5. This basically classifies the channels into a state of enable and disable. If the importance scores are less than or equal to 0.5, then they are pruned.

Now, how exactly are the importance scores learned? This is done through gradient-based optimization during the training phase. The nn.Parameter wrapper makes the importance scores trainable parameters. Essentially
enabling them to be updated through backpropagation. Such that:
- During Forward Pass, the binary threshold is used as a decision boundary to classify the channels.
- During Backward Pass, the gradients flow through the sigmoid activation function.

2. **Pruning and Replacement Module** 

This module is divided into two stages; Firstly applying pruning to the convolution layers and secondly replacing the original U-Net with the pruned U-Net.  

The first stage involves wrapping of a prunable layer such that selective channel activation is achieved. How this works is that the input passes the convolution layer where the channel importance module generates a binary mask such that the pruned channels are multiplied by 0 and the active channels are multiplied by 1. This is a vital stage because of the fact that these zeroed channels do not contribute to the computations thereby helping in reducing the memory usage.

The second stage replaces the affected convolution layers in the U-Net with the Pruned U-Net. As not all blocks are suitable for pruning, we work only on the middle blocks because it provides low risk of image quality degradation while at the same time providing a good amount of redundant channels.

3. **Training Phase**  

During training two loss functions are used to optimize the importance scores:

- Reconstruction Loss

  When pruning a channel, if the image quality decreases the reconstruction loss increases the importance score of that channel. This is Mean Squared Error (MSE) Loss.

- Sparsity Loss

  This is L1 regularization that aims to penalise the importance scores by pushing them towards 0. Thereby encouraging pruning.

To summarise this, the model finds out which channels are redundant with the help of the above mentioned competing losses. Where the reconstruction loss preserves the channels that provide image quality and the sparsity loss encourages pruning. The optimal point provides the best memory reduction while at the same time preserving the image generation quality.


## Academic Context
Completed during my Graduate studies at **Nanyang Technological University (NTU), Singapore**.

* **Course:** Generative AI for Visual Synthesis
* **Semester:** AY 2025/2026, Semester 1
* **Grade Achieved:** A


## Usage
**Note for current/future NTU students:** While this repository is public, please ensure you adhere to NTU's Academic Integrity Policy. This is intended as a reference for my personal portfolio; using this code for your own graded assignments is strictly prohibited by the University.
