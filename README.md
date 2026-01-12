# Learned Dynamic Channel Pruning

Deep generative models like Stable Diffusion perform exceptionally for image generation tasks. However, the computational costs require high memory consumption. Hence Learned Dynamic Channel Pruning is proposed that aims to reduce the per-step resource computation costs of the Stable Diffusion U-Net. This technique is based on the fact that the importance of channels vary widely such that some channels are redundant and can be removed. A good thing to note here is that the implementation performs dynamic channel pruning not static channel pruning where the channels are permanently removed. Whereas, the proposed approach aims to learn which channels are important and then dynamically turn pruning on or off during inference stage. Hence, this can be broken down into:  

1. Learned: Learning channel importance scores.  
2. Dynamic: Ability to turn pruning on or off during inference.
3. Channel Pruning: Using the channel importance scores, the channels in the U-Net’s convolution layers are pruned.  

In this implementation Learned Dynamic Channel Pruning is applied to **Stable Diffusion 1.5** model and the aim is to reduce peak memory consumption while at the same time preserving the image quality and speed of generation.

## Dataset Used  

The dataset used for the experimentation is **FFHQ (Flickr-Faces-HQ)** Dataset. It is a high-quality image dataset of human faces, maintained by NVIDIA and originally created as a benchmark for generative adversarial networks (GAN). This dataset contains 70,000 high quality images with a resolution of 1024 x 1024. This dataset is widely used in face generation and face editing applications. Furthermore, the dataset has a wide range of variation of age, gender and ethnicity in human faces. In addition to this, the dataset also has good coverage in terms of accessories like sunglasses, hats, jewellery etc. Hence, making it a good choice for our proposed experiments.

## Methodology  




## Academic Context
Completed during my Graduate studies at **Nanyang Technological University (NTU), Singapore**.

* **Course:** Generative AI for Visual Synthesis
* **Semester:** AY 2025/2026, Semester 1
* **Grade Achieved:** A


## Usage
**Note for current/future NTU students:** While this repository is public, please ensure you adhere to NTU's Academic Integrity Policy. This is intended as a reference for my personal portfolio; using this code for your own graded assignments is strictly prohibited by the University.
