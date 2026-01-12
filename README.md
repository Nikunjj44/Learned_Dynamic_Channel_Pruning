# Learned Dynamic Channel Pruning

Deep generative models like Stable Diffusion perform exceptionally for image generation tasks. However, the computational costs require high memory consumption. Hence Learned Dynamic Channel Pruning is proposed that aims to reduce the per-step resource computation costs of the Stable Diffusion U-Net. This technique is based on the fact that the importance of channels vary widely such that some channels are redundant and can be removed. A good thing to note here is that the implementation performs dynamic channel pruning not static channel pruning where the channels are permanently removed. Whereas, the proposed approach aims to learn which channels are important and then dynamically turn pruning on or off during inference stage. Hence, this can be broken down into:  

1. Learned: Learning channel importance scores.  
2. Dynamic: Ability to turn pruning on or off during inference.
3. Channel Pruning: Using the channel importance scores, the channels in the U-Net’s convolution layers are pruned.  

In this implementation Learned Dynamic Channel Pruning is applied to **Stable Diffusion 1.5** model and the aim is to reduce peak memory consumption while at the same time preserving the image quality and speed of generation.

## Tech Stack Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-0057B8?style=for-the-badge&logo=stablediffusion&logoColor=white)
![Generative AI](https://img.shields.io/badge/Generative%20AI-FF6F00?style=for-the-badge&logo=openai&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

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

## Experimental Setup

For the training setup 5,000 images were extracted from the FFHQ dataset. To this dataset we did a 90:10 split for training and validation set. Such that the training dataset consists of 4,500 images and the validation set consists of 500 images. The resolution of images in the training and validation set was set to 512 x 512. Training was performed in two ways – Once when one layer was pruned and second when 3 layers were pruned. Such that:  
- 1 layer configuration had 361,473 trainable parameters  
- 3 layer configuration had 1,084,419 trainable parameters

## Hyperparameter Tuning & Ablation Studies

The pruning is controlled by two important hyperparameters namely **target_sparsity** and **lambda_sparsity**. These two parameters are tested jointly. The parameter target_sparsity refers to the portion of channels to be pruned. For example; if the middle block has C channels and target sparsity is 0.4, then aim is to prune 40% of these C channels. The range tested during our experimentation is [0.1, 0.6] which refers to 40% – 60% pruning.

The parameter lambda_sparse refers to the weight coefficient (𝛌) for the sparsity loss term. The range tested during our experimentation is [0.003, 0.02]

<img width="545" height="264" alt="image" src="https://github.com/user-attachments/assets/ad26250a-2469-4578-b05d-ae73f0594175" />

## Results and Evaluation

<img width="545" height="343" alt="image" src="https://github.com/user-attachments/assets/9ad5d1d4-beab-4284-ab76-c28b97d62618" />  

Through Fig. 2. it can be concluded that there is no difference when increasing the number of trainable parameters in the middle block. To further validate this a standard test with target_sparsity = 0.3 and lambda_sparse = 0.01 on an increased number of epochs for both 1 layer configuration and 3 layer configuration were run.

<img width="545" height="82" alt="image" src="https://github.com/user-attachments/assets/480f3a30-b1e1-414e-80f2-c4fee6ab06de" />  

Table 3 successfully validated the observation and confirms that apart from some minor difference in TFLOP reduction, there is not much impact of increasing the number of training parameters. Secondly, the analysis of the impact of the parameters target_sparsity and lambda_sparse on memory reduction and image quality was performed.

<img width="545" height="295" alt="image" src="https://github.com/user-attachments/assets/e027e2cd-4548-42b6-8195-bbc497c915e8" />  

From the above chart it can be seen that as target sparsity is increased the memory reduction capacity also increases. Simultaneously as lambda increases the memory reduction capacity increases as well. Hence it can be concluded that as the target sparsity and lambda values are increased, the memory reduction also increases significantly. Thereby, reducing memory consumption.

To test this side by side comparison of generated images is performed (as shown in Fig. 4). If we look at the prompt “a smiling face” there is significant improvement in finer details like the teeth area. For which at higher target sparsity and lambda there are visible extra white blobs near the teeth area that seem out of place. Whereas, for less target sparsity and lambda the image is much more refined and looks good. If we consider edge cases and provide a short prompt with a widely ranged attribute like “a beautiful face”. Then to get even better image quality and generation the target sparsity and lambda need to be reduced further. As seen from Fig 4. at target sparsity = 0.1 and lambda = 0.003 a high quality accurate image is generated. Therefore, we can conclude that by reducing the target sparsity and lambda values, the image quality can be improved significantly.

<img width="652" height="699" alt="image" src="https://github.com/user-attachments/assets/56c0c813-9e1f-43d0-ae06-d79d1e390fc0" />  

Therefore, to summarise the impact of hyperparameters:
- When the values of hyperparameters are increased, memory reduction is prioritised.
- When the values of hyperparameters are decreased, image quality is prioritised.

Furthermore, it is concluded that there is just some slight variations in speed (ranging from 0.98x – 0.99x) and and TFLOP reduction (-1.1% – -1.8%) showing that they are not significantly impacted and the hyperparameters do not help in increasing the speed or reducing TFLOPs.

## High Image Quality

As observed from the above results -- a low value for target sparsity (0.1) and lambda (0.003) provides us with high quality image generation. To validate this further, the model was trained on much more epochs.

<img width="652" height="621" alt="image" src="https://github.com/user-attachments/assets/6920708d-2dfd-4fc8-b0b9-1704064d9dea" />  

Therefore, when prioritising image quality we see some minor memory reduction of about 8% along with a negative impact of TFLOPs reduction of about 1.9%. Though there is slight reduction in memory consumption the main objective of this particular implementation is to analyse image quality.

<img width="652" height="647" alt="image" src="https://github.com/user-attachments/assets/060c3456-1217-4c57-b0db-fa97b72c4248" />  

We can clearly see that in these settings the image generation works very well and provides high quality generated images. These images are also very varied and can properly differentiate between a person’s age and gender. Moreover, the model works well by applying accessories like sunglasses as well. Furthermore, we tested if there is variation in similar prompts by using prompts like “a old person’s face” and “an elderly face” – we observed good variation in this as well where the generated person images had different facial structures and expressions. Finally, we gave prompts that captured fine grained details like “a face with makeup” and “a clean-shaven face” and these attributes were captured accurately as well.

## Ideal Solution

The main objective of this modification is to reduce memory consumption while at the same time preserving image quality. Hence, the ideal solution should be able to handle both. And to do so we trained the model on an increased number of epochs with the ideal parameters (Target Intensity = 0.3, Lambda = 0.01).
This model yields the following results:  
- Estimated Memory Reduction = ~24%
- Estimated TFLOP Reduction = ~-1.1%
- Good quality images

<img width="652" height="491" alt="image" src="https://github.com/user-attachments/assets/2d026949-3f7a-40b0-8e81-f5c1b8b441f2" />

These settings help reduce the memory consumption by about 24% while maintaining image quality. And is viewed as a balanced approach as it provides high memory reduction capabilities while maintaining good image quality.

## Strengths and Limitations

This modification shows many strengths that makes it an effective approach to model compression a diffusion model. The **architecture is simple and not complex** because it only affects the middle block making it easier to incorporate into existing stable diffusion models. The **dynamic pruning** allows for flexibility during the run time, instead of saving and maintaining frequent checkpoints. And most importantly, this modification yields **substantial memory reduction** (8% - 48%) while **maintaining inference speed** (0.98x - 1.00x) enabling stable deployment on devices with restricted resources. The **tunability nature** of this approach to prefer memory efficiency or quality is a major positive as well. 

Despite these strengths, there are some limitations that exist. The middle block only represents **only one component** of the U-Net architecture; other blocks like upsampling and downsampling blocks may contain further redundant channels which could yield further compression. There is some **negative TFLOP reduction** (-0.5% – -1.8%) present which indicates that the importance scoring mechanism leads to some overhead expenses. The overall evaluation can be improved further by incorporating detailed prompts (to showcase generalization) and incorporating quantitative metrics (like FID and CLIP scores) for evaluating image quality.

## Academic Context
Completed during my Graduate studies at **Nanyang Technological University (NTU), Singapore**.

* **Course:** Generative AI for Visual Synthesis
* **Semester:** AY 2025/2026, Semester 1
* **Grade Achieved:** A


## Usage
**Note for current/future NTU students:** While this repository is public, please ensure you adhere to NTU's Academic Integrity Policy. This is intended as a reference for my personal portfolio; using this code for your own graded assignments is strictly prohibited by the University.
