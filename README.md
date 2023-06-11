# Emotion Recognition with Facial Attention and Objective Activation Functions

This projected explored the effect of introducing channel and spatial attention mechanisms, namely  SEN-Net, ECA-Net, and CBAM to existing CNN vision-based models such as VGGNet, ResNet, and ResNetV2 to perform the Facial Emotion Recognition task. 

The project displayed that not only attention can significantly improve the performance of these models, but also that combining them with a different activation function can further help increase the performance of these models.

Paper for this project was published in the The 29th International Conference on Neural Information Processing (ICONIP 2022) [Paper Link](https://link.springer.com/chapter/10.1007/978-981-99-1642-9_43) [Original Disertation](docs\documents\Final-Disertation-Andrzej-Miskow.pdf) 

# Development

## Pre-Processing
### Face Detection 
This paper uses state-of-the-art facial detector built on top of YOLO framework [Refrence YOLO5 Face Detection].

YOLO [Refrence YOLO] architecture was chosen due to its efficient one-stage object detection capability that is comparable to the performances of a two-stage detectors whilst offering significantly better computational performance.

![YOLO 5 Face Output](https://github.com/AndrzejMiskow/FER-with-Attention-and-Objective-Activation-Functions/blob/main/docs/images/Yolo5Face_Output.png)

## CNN Models
### VGGNet
![VGGNet Architecture](https://github.com/AndrzejMiskow/FER-with-Attention-and-Objective-Activation-Functions/blob/main/docs/images/VGGNet_Architecture.png)

Authors in proposed the Visual Geometry Group Network (VGGNet) architecture. VGGNet showed that it is possible to increase the depth of a CNN through small-sized kernels. VGGNet showed significant improvement over prior architecture by increasing the depth of the network to 16–19 layers. 

### ResNet V1
![ResNet Architecture](https://github.com/AndrzejMiskow/FER-with-Attention-and-Objective-Activation-Functions/blob/main/docs/images/ResNet_Architecture.png)

Authors in devised the ResNet architecture, which introduced the concept of residual learning and proposed their computational block, the “residual” block. ResNet aimed at solving the issues found in deep CNN architectures: the vanishing gradient and the degradation problem.

### ResNet V2
After the release of ResNet, it was discovered that the degradation problem was still present when the depth of the network exceeded 200 layers. The degradation problem inspired the development of ResNet V2, which fully solved the issue of both the vanishing gradient and the degradation problem by implementing pre-activations in the residual blocks. The new version of ResNet increased accuracy for ultra-deep networks exceeding  1001 layers.

## Attention Modules
### SEN-Net
SENet (Squeeze-and-Excitation Network) incorporates a channel-wise attention mechanism into CNNs to recalibrate channel-wise feature responses adaptively. It introduces two additional operations called "squeeze" and "excitation." The squeeze operation aggregates spatial information globally, and the excitation operation models the interdependencies between channels. By explicitly emphasizing informative channels and suppressing less important ones, SENet enables CNNs to focus on more discriminative features.

![SEN-Net](docs\images\SEN-Net.jpg)

### ECA-Net
ECA-Net (Efficient Channel Attention Network) introduces the "Efficient Channel Attention" module, which aims to capture channel-wise dependencies effectively while minimizing computational overhead. It utilizes a 1D convolutional kernel to model the interdependencies between channels. By adaptively recalibrating channel-wise feature responses, ECA-Net can enhance the discriminative power of CNNs.

![SEN-Net](docs\images\ECA-net.jpg)

### CBAM

![CBAM-Module](docs\images\CBAM-Module.png)

CBAM (Convolutional Block Attention Module is a versatile attention module that combines both spatial and channel attention mechanisms. It includes two branches: one for spatial attention and the other for channel attention. The spatial attention branch captures the interdependencies between spatial locations within a feature map, while the channel attention branch models the interdependencies between channels. By adaptively attending to relevant spatial locations and informative channels, CBAM enhances the representational power of CNNs.

![CBAM-Blocks](docs\images\CBAM-Blcoks.gif)

# Results
## FER Datasets
Project was evaluted over 3 datasets of diffrent sizes. The small JAFFA dataset , medium sized CK+ dataset and large FER2013 dataset.

## Evaluation of CNN architectures with ELU on CK+, JAFFE and FER2013

| **Architecture** | **Parameters** | **CK+ Accuracy** | **JAFFE Accuracy** | **FER2013 Accuracy** |
|------------------|----------------|------------------|--------------------|----------------------|
| VGG-16           | 39.92M         | 87.91\%          | 64.44\%            | 60.66\%              |
| VGG-19           | **42.87M**     | **90.66**        | **68.89\%**        | **60.92\%**          |
| ResNet-50        | 23.49M         | 87.91\%          | **73.33\%**        | 58.61\%              |
| ResNet-101       | 42.46M         | **88.46%**       | 60.00\%            | 58.67\%              |
| ResNet-152       | **58.08M**     | 85.71\%          | 15.66\%            | **59.36\%**          |
| ResNetV2-50      | 23.48M         | 88.46\%          | **77.78\%**        | 58.72\%              |
| ResNetV2-101     | 42.44M         | 88.62\%          | 62.22\%            | 59.07\%              |
| ResNetV2-152     | **58.05M**     | **89.01\%**      | 66.67\%            | **59.40\%**          |


## CNNs with Different Attention Mechanisms

| Architecture | Param | CK+ Accuracy | JAFFE Accuracy | FER2013 Accuracy |
|---|---|---|---|---|
| VGG-16 | 39.92 M | 87.91\% | 64.44\% | 60.66\% |
| VGG-16 + SEN-Net | 39.95M | 88.46\% | 68.89\% | 63.05\% |
| VGG-16 + ECA-Net | 39.92M | 89.01\% | 73.33\% | 62.72\% |
| VGG-16 + CBAM | **39.95M** | **89.56\%** | **75.56\%** | **63.46\%** |
| VGG-19 | 42.87M | 90.66\% | 68.89\% | 60.92\% |
| VGG-19 + SEN-Net | 45.26M | 91.21\% | 73.33\% | 63.23\% |
| VGG-19 + ECA-Net | 45.23M | 91.76\% | 75.56\% | 63.49\% |
| VGG-19 + CBAM | **45.26M** | **92.31\% (↑ 1.65\%)** | **77.78\%** | **64.07\% (↑ 3.15\%)** |
| ResNet-50 | 23.49M | 87.91\% | 73.33\% | 58.61\% |
| ResNet-50 + SEN-Net | 26.02M | 89.01\% | 75.56\% | 58.84\% |
| ResNet-50 + ECA-Net | 23.65M | 90.11\% | 77.78\% | 59.73\% |
| ResNet-50 + CBAM | **26.02M** | **91.21\%** | **82.22\%** | **59.90\%** |
| ResNet-101 | 42.46M | 88.46\% | 60.00\% | 58.67\% |
| ResNet-101 + SEN-Net | 47.24M | 89.01\% | 68.89\% | 58.92\% |
| ResNet-101 + ECA-Net | 42.81M | 89.56\% | 73.33\% | 60.15\% |
| ResNet-101 + CBAM | **47.24M** | **90.11\%** | **75.56\%** | **60.92\%** |
| ResNet-152 | 58.08M | 85.71\% | 15.66\% | 59.36\% |
| ResNet-152 + SEN-Net | 64,71M | 88.46\% | 15.66\% | 59.73\% |
| ResNet-152 + ECA-Net | 58.60M | 89.56\% | 15.66\% | 60.92\% |
| ResNet-152 + CBAM | **64.71M** | **90.11\%** | **15.66\%** | **61.54\%** |
| ResNetV2-50 | 23.48M | 88.46\% | 77.78\% | 58.72\% |
| ResNetV2-50 + SEN-Net | 26.01M | 88.66\% | 82.22\% | 59.36\% |
| ResNetV2-50 + ECA-Net | 23.64M | 88.91\% | 82.22\% | 59.73\% |
| ResNetV2-50 + CBAM | **26.01M** | **89.01\%** | **84.44\%(↑ 6.55\%)** | **60.15\%** |
| ResNetV2-101 | 42.44M | 88.62\% | 62.22\% | 59.07\% |
| ResNetV2-101 + SEN-Net | 47,22M | 89.01\% | 68.89\% | 59.73\% |
| ResNetV2-101 + ECA-Net | 42.79M | 89.56\% | 70.83\% | 60.15\% |
| ResNetV2-101 + CBAM | **47.22M** | **90.66\%** | **73.33\%** | **60.92\%** |
| ResNetV2-152 | 58.05M | 89.01\% | 66.67\% | 59.40\% |
| ResNetV2-152 + SEN-Net | 64.68M | 89.56\% | 68.89\% | 60.72\% |
| ResNetV2-152 + ECA-Net | 58.57M | 89.82\% | 73.33\% | 61.54\% |
| ResNetV2-152 + CBAM | **64.69M** | **90.11\%** | **77.78\%** | **62.05\%** |

# Conclusions
This project aimed to classify the seven basic human emotions using deep learning models. The project focused on implementing and modifying popular CNN models such as VGGNet, ResNet, and ResNetV2. By replacing the internal activation function with ELU, the models effectively addressed the bias-shift problem and improved their performance.

The project also explored the concept of attention to refine extracted features and enhance model generalization. Three attention modules, SEN-Net, ECA-Net, and CBAM, were implemented to improve the CNNs' performance. SEN-Net and ECA-Net focused on channel attention, while CBAM utilized both channel and spatial attention. It was observed that CBAM achieved the most significant performance improvement, benefiting from attention in both aspects.

To verify the impact of attention on CNN performance, extensive experiments were conducted on three datasets of different sizes: CK+, JAFFE, and FER2013. Results demonstrated that each attention module outperformed the baseline models on all datasets, showcasing the generalization improvement. Notably, CBAM consistently achieved the best performance across all experiments. Overall, attention modules improved the baseline models by 1.65% for CK+, 6.55% for JAFFE, and 3.15% for FER2013, achieving accuracies of 92.31%, 84.44%, and 64.07%, respectively. The accuracy achieved on FER2013 even matched human performance.

In conclusion, this project successfully demonstrated the effectiveness of attention modules in enhancing the performance of CNNs for emotion recognition tasks. The findings highlight the importance of attention in refining feature representation and improving generalization across datasets of various sizes. By combining deep learning models with attention mechanisms, significant performance gains can be achieved in emotion recognition applications.

# Refrences
- YOLO
    - [Yolo5](https://github.com/ultralytics/yolov5)
    - [YOLO5Face](https://arxiv.org/abs/2105.12931)
- Activation Functions
    - [SELU](https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html) , [ELU](https://arxiv.org/abs/1511.07289)   
- CNN Architectures
    - [VGGNet](https://arxiv.org/abs/1409.1556) , [ResNetV1](https://arxiv.org/abs/1512.03385) , [ResNetV2](https://arxiv.org/abs/1603.05027) 
- Attention Modules
    - [SEN-Net](https://arxiv.org/pdf/1709.01507.pdf), [ECA-Net](https://arxiv.org/abs/1910.03151) and [CBAM](https://arxiv.org/abs/1807.06521)
- Datasets
    - [CK+](https://www.kaggle.com/datasets/shawon10/ckplus) , [JAFFE](https://zenodo.org/record/3451524) , [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
