# EEL6812 DeepFool Project

## Introduction
The purpose of this project was to re-implement the results from the paper *DeepFool: a simple and accurate method to fool deep neural networks* by Moosavi-Dezfooli, et al. [[1]](https://arxiv.org/pdf/1511.04599.pdf). Along with the DeepFool method being used in this project, the Fast Gradient Sign Method (FGSM) from *Explaining and Harnessing Adversarial Examples* by Goodfellow, et al. [[2]](https://arxiv.org/pdf/1412.6572.pdf) was used for comparison in this project. Models used in DeepFool were also replicated for this project based on their descriptions in the paper, and had their test error, adversarial inference, adversarial robustness evaluated. The DeepFool paper used fine-tuning and adversarial training on the models, which were also attempted to be re-implemented in this project.

## Datasets Used
The following datasets used for the project are [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), and [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/), which were used in the DeepFool paper. The training and validation sets of MNIST and CIFAR-10 were used for adversarial training and evaluation in this project, while the validation set of ILSVRC2012 was used only for adversarial evaluation. The specifications of the datasets are the following:
* MNIST Dataset
  * Has 28x28 grayscale images of hand-written digits
  * Has 60,000 training images and 10,000 validation images
  * 10,000 images were split for validation from training, and the original validation images were used for testing
  * Dataset was normalized for this project along with random horizontal flipping of images for training/validation
* CIFAR-10 Dataset
  * Has 32x32 RGB images of airplanes, automobiles, birds, cats, deers, dogs, frogs, horses, ships, and trucks
  * Has 50,000 training images and 10,000 validation images 
  * 10,000 images were split for validation from training, and the original validation images were used for testing
  * Dataset was normalized for this project along with random horizontal flipping and cropping of images for training/validation
* ILSVRC2012 Validation Dataset
  * Has 1000 classes and 50,000 validation images
  * Images were resized to size of 256 and cropped to size of 224 for GoogLeNet model
  * Dataset was normalized for this project

## Models Used
The models used for the project are the following:
* LeNet-5 Model for MNIST
  * Uses two convolutional layers with kernel size of 5 and stride of 1
  * Uses two max pooling layers with kernel size of 2 and stride of 2
  * Uses two linear hidden layers with sizes of 120 and 84
  * Uses ReLU for activation functions
* FC-500-150 Model for MNIST
  * Uses two linear hidden layers with sizes of 500 and 150
  * Uses ReLU for activation functions
* Network-In-Network Model [[3]](https://arxiv.org/pdf/1312.4400.pdf) for CIFAR-10
  * Uses the following GitHub [[4]](https://github.com/jiecaoyu/pytorch-nin-cifar10) implementation using PyTorch
  * This model does not use linear layers and replaces them with convolutional layers followed by global average pooling
    * Using a mini-network instead of linear layers helps reduce overfitting and improves accuracy
* LeNet-5 Model for CIFAR-10
  * Uses three convolutional layers with kernel size of 5, stride of 1, and padding of 2
  * Uses three max pooling layers with kernel size of 2 and stride of 2
  * Uses three linear hidden layers with sizes of 400, 120, and 84
  * Uses ReLU for activation functions
* GoogLeNet for ILSVRC2012
  * State of the art model that ranked high in object detection and classification categories for the [ILSVRC2014 challenge](https://image-net.org/challenges/LSVRC/2014/results)
  * Uses pre-trained weights that were trained on ImageNet
* CaffeNet for ILSVRC2012
  * Due to time constraints, this model was not implemented for the project

## Experiment Setup
The parameters used for the project are the following:
* The models were trained/evaluated using an Nvidia RTX 2070 Super graphics card
* Models were trained for 50 epochs using SGD with learning rate of 0.004 and momentum of 0.9
  * Network-In-Network model was trained for 100 epochs with learning rate of 0.1
  * Network-In-Network learning rate is divided by 10 every 80 epochs
* Models were fine-tuned and adversarially trained using half the learning rate for 50 epochs
  * Network-In-Network model was adversarially trained using 100 epochs
  * DeepFool adversarial training was cut at 5 epochs, as accuracy did not improve compared to FGSM
* DeepFool Parameters
  * The parameters used are based off the parameters the authors used in the DeepFool paper
  * Classes limit = 10
  * Overshoot = 0.02
  * Max Iterations = 50
* FGSM Parameters
  * The following epsilon values were used which resulted in a misclassification close to 90%
  * Epsilon = 0.6 (LeNet-5 for MNIST)
  * Epsilon = 0.2 (FC-500-150 for MNIST)
  * Epsilon = 0.2 (Network-In-Network for CIFAR-10)
  * Epsilon = 0.1 (LeNet-5 for CIFAR-10)

## Experiment Results
### Adversarial Inference & Robustness
![Table 1{captain=Table 1 - Adversarial Inference/Robustness of Project and DeepFool Paper}](/images/adversarial_inference.png)

### Adversarial Training
![Table 2{captain=Table 2 - Adversarial Training of Project and DeepFool Paper}](/images/adversarial_training.png)

### MNIST Adversarial Examples (FC-500-150)
![Figure 1{captain=Figure 1 - MNIST Adversarial Examples Generated using FC-500-150}](/images/examples_fc-500-150.png)

### CIFAR-10 Adversarial Examples (LeNet-5)
![Figure 2{captain=Figure 2 - CIFAR-10 Adversarial Examples Generated using LeNet-5}](/images/examples_lenet-5.png)

### ILSVRC2012 Adversarial Examples (GoogLeNet)
![Figure 3{captain=Figure 3 - ILSVRC2012 Adversarial Examples Generated using GoogLeNet}](/images/examples-googlenet.png)

## Conclusion
The adverarial inference/robustness results of the project were similar to the DeepFool paper. Inference was faster, as the hardware used for this project is faster than the hardware used for DeepFool. Adversarial robustness was not the same as the original results, as the epsilon values for FGSM were never mentioned in the paper. For adversarial training, DeepFool was found to not improve adversarial robustness, but this could be due to the fact that if the algorithm has access to the updated weights from the previous batch, it would always manage to find the minimal perturbations to have the classifier misclassify. For FGSM, having the model train on adversarials generated for each batch increases model robustness for MNIST and CIFAR-10. However for CIFAR-10, the accuracy for clean images decreases in exchange for adversarial robustness, but this is an expected result when doing adversarial training.

## Project Setup
The code for this project was ran on a Docker image [[5]](https://github.com/pman0214/docker_pytorch-jupyterlab) using PyTorch w/ CUDA on JupyterLab. Additional dependencies of this project include NumPy, Pandas, and Matplotlib, which are included within the Docker image.

## References
[1] S. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, *DeepFool: a simple and accurate method to fool deep neural networks*. arXiv, 2015. doi: 10.48550/ARXIV.1511.04599.

[2] I. Goodfellow, J. Shlens, and C. Szegedy, *Explaining and Harnessing Adversarial Examples*. arXiv, 2014. doi: 10.48550/ARXIV.1412.6572.

[3] M. Lin, Q. Chen, and S. Yan, *Network In Network*. arXiv, 2013. doi: 10.48550/ARXIV.1312.4400.

[4] https://github.com/jiecaoyu/pytorch-nin-cifar10

[5] https://github.com/pman0214/docker_pytorch-jupyterlab
