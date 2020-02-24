# Welcome to the Neural-Network-Translator-Repository

<p align="center"><img src="https://github.com/pg020196/Neural-Network-Translator/blob/Sprint06/.github/resources/readme/nnt_icon.png" alt="Sketch of the translators function" width="70%"/></p>

## Project Motivation

Nowadays, neural networks are often modeled and trained with powerful frameworks, such as Microsoft Cognitive Toolkit, Keras or TensorFlow. While the training of these neural networks is often performed on very powerful hardware, the hardware of the final product on which the models will be executed later, however, is usually much less powerful due to the cost-driven requirements. Particularly in the embedded area, the question, therefore, arises how the trained models can be implemented on the embedded systems used in practice. This project provides a code translator, which makes it possible to translate a neural network model into native code for a specified platform of an embedded device. 

Basically, the implemented process is to translate a given neural network model containing information about its structure and data, such as weights and bias values, to a general intermediate file format. This file format simply is a JSON-file which holds all relevant information of the neural network model. Using the given JSON-file as an intermediate layer allows to split the translation process into two subprocesses. The frontent handles the information extraction from the neural network model and the parsing of these information in the general file format. After having populated the intermediate file completely, the backend can translate the information given in the intermediate file format into platform specific code. Thereby, the implementation of each the frontend and the backend is handled in a plug-in-fashion which means that you can add specific plug-ins for every desired framework. The following figure shows the concept in a compact way:

<p align="center"><img src="https://github.com/pg020196/Neural-Network-Translator/blob/Sprint06/.github/resources/wiki/home_functionality_sketch.png" alt="Sketch of the translators function" width="70%"/></p>

## Supported Components

Currently, the project supports the following configurations:

### Supported Neural Network Structures

| Neural Network Structure     | Status |
| :--------------------------- | :----: |
| Feed Forward Neural Network  |   ✔️    |
| Recurrent Neural Network     |        |
| Convolutional Neural Network |        |

### Supported Frameworks (Frontend)

| Neural Network Structure   |  Status  |
| :------------------------- | :------: |
| Keras                      |    ✔️     |
| Microsoft Cognitive Toolkit (CNTK) | :candy:* |
| PyTorch                    | :candy:*​ |
| Theano                     |          |
| Caffe                      |          |
| Apache mxnet               |          |

*:candy:: Proof of concept for data extraction is available as a Jupyter Notebook. These notebooks can be used as a basis to develop additional frontend plug-ins for both of the frameworks.

### Supported Output (Backend)

| Neural Network Structure   |  Status  |
| :------------------------- | :------: |
| GCC-Compiler                     |    ✔️     |
| JSON | ✔️ |

### Supported Layer Types

| Layer Type          | Status |
| :------------------ | :----: |
| DropOut             |   ✔️    |
| Dense               |   ✔️    |
| Flatten             |   ✔️    |
| MaxPooling1D        |   ✔️    |
| MaxPooling2D        |   ✔️    |
| MaxPooling3D        |        |
| AvgPooling1D        |   ✔️    |
| AvgPooling2D        |   ✔️    |
| AvgPooling3D        |        |
| Conv1D              |        |
| Conv2D              |        |
| Conv3D              |        |
| Activation          |   ✔️    |
| Batch Normalization |        |
| Bias                |   ✔️    |

### Supported Activation Functions

| Activation Function (af) | Status |
| :----------------------- | :----: |
| Linear                   |   ✔️    |
| Sigmoid                  |   ✔️    |
| ReLu                     |   ✔️    |
| TanH                     |   ✔️    |
| Softmax                  |   ✔️    |

## How do I use this?
For a detailed description on how to use or develop plug-ins for this translator, please see the project's wiki.
