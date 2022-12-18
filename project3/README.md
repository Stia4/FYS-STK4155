# FYS-STK4155 Project 3: Convolutional Neural Network

This is the code repository for our final project in Applied Data Analysis and Machine Learning, where we try to classify images of playing cards using a convolutional nerual network.
Below is a description of the repo contents, as well as a quick introduction to running the code.

## Repository contents

- src/
  * NeuralNetwork.py

    Container for the NerualNetwork class which is the central controller for our network. Acts as a container and API to setting up and calling upon various components of the network, including forward sweeps and network training. Has suport for both custom user-input learning rate schedulers and cost functions, as well as several preset methods which can be set. All calculation is offset to actual nerual layers (see src/Layers.py), and the network class is intended to simplify setting up and calling upon these for calculation and training.
  
  * Layers.py

    Collection of various nerual layer classes, at the moment containing DenseLayer, ConvolutionalLayer, and PoolingLayer classes. Each layer class follows a similar structure where they have unique methods for initialization, calling (forward calculation), and updating parameters (backward calculation). They also share methods for parameter initialization and reset. While the layers can be used on their own (as seen in conv_layer_test.py and pool_layer_test.py) they are intended to be used mainly through the NerualNetwork class from src/NeuralNetwork.py.

- raw_data/
  
  Folder for depositing data to be analyzed/processed. See raw_data/README.md for data source and required structure for this project.

- fig/ and params/

  Folders to store finished plots and trained network weights. Figure folder also contains 'setups' folder intended as a newtork setup log for assisting figure recreation through the network.

- DataProcessing.py

  Program to process the full 224x224 RGB images into smaller 50x50 grayscale versions, storing them in a new folder 'data'. Also contains a function 'load_data' for an easy way to access images.

- CardClassification.py

  Main program of convolutional analysis. Contains functions defining different classification scenarios, including simple two card differentiation (binary problem), classification by suit, and full 52 card classification. These were used to train the network and analyze setups and results. Currently the code is set up to recreate the figures used in our report both by live training the network and using pre-trained weights.

- conv_layer_test.py and pool_layer_test.py

  Testing programs used to check the behaviour of neural layers. Not used as part of the final product.

- trees.py

  Code which tries to classify all 52 cards using the same dataset as our convolutional network, except through decision trees and bagging methods. Used to compare results with our network and is therefore using premade methods from machine learning library scikit-learn (sklearn).

- BVtradeoff_analysis.py

  Code for the additonal (optional) exercise where we study the bias-variance tradeoff as function of model complexity. We look at standard linear regression, decision trees, and neural network methods. As before we use machine learning library scikit-learn (sklearn) for ease of analysis.

## Running the code

To reproduce figures used in our report, follow these steps:

1. Download and extract data according to instructions in raw_data/README.md
2. Run DataProcessing.py to generate downscaled data
   
   ```
   $ python3 DataProcessing.py
   ```
3. Run analysis programs, which will produce figures and save them to folder 'fig', as well as print information to the terminal.
   
   ```
   $ python3 CardClassification.py
   $ python3 trees.py
   $ python3 BVtradeoff_analysis.py
   ```
