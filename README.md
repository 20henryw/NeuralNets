# NeuralNets

## Data
### files.csv
This file stores the file paths to other relevant files to the Neural Network.
Use this file to point to other files that store network config, run time inputs, and training data

### Config
Under Layers, write the number a comma separated list of activations per layer

Skip a line.
The next line should say manual or random to choose the style of initializing weights.
If you have manual weights, each line is a layer, with the the list of weights in order per line.

A 2,2,1 network could look like this.

1,2,3,4
9,8

### Inputs
Enter run time inputs in a comma separated list.

### Training
constants.csv stores many constants relevant to training the network. Enter specified values under the name of your constant in the file.

The format of training file paths should be as follows:
Alternating lines of correspond inputs and target final layer values. For example,
0,0
0,1,0
0,1
1,1,1

There are two cases: 0,0 and 0,1. They have targets 0,1,0 and 1,1,1 respectively.