# NeuralNets

## Data

###layers.csv
This file describes the number of no'des in each layer.  
For example:  
2,2,1  
This describes the first layer with 2 nodes, the second layer with 2 nodes, and the third layer with 1 node. 
###weights.csv
For manual input, the first line must contain only the word "random". For random weights between -0.5 and 0.5, change the first line to anything else.  
Each line in this file represents one layer of the network. Then, values are listed in increasing order of nodes in the layer,  
with each weight to the following node. For example, for a network with two input nodes and two nodes in the next layer, the  
weights would be as follows:  
000, 001, 010, 011  
The first digit is the layer, the second is the "from" node, and the 3rd digit is the "to" node.
Therefore, the second number represents the weight connecting the 1st node in the 1st layer and the 2nd node in the next layer.  
All inputs are read in as doubles.
###input.csv
The inputs of the system in the order that they should be placed into nodes.  
For example:  
0.5,1  
This sets the first input node to 0.5, and the second to 1  
All inputs are read in as doubles.
