# NeuralNets

## Data
###layers.csv

###weights.csv
Each line in this file represents one layer of the network. Then, values are listed in increasing order of nodes in the layer,  
with each weight to the following node. For example, for a network with two input nodes and two nodes in the next layer, the  
weights would be as follows:  
000, 001, 010, 011  
The first digit is the layer, the second is the "from" node, and the 3rd digit is the "to" node.
Therefore, the second number represents the weight connecting the 1st node in the 1st layer and the 2nd node in the next layer.