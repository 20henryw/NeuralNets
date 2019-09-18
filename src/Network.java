/**
 * Created by Henry Wiese
 * 9.4.19
 */

import java.io.*;

/**
 * The Network class builds a neural network with an inputted number of layers, nodes, and weights.
 * It stores the number of nodes in each layer, every activation, and every weight.
 * All nodes in one layer are connected to all nodes in the next layer. No other connections are made.
 *
 *  Network()      - constructs a network object from user specified data
 *  propagate()    - calculates the activations of each layer, returning an array of the output layer's activations
 *  outFunc()      - combines the output of a node and weight into one number
 *  netInputFunc() - combines all inputs into a node into one number
 *  loadData()     - loads user inputted data into the network
 *  randWeights()  - loads random weights between -0.5 and 0.5
 */
public class Network
{
   private int[] layers;
   private int numLayers;
   private int MAX_LAYER_SIZE;
   private double[][] activations;
   private double[][][] weights;
   private String INPUT_PATH = "/Users/henry/Documents/2019-2020/NeuralNets/data/input.csv";
   private String LAYERS_PATH = "/Users/henry/Documents/2019-2020/NeuralNets/data/layers.csv";
   private String WEIGHTS_PATH = "/Users/henry/Documents/2019-2020/NeuralNets/data/weights/xor.csv";
   private String FILES_PATH = "/Users/henry/Documents/2019-2020/NeuralNets/data/files.csv";

   public Network() throws IOException {
      loadData();
   }

   /**
    * This function takes inputs and then calculates the state of all the nodes until the final layer's value is
    * calculated.
    * @return an array containing the states of the nodes in the final layer
    */
   public double[] propagate() throws IOException {

      File input = new File(INPUT_PATH);
      BufferedReader br = new BufferedReader(new FileReader(input));
      String line = br.readLine();
      String[] values = line.split(",");
      br.close();

      for (int i = 0; i < values.length; i++) {
         activations[0][i] = Double.parseDouble(values[i]);
      }

      // weights[layer][from][to]. you go from the current layer
      for (int layer = 1; layer < numLayers; layer++) {
         for (int i = 0; i < layers[layer]; i++) {        // the to
            double[] nodeOutputs = new double[layers[layer -1]];
            for (int j = 0; j < layers[layer - 1]; j++) { // the from
               nodeOutputs[j] = outFunc(activations[layer - 1][j], weights[layer - 1][j][i]);
            }
            activations[layer][i] = netInputFunc(nodeOutputs);
         }
      }

      //clones output into a new array to prevent pointer errors
      return activations[numLayers - 1].clone();
   }

   /**
    * Calculates the output function of a node.
    * Currently multiplies the activation and weight together, then finds the sigmoid of that
    * @param state the state of the node
    * @param weight the weight between the current node and the next node
    * @return the desired output function
    */
   private double outFunc(double state, double weight) {
      double dot = state * weight;
      return (1.0 / (1 + Math.exp(-dot)));
   }

   /**
    * Combines the inputs of many nodes into a net input.
    * Currently adds together all inputs into a net input.
    * @param inputs all inputs to a node
    * @return the desired output of the input function.
    */
   private double netInputFunc(double[] inputs) {
      double netInput = 0;

      for (double input : inputs) {
         netInput += input;
      }

      return netInput;
   }

   /**
    * Loads layer information nto layers[] and weights information into weights[][][].
    * The method first reads the second line of the weight file to determine the number of nodes in each layer
    * and the total number of layers. It then sets numLayers and MAX_LAYER_SIZE according to the input.
    * The weights array is created based on numLayers and MAX_LAYER_SIZE. It is then populated randomly
    * or with values in the weight file based on the user's choice.
    * More info about correctly using a weight file is in the readme.
    */
   private void loadData() throws IOException {
      File weightFile = new File(WEIGHTS_PATH);
      BufferedReader br = new BufferedReader(new FileReader(weightFile));

      String line = br.readLine(); //skip the first line, which is only there to add visual clarity
      line = br.readLine();

      String[] values = line.split(",");
      numLayers = values.length;
      layers = new int[numLayers];

      int bigLayer = Integer.MIN_VALUE;

      //read layer data, and determine the largest layer size
      for (int i = 0; i < numLayers; i++) {
         layers[i] = Integer.parseInt(values[i]);
         if (layers[i] > bigLayer)
            bigLayer = layers[i];
      }

      MAX_LAYER_SIZE = bigLayer;
      activations = new double[numLayers][MAX_LAYER_SIZE];
      weights = new double[numLayers][MAX_LAYER_SIZE][MAX_LAYER_SIZE];

      line = br.readLine();
      while (line.compareTo("manual") != 0 && line.compareTo("random") != 0)
      {
         line = br.readLine();
      }

      //load manual input
      if (line.compareTo("manual") == 0) {
         for (int layer = 0; layer < numLayers - 1; layer++) {
            line = br.readLine();
            values = line.split(",");
            int valIndex = 0;
            for (int i = 0; i < layers[layer]; i++) {
               for (int j = 0; j < layers[layer + 1]; j++) {
                  weights[layer][i][j] = Double.parseDouble(values[valIndex]);
                  valIndex++;
               }
            }
         }
      }
      else { //user wants random input
         randWeights();
      }

      br.close();
   }

   /**
    * Randomly assigns starting weights between -0.5 and 0.5
    */
   private void randWeights() {
      // numLayers -1 to avoid index out of bound errors, since you don't calculate weights from the output layer
      for (int layer = 0; layer < numLayers - 1; layer++) {
         for (int i = 0; i < layers[layer]; i++) {
            for (int j = 0; j < layers[layer + 1]; j++) {
               weights[layer][i][j] = Math.random() - 0.5;
            }
         }
      }
   }

   /**
    * For a certain number of epochs, train will attempt to reduce the error of an input.
    * It iterates through every input case and finds the delta weights based on each case's respective target output
    * Then, the system propogates with the new weights. If the new error is less, lambda, the learning rate will
    * double. If the new error is more, the activations will be reset to the previous value and lambda
    * will be divided by 2.
    * @param maxEpochs the number of times the weights should be trained
    * @param lambda the initial learning factor
    */
   private void train(double[][] inputs, double[] targets, int maxEpochs, int lambda) throws IOException {
      double[][] prevActivations = new double[numLayers][MAX_LAYER_SIZE];
      double[][][] deltaWeights = new double[numLayers][MAX_LAYER_SIZE][MAX_LAYER_SIZE];;

      for (int epoch = 0; epoch < maxEpochs; epoch++) {
         for (int i = 0; i < prevActivations.length; i++) {
            for (int j = 0; j < prevActivations[i].length; j++) {
               prevActivations[i][j] = activations[i][j];
            }
         }

         for (int i = 0; i < inputs.length; i++) {
            deltaWeights = getDeltaWeights(inputs[i], targets[i]);
         }

         for (int layer = 1; layer < numLayers; layer++) {
            for (int i = 0; i < layers[layer]; i++) {        // the to
               double[] nodeOutputs = new double[layers[layer -1]];
               for (int j = 0; j < layers[layer - 1]; j++) { // the from
                  weights[layer][i][j] += lambda * deltaWeights[layer][i][j];
               }
            }
         }

         propagate();

         //finish checking new propogation error and changing lambda

      }

   }

   /**
    * Loops over the weights in the hidden layer and calculates deltas,
    * then loops over the weights in the input layer and calculates deltas.
    * @param inputs the inputs to the network
    * @param target the target output of the network
    */
   private double[][][] getDeltaWeights(double inputs[], double target) {


   }
}
