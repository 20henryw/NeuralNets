/**
 * Created by Henry Wiese
 * 9.4.19
 */

import java.io.*;

/**
 * TODO ADD DOCUMENTATION
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
   private String WEIGHTS_PATH = "/Users/henry/Documents/2019-2020/NeuralNets/data/weights.csv";

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
    * Currently multiplies the state and weight together.
    * @param state the state of the node
    * @param weight the weight between the current node and the next node
    * @return the desired output function
    */
   private double outFunc(double state, double weight) {
      return state * weight;
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
    * Loads information from layers.csv into layers[] and weights.csv into weights[][][].
    * The method first reads layers.csv to determine the number of nodes in each layer and the total number of layers.
    * It then sets numLayers and MAX_LAYER_SIZE according to the input. The weights array is created based on
    * numLayers and MAX_LAYER_SIZE. It is then populated randomly or with values in weights.csv based on
    * the user's choice in weights.csv. More info about correctly using weights.csv is in the readme.
    */
   private void loadData() throws IOException {
      File layerFile = new File(LAYERS_PATH);
      File weightFile = new File(WEIGHTS_PATH);
      BufferedReader br = new BufferedReader(new FileReader(layerFile));

      String line = br.readLine();

      String[] values = line.split(",");
      numLayers = values.length;
      layers = new int[numLayers];

      int bigLayer = Integer.MIN_VALUE;

      //read in data from layers.csv
      for (int i = 0; i < numLayers; i++) {
         layers[i] = Integer.parseInt(values[i]);
         if (layers[i] > bigLayer)
            bigLayer = layers[i];
      }

      MAX_LAYER_SIZE = bigLayer;
      activations = new double[numLayers][MAX_LAYER_SIZE];
      weights = new double[numLayers][MAX_LAYER_SIZE][MAX_LAYER_SIZE];

      //read in data from weights.csv
      br = new BufferedReader(new FileReader(weightFile));
      line = br.readLine();

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

      System.out.println("pause");
   }


}
