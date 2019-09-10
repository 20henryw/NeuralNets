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
   private int MAX_LAYER_SIZE = 2;
   private double[][] activations;
   private double[][][] weights;

   public Network() {

      layers = new int[]{2, 2, 1};

      numLayers = layers.length;
      activations = new double[numLayers][MAX_LAYER_SIZE];
      weights = new double[numLayers][MAX_LAYER_SIZE][MAX_LAYER_SIZE];
      fillWeights();
   }

   /**
    * Randomly assigns starting weights between -0.5 and 0.5
    */
   private void fillWeights() {
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

   /**
    * This function takes inputs and then calculates the state of all the nodes until the final layer's value is
    * calculated.
    * @param inputs the inputs corresponding to each node in the input layer
    * @return an array containing the states of the nodes in the final layer
    */
   public double[] propagate(double[] inputs) {
      double[] netOutput = new double[layers[numLayers -1]];

      for (int i = 0; i < inputs.length; i++) {
         activations[0][i] = inputs[i];
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

      //copies output into a new array to prevent pointer errors
      for (int i = 0; i < layers[numLayers -1]; i++) {
         netOutput[i] = activations[numLayers - 1][i];
      }

      return netOutput;
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

      for (int i = 0; i < inputs.length; i++) {
         netInput += inputs[i];
      }

      return netInput;
   }

   /**
    * Loads information from Settings.txt into layers[] and weights[][][]
    */
   public void loadSettings() throws IOException {
      File file = new File("src/Settings.txt");
      BufferedReader br = new BufferedReader(new FileReader(file));

      String line = br.readLine();
      if (line.compareTo("LAYERS") != 0) {
         System.out.println("you done goofed");
      }

      String test = Integer.toString((br.read()));
      System.out.println(test);

      br.close();
   }

}
