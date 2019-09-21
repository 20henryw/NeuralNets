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
 *  loadData()     - loads user inputted data into the network
 *  randWeights()  - loads random weights between -0.5 and 0.5
 */
public class Network
{
   private int numLayers;
   private double[][] activations;
   private double[][][] weights;
   private String FILES_PATH = "/Users/henry/Documents/2019-2020/NeuralNets/data/files.csv";
   private String INPUTS_PATH;
   private String TRAINING_PATH;

   private boolean DEBUG = true;

   public Network() throws IOException {
      loadData();
   }

   /**
    * This function takes inputs and then calculates the state of all the nodes until the final layer's value is
    * calculated.
    * @return an array containing the states of the nodes in the final layer
    */
   public double[] propagate() throws IOException {

      File input = new File(INPUTS_PATH);
      BufferedReader br = new BufferedReader(new FileReader(input));
      String line = br.readLine();
      String[] values = line.split(",");
      br.close();

      for (int i = 0; i < values.length; i++) {
         activations[0][i] = Double.parseDouble(values[i]);
      }

      // weights[layer][from][to]. you go from the current layer
      for (int layer = 1; layer < numLayers; layer++) {
         for (int to = 0; to < weights[layer].length; to++) {        // the to
            double netInput = 0;
            if (DEBUG) System.out.print("DEBUG: a[" + layer + "][" + to + "] = f(");

            for (int from = 0; from < weights[layer - 1].length; from++) { // the from
               netInput += activations[layer - 1][from] * weights[layer - 1][from][to];
               if (DEBUG) System.out.print("a[" + (layer - 1) + "][" + from + "]w[" + (layer - 1) + "][" + from + "][" + to + "] + ");
            }
            if (DEBUG) System.out.println(")");

            activations[layer][to] = outFunc(netInput);

         }
      }

      //clones output into a new array to prevent pointer errors
      return activations[numLayers - 1].clone();
   }

   /**
    * Calculates the output function of a node.
    * Currently is the threshold function.
    * @param x the variable of the function
    * @return the desired output function
    */
   private double outFunc(double x) {
      return (1.0 / (1 + Math.exp(-x)));
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
      File files = new File(FILES_PATH);
      BufferedReader br = new BufferedReader(new FileReader(files));

      //unused lines are only for visual clarity
      String line = br.readLine();
      String weightPath = br.readLine();
      line = br.readLine();
      INPUTS_PATH = br.readLine();
      line = br.readLine();
      TRAINING_PATH = br.readLine();
      br.close();

      br = new BufferedReader(new FileReader(new File(weightPath)));
      line = br.readLine(); //for visual clarity
      line = br.readLine();

      String[] values = line.split(",");
      numLayers = values.length;
      int[] layers = new int[numLayers];

      for (int i = 0; i < numLayers; i++) {
         layers[i] = Integer.parseInt(values[i]);
      }

      activations = new double[numLayers][0];
      for (int i = 0; i < activations.length; i++) {
         activations[i] = new double[layers[i]];
      }

      weights = new double[numLayers - 1][][];
      for (int i = 0; i < weights.length; i++) {
         weights[i] = new double[layers[i]][layers[i + 1]];
      }

      //reads in weights
      line = br.readLine(); //for visual clarity
      line = br.readLine();

      //load manual input.,
      if (line.compareTo("manual") == 0) {
         for (int layer = 0; layer < weights.length; layer++) {
            line = br.readLine();
            values = line.split(",");
            int valIndex = 0;
            for (int from = 0; from < weights[layer].length; from++) {
               for (int to = 0; to < weights[layer][from].length; to++) {
                  weights[layer][from][to] = Double.parseDouble(values[valIndex]);
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
      for (int layer = 0; layer < weights.length; layer++) {
         for (int i = 0; i < weights[layer].length; i++) {
            for (int j = 0; j < weights[layer][i].length; j++) {
               weights[layer][i][j] = Math.random() - 0.5;
            }
         }
      }
   }

   /**
    * For a certain number of epochs, train will attempt to reduce the error of an input.
    * It iterates through every input case and finds the delta weights based on each case's respective target output
    * Then, the system propagates with the new weights. If the new error is less, lambda, the learning rate will
    * double. If the new error is more, the weights will be reset to their previous values and lambda
    * will be divided by 2.
    * @param maxEpochs the number of times the weights should be trained
    * @param lambda the initial learning factor
    */
   private void train(double[][] inputs, double[][] targets, int maxEpochs, int lambda) throws IOException {
      //TODO: READ TRAINING DATA IN FROM THE TRAINING FILE. CONSTRUCT ARRAYS OUT OF THEM, AND PASS VALUES TO getDeltaWeights()



         //finish checking new propagation error and changing lambda

   }

   /**
    * Loops over the weights in the hidden layer and calculates deltas,
    * then loops over the weights in the input layer and calculates deltas.
    * @param inputs the inputs to the network
    * @param target the target output of the network
    */
   private double[][][] getDeltaWeights(double[] inputs, double[] target) {

      return new double[1][1][1];

   }

   private double getError(double[][] inputs, double[][] target) {
      double error = 0;

      return error;
   }
}
