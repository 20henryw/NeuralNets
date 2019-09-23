/**
 * Created by Henry Wiese
 * 9.4.19
 */

import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;

import java.io.*;
import java.util.ArrayList;

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
    * Loads input from the input file and propagates the network.
    * @return an array containing the activations in the final layer
    * @throws IOException
    */
   public double[] run() throws IOException {
      File input = new File(INPUTS_PATH);
      BufferedReader br = new BufferedReader(new FileReader(input));
      String line = br.readLine();
      String[] values = line.split(",");
      br.close();

      for (int i = 0; i < values.length; i++) {
         activations[0][i] = Double.parseDouble(values[i]);
      }

      return propagate();
   }

   /**
    * Loads inputs from the parameters and propagates the network.
    * @param inputs the input layer's values
    * @return an array containing the activations in the final layer
    * @throws IOException
    */
   public double[] run(double[] inputs) throws IOException {
      for (int i = 0; i < inputs.length; i++) {
         activations[0][i] = inputs[i];
      }

      return propagate();
   }
   /**
    * This function takes inputs and then calculates the state of all the nodes until the final layer's value is
    * calculated.
    * @return an array containing the states of the nodes in the final layer
    */
   private double[] propagate() throws IOException {


      // weights[layer][from][to]. you go from the current layer
      for (int layer = 1; layer < numLayers; layer++) {
         for (int to = 0; to < activations[layer].length; to++) {
            double netInput = 0;
            if (DEBUG) System.out.print("DEBUG: a[" + layer + "][" + to + "] = f(");

            for (int from = 0; from < weights[layer - 1].length; from++) {
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
    * Calculates the derivative of the output function
    * @param x the variable
    * @return the desired derivative
    */
   private double dOutFunc(double x) {
      return outFunc(x) * (1 - outFunc(x));
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

      activations = new double[numLayers][];
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
         for (int from = 0; from < weights[layer].length; from++) {
            for (int to = 0; to < weights[layer][from].length; to++) {
               weights[layer][from][to] = Math.random() - 0.5;
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
    * )nly works if there is one node in the final layer
    * @param maxEpochs the number of times the weights should be trained
    * @param lambda the initial learning factor
    */
   public void train(int maxEpochs, double lambda) throws IOException {
      ArrayList<double[]> inputs = new ArrayList<>();
      ArrayList<Double> targets = new ArrayList<>();
      String[] values;

      BufferedReader br = new BufferedReader(new FileReader(new File(TRAINING_PATH)));
      String line = br.readLine();
      int count = 0;

      while (line != null && line.compareTo("") != 0) {
         values = line.split(",");
         inputs.add(new double[values.length]);
         for (int i = 0; i < values.length; i++) {
            System.out.println(Double.parseDouble(values[i]));
            inputs.get(count)[i] = Double.parseDouble(values[i]);
         }

         line = br.readLine();
         values = line.split(",");
         targets.add(Double.parseDouble(values[0]));

         line = br.readLine();
         count++;
      }
      br.close();

      for (int i = 0; i < targets.size(); i++) {
         for (int j = 0; j < inputs.get(i).length; j++) {
            getDeltaWeights(propagate()[0], targets.get(i));
            //TODO: Create run() method that calls propagate. run is overloaded, either taking inputs, or reading from an input file.
         }
      }


      //finish checking new propagation error and changing lambda

   }

   /**
    * Loops over the weights in the hidden layer and calculates deltas,
    * then loops over the weights in the input layer and calculates deltas.
    * @param output the inputs to the network
    * @param target the target output of the network
    */
   private double[][][] getDeltaWeights(double output, double target) {
      double[][][] deltaWeights = new double[weights.length][][];
      double diff = target - output;

      //final weight layer calculations
      for (int i = 0; i < weights[1].length; i++) {
         double dots = 0;
         for (int j = 0; j < weights[1].length; j++) {
               dots += activations[1][j] * weights[1][j][0];
         }
         deltaWeights[1][i][0] = -diff * dOutFunc(dots) * activations[1][i];
      }
      return deltaWeights;

   }

   /**
    * Calculates error based on the formula in Design Document 1.
    * Only works if there is one node in the final layer.
    * @param outputs The respective outputs for each input
    * @param targets The respective target values for each input.
    * @return the error
    */
   private double getError(double[] outputs, double[] targets) {
      double error = 0;

      for (int i = 0; i < outputs.length; i++) {
         error += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
      }

      return error / 2;
   }
}
