/**
 * Created by Henry Wiese
 * 9.4.19
 */

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * The Network class builds a neural network with an inputted number of layers, nodes, and weights.
 * It stores the number of nodes in each layer, every activation, and every weight.
 * All nodes in one layer are connected to all nodes in the next layer. No other connections are made.
 * Building a multi-layer perceptron.
 *
 * Network()         - constructs a network object from user specified data
 * getWeights()      - returns the weights array
 * setWeights()      - sets a weight array
 * run()             - calls propagate with specified inputs
 * propagate()       - calculates the activations of each layer, returning an array of the output layer's activations
 * outFunc()         - calculates the result of a specified function
 * dOutFunc()        - calculates the derivative of the output function
 * loadData()        - loads user inputted data into the network
 * randWeights()     - loads random weights between -0.5 and 0.5
 * train()           - optimizes weights until end conditions are satisfied
 * optimizeWeights() - optimizes the weight array for one instance
 * getCaseError()    - gets the error of one case of inputs and outputs
 * getError()        - gets the error of multiple cases
 * initializeJaggedArray() - initializes a jagged array based on the network's layer structure
 *
 */
public class Network
{
   private int[] layers;
   private int numLayers;
   private double[][] activations;
   private double[][] theta;
   private double[][] omega;
   private double[][] psi;
   private double[][][] weights; // [layer][from][to]
   private String FILES_PATH;
   private String INPUTS_PATH;

   private boolean DEBUG = false;

   public Network() throws IOException
   {
      FILES_PATH = "/Users/henry/Documents/2019-2020/NeuralNets/data/files.csv";
      loadData();
   }

   public Network(String FILES_PATH) throws IOException
   {
      this.FILES_PATH = FILES_PATH;
      loadData();
   }

   public double[][][] getWeights()
   {
      return weights;
   }

   public void setWeights(double[][][] weights)
   {
      this.weights = weights;
   }

   /**
    * Loads input from the input file and propagates the network.
    *
    * @return an array containing the activations in the final layer
    * @throws IOException
    */
   public double[] run() throws IOException
   {
      File input = new File(INPUTS_PATH);
      BufferedReader br = new BufferedReader(new FileReader(input));
      String line = br.readLine();
      String[] values = line.split(",");
      br.close();

      for (int i = 0; i < values.length; i++)
      {
         activations[0][i] = Double.parseDouble(values[i]);
      }

      return propagate();
   }

   /**
    * Loads inputs from the parameters and propagates the network.
    *
    * @param inputs the input layer's values
    * @return an array containing the activations in the final layer
    * @throws IOException
    */
   public double[] run(double[] inputs) throws IOException
   {
      for (int i = 0; i < inputs.length; i++)
      {
         activations[0][i] = inputs[i];
      }

      return propagate();
   }

   /**
    * This function takes inputs and then calculates the state of all the nodes until the final layer's value is
    * calculated.
    *
    * @return an array containing the states of the nodes in the final layer
    */
   private double[] propagate() throws IOException
   {

      for (int layer = 1; layer < numLayers; layer++)
      {
         for (int to = 0; to < activations[layer].length; to++)
         {
            double netInput = 0;
            if (DEBUG) System.out.print("DEBUG: a[" + layer + "][" + to + "] = f(");

            for (int from = 0; from < weights[layer - 1].length; from++)
            {
               netInput += activations[layer - 1][from] * weights[layer - 1][from][to];
               if (DEBUG)
                  System.out.print("a[" + (layer - 1) + "][" + from + "]w[" + (layer - 1) + "][" + from + "][" + to + "] + ");
            }
            if (DEBUG) System.out.println(")");

            double debugTemp = netInput;
            activations[layer][to] = outFunc(netInput);
         }
      }

      if (DEBUG) System.out.println("");

      return activations[numLayers - 1];
   }

   /**
    * Calculates the output function of a node.
    * Currently is the threshold function.
    *
    * @param x the variable of the function
    * @return the desired output function
    */
   private double outFunc(double x)
   {
      return (1.0 / (1.0 + Math.exp(-x)));
   }

   /**
    * Calculates the derivative of the output function
    *
    * @param x the variable
    * @return the desired derivative
    */
   private double dOutFunc(double x)
   {
      double f = outFunc(x);
      return f * (1.0 - f);
   }

   /**
    * Loads layer information into layers[] and weights information into weights[][][].
    * The method first reads the second line of the weight file to determine the number of nodes in each layer
    * and the total number of layers. It then sets numLayers and MAX_LAYER_SIZE according to the input.
    * The weights array is created based on numLayers and MAX_LAYER_SIZE. It is then populated randomly
    * or with values in the weight file based on the user's choice.
    * More info about correctly using a weight file is in the readme.
    */
   private void loadData() throws IOException
   {
      File files = new File(FILES_PATH);
      BufferedReader br = new BufferedReader(new FileReader(files));

      //unused lines are only for visual clarity
      String line = br.readLine();
      String weightPath = br.readLine();
      line = br.readLine();
      INPUTS_PATH = br.readLine();
      br.close();

      br = new BufferedReader(new FileReader(new File(weightPath)));
      line = br.readLine(); //for visual clarity
      line = br.readLine();

      String[] values = line.split(",");
      numLayers = values.length;
      layers = new int[numLayers];

      for (int i = 0; i < numLayers; i++)
      {
         layers[i] = Integer.parseInt(values[i]);
      }

      activations = new double[numLayers][];
      theta = new double[numLayers][];
      omega = new double[numLayers][];
      psi = new double[numLayers][];

      for (int i = 0; i < activations.length; i++)
      {
         activations[i] = new double[layers[i]];
         theta[i] = new double[layers[i]];
         omega[i] = new double[layers[i]];
         psi[i] = new double[layers[i]];
      }


      weights = initializeJaggedArray();

      line = br.readLine();

      //load manual input
      if (line.compareTo("manual") == 0)
      {
         for (int layer = 0; layer < weights.length; layer++)
         {
            line = br.readLine();
            values = line.split(",");
            int valIndex = 0;
            for (int from = 0; from < weights[layer].length; from++)
            {
               for (int to = 0; to < weights[layer][from].length; to++)
               {
                  weights[layer][from][to] = Double.parseDouble(values[valIndex]);
                  valIndex++;
               }
            }
         }
      } else
      {  //user wants random input
         randWeights(-1,1);
      }


      br.close();
   }

   /**
    * Randomly assigns starting weights between -0.1 and 0.1
    */
   public void randWeights(double min, double max)
   {
      // numLayers -1 to avoid index out of bound errors, since you don't calculate weights from the output layer
      for (int layer = 0; layer < weights.length; layer++)
      {
         for (int from = 0; from < weights[layer].length; from++)
         {
            for (int to = 0; to < weights[layer][from].length; to++)
            {
//               weights[layer][from][to] = new Random().nextGaussian();
               weights[layer][from][to] = (Math.random() * (max - min)) + min;
//               weights[layer][from][to] = (Math.random() * 2) - 1;
//               weights[layer][from][to] = 0;
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
    *
    */
   public String train(ArrayList<double[]> inputs, ArrayList<double[]> targets, double lambda, int MAX_EPOCHS,
                     double lambdaFactor, double MIN_LAMBDA, double ERROR_THRESHOLD, int NUM_PRINTS) throws IOException
   {
      int endCondition = 0;
      String endString = "";
      double[][][] prevWeights = initializeJaggedArray();
      double prevError = 0;
      int epochs = 0;
      int printFactor = MAX_EPOCHS / NUM_PRINTS;
      int lastShift = 0;

      while (endCondition == 0)
      {
         prevWeights = weights;

         if((epochs % printFactor) == 0)
         {
            System.out.println("Epoch " + epochs + ": " + prevError);
         }

         for (int i = 0; i < targets.size(); i++)
         {
            weights = optimizeWeights(inputs.get(i), targets.get(i), lambda);
         }

         /**
         for (int i = 0; i < targets.size(); i++)
         {
            prevError = getCaseError(inputs.get(i), targets.get(i));
//            prevError = getChaituError(inputs, targets);

            weights = optimizeWeights(inputs.get(i), targets.get(i), lambda);

            double newError = getCaseError(inputs.get(i), targets.get(i));
//            double newError = getChaituError(inputs, targets);
            if (newError < prevError)
            {
               lastShift = epochs;
               lambda *= lambdaFactor;
               prevError = newError;
               prevWeights = weights;
            } else
            {
               weights = prevWeights;
               lambda /= lambdaFactor;
            }

         }
          */

         epochs++;

         if (epochs >= MAX_EPOCHS)
         {
            endCondition = 1;
            endString += "ENDED on epochs.";
         }
         else if (lambda <= MIN_LAMBDA)
         {
            endCondition = 2;
            endString += "ENDED on lambda value.";
         }
         else if (getError(inputs, targets) < ERROR_THRESHOLD)
         {
            endCondition = 3;
            endString += "ENDED on error.";
         }
      }

      //System.out.println("FINAL lambda: " + lambda);
      //System.out.println("FINAL Error: " + getError(inputs, targets));
      //System.out.println("FINAL Epochs: " + epochs);
      //System.out.println(Arrays.deepToString(weights));

      endString += "\nEpochs: " + epochs + "\nLambda: " + lambda + "\nError: " + getError(inputs, targets);
      return endString;
   }


   /**
    * Method title inspired by Kyle Li.
    *
    * Returns a new weights[][][] array based on the error from the target values and the lambda change factor
    *
    * First propagates forward through the network, calculating the values of each activation while storing values
    * that are reused later to change weights. Then, it propagates backwards through the activation layers, calculating
    * deltas and storing important values for calculating deltas in earlier layers.
    *
    * The important values to keep track of are theta, omega, and psi. A full derivation of the math can be found in
    * Design Document 3.
    * Iterator names follow the conventions in the Design Documents.
    *
    * @param targets
    * @return
    */
   private double[][][] optimizeWeights(double[] inputs, double[] targets, double lambda)
   {
      long startTime = System.currentTimeMillis();
      double[][][] newWeights = initializeJaggedArray();
      int prevLayer;

      // layer indices currently used for back prop
      int inputLayer = 0;
      int hiddenLayer = 1;
      int finalLayer = 2;

      for (int i = 0; i < inputs.length; i++)
      {
         activations[inputLayer][i] = inputs[i];
      }

      for (int layer = 1; layer < numLayers; layer++)
      {
         prevLayer = layer - 1;
         for (int to = 0; to < activations[layer].length; to++)
         {
            theta[layer][to] = 0;

            for (int from = 0; from < weights[prevLayer].length; from++)
            {
               theta[layer][to] += activations[prevLayer][from] * weights[prevLayer][from][to];
            }

            activations[layer][to] = outFunc(theta[layer][to]);
         }
      }

      for (int i = 0; i < activations[finalLayer].length; i++)
      {
         omega[finalLayer][i] = targets[i] - activations[finalLayer][i];
         psi[finalLayer][i] = omega[finalLayer][i] * dOutFunc(theta[finalLayer][i]);

         for (int j = 0; j < activations[hiddenLayer].length; j++)
         {
            newWeights[hiddenLayer][j][i] = weights[hiddenLayer][j][i] + lambda * activations[hiddenLayer][j] * psi[finalLayer][i];
         }
      }

      for (int j = 0; j < activations[hiddenLayer].length; j++)
      {
         omega[hiddenLayer][j] = 0;
         for (int I = 0; I < activations[finalLayer].length; I++)
         {
            omega[hiddenLayer][j] += psi[finalLayer][I] * weights[hiddenLayer][j][I];
         }

         psi[hiddenLayer][j] = omega[hiddenLayer][j] * dOutFunc(theta[hiddenLayer][j]);
         for (int k = 0; k < activations[inputLayer].length; k++)
         {
            newWeights[inputLayer][k][j] = weights[inputLayer][k][j] + lambda * activations[inputLayer][k] * psi[hiddenLayer][j];
         }
      }

      return newWeights;
   }

   /**
    * Calculates Error based of description in Design Doc 2
    * @param input
    * @param target
    * @return
    * @throws IOException
    */
   public double getCaseError(double[] input, double[] target) throws IOException
   {
      double[] finLayer = run(input);
      double diff;
      double error = 0.0;

      for (int i = 0; i < target.length; i++)
      {
         diff = target[i] - finLayer[i];
         error += diff * diff;
      }

      return error / 2.0;
//      return Math.sqrt(error) / 2.0;
   }

   /**
    * Calculates error based on the formula in Design Document 1, then takes the sqrt of the errors' squares
    * Only works if there is one node in the final layer.
    * targets is an ArrayList because that is the data structure used for storing targets in train(), which calls
    * getError()
    *
    * @param inputs
    * @param targets The respective target values for each input.
    * @return the error
    */
   public double getError(ArrayList<double[]> inputs, ArrayList<double[]> targets) throws IOException
   {
      double error = 0.0;
      double caseError = 0.0;

      for (int i = 0; i < inputs.size(); i++)
      {
         caseError = getCaseError(inputs.get(i), targets.get(i));
         error += caseError * caseError;
      }

      return Math.sqrt(error);
   }

   /**
    * Calculates error the way Chaitu Ravuri would calculate error.
    */
   public double getChaituError(ArrayList<double[]> inputs, ArrayList<double[]> targets) throws IOException
   {
      double error = 0.0;
      for (int i = 0; i < inputs.size(); i++)                                            // for each test case
      {
         double[] output = run(inputs.get(0));                                        // propagate to get the output
         double singleError = 0.0;
         for (int j = 0; j < output.length; j++)
         {
            singleError += (targets.get(i)[j] - output[j]) * (targets.get(i)[j] - output[j]);   // compare output with expected
         }
         error += (0.5 * singleError) * (0.5 * singleError);                              // sum this up for each case
      }

      return error;

   }

   /**
    * Creates a jagged weight array based on the number of nodes per layer
    *
    * @return the jagged array specified by layers[]
    */
   private double[][][] initializeJaggedArray()
   {
      double[][][] jagged = new double[layers.length - 1][][];
      for (int i = 0; i < jagged.length; i++)
      {
         jagged[i] = new double[layers[i]][layers[i + 1]];
      }

      return jagged;
   }

}
