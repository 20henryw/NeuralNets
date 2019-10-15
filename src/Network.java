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
 * <p>
 * Network()      - constructs a network object from user specified data
 * propagate()    - calculates the activations of each layer, returning an array of the output layer's activations
 * outFunc()      - combines the output of a node and weight into one number
 * loadData()     - loads user inputted data into the network
 * randWeights()  - loads random weights between -0.5 and 0.5
 */
public class Network
{
   private int[] layers;
   private int numLayers;
   private double[][] activations;
   private double[][][] weights; // [layer][from][to]
   private String FILES_PATH = "/Users/henry/Documents/2019-2020/NeuralNets/data/files.csv";
   private String INPUTS_PATH;
   private String TRAINING_PATH;

   private boolean DEBUG = false;

   public Network() throws IOException
   {
      loadData();
   }

   public double[][][] getWeights()
   {
      return weights;
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
      return (1.0 / (1 + Math.exp(-x)));
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
      return f * (1 - f);
   }

   /**
    * Loads layer information nto layers[] and weights information into weights[][][].
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
      line = br.readLine();
      TRAINING_PATH = br.readLine();
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
      for (int i = 0; i < activations.length; i++)
      {
         activations[i] = new double[layers[i]];
      }

      weights = initializeJaggedArray();

      line = br.readLine(); //for visual clarity
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
         randWeights();
      }

      br.close();
   }

   /**
    * Randomly assigns starting weights from a Gaussian distribution.
    * Idea to use a Gaussian was given by Chaitu.
    */
   public void randWeights()
   {
      // numLayers -1 to avoid index out of bound errors, since you don't calculate weights from the output layer
      for (int layer = 0; layer < weights.length; layer++)
      {
         for (int from = 0; from < weights[layer].length; from++)
         {
            for (int to = 0; to < weights[layer][from].length; to++)
            {
               weights[layer][from][to] = new Random().nextGaussian();
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
    * Only works if there is one node in the final layer
    *
    */
   public void train(ArrayList<double[]> inputs, ArrayList<double[]> targets, double lambda, int MAX_EPOCHS,
                     double lambdaFactor, double MIN_LAMBDA, double ERROR_THRESHOLD) throws IOException
   {
      double[][][] prevWeights = initializeJaggedArray();
      double prevError = 0;
      int epochs = 0;
      int lastShift = 0;
      while (epochs < MAX_EPOCHS && lambda != 0)
      {
         prevWeights = weights;

         for (int i = 0; i < targets.size(); i++)
         {
            prevError = getCaseError(inputs.get(i), targets.get(i));

            weights = optimizeWeights(run(inputs.get(i)), targets.get(i), lambdaFactor);

            double newError = getCaseError(inputs.get(i), targets.get(i));
            if (newError < prevError)
            {
               lastShift = epochs;
               lambda *= lambdaFactor;
               prevError = newError;
            } else
            {
               weights = prevWeights;
               lambda /= lambdaFactor;
            }

         }

         // compare to old error, change lambda accordingly

         epochs++;
      }

      //System.out.println("FINAL lambda: " + lambda);
      System.out.println("FINAL Error: " + getError(inputs, targets));
      //System.out.println("FINAL Epochs: " + epochs);
      //System.out.println(Arrays.deepToString(weights));
   }

   /**
    * Loops over the weights in the hidden layer and calculates deltas,
    * then loops over the weights in the input layer and calculates deltas.
    *
    * @param output the inputs to the network
    * @param target the target output of the network
    */
   private double[][][] getDeltaWeights(double output, double target)
   {
      double[][][] deltaWeights = initializeJaggedArray();
      double diff = target - output;


      //final weight layer calculations
      for (int j = 0; j < weights[1].length; j++)
      {
         double dots = 0;

         for (int J = 0; J < weights[1].length; J++)
         {
            dots += activations[1][J] * weights[1][J][0];
         }
         deltaWeights[1][j][0] = -diff * dOutFunc(dots) * activations[1][j];
      }

      // using iterators that match the design doc
      for (int k = 0; k < weights[0].length; k++)
      {
         for (int j = 0; j < weights[1].length; j++)
         {
            double dotsK = 0;

            for (int K = 0; K < weights[0].length; K++)
            {
               dotsK += activations[0][K] * weights[0][K][j];
            }

            double dotsJ = 0;
            for (int J = 0; J < weights[1].length; J++)
            {
               dotsJ += activations[1][J] * weights[1][J][0];
            }
            deltaWeights[0][k][j] = -activations[0][k] * dOutFunc(dotsK) * diff * dOutFunc(dotsJ) * weights[1][j][0];
         }
      }

      return deltaWeights;
   }

   /**
    * Method title inspired by Kyle Li.
    * @param outputs
    * @param targets
    * @return
    */
   private double[][][] optimizeWeights(double[] outputs, double[] targets, double lambdaFactor)
   {
      double[][][] newWeights = initializeJaggedArray();
      double diff;
      double dotsJ;
      double dotsK;
      double sumI;
      double dDotsK_sumI;

      diff = targets[0] - outputs[0];
      for (int j = 0; j < weights[1].length; j++)
      {
         dotsJ = 0;
         for (int J = 0; J < weights[1].length; J++)
         {
            dotsJ += activations[1][J] * weights[1][J][0];
         }
         newWeights[1][j][0] = weights[1][j][0] + lambdaFactor * diff * dOutFunc(dotsJ) * activations[1][j];
      }

      for (int j = 0; j < weights[1].length; j++)
      {
         dotsK = 0;

         for (int K = 0; K < weights[0].length; K++)
         {
            dotsK += activations[0][K] * weights[0][K][j];
         }

         dotsJ = 0;
         for (int J = 0; J < weights[1].length; J++)
         {
            dotsJ += activations[1][J] * weights[1][J][0];
         }

         double preSum = dOutFunc(dotsK) * diff * dOutFunc(dotsJ) * weights[1][j][0];

         for (int k = 0; k < weights[0].length; k++)
         {
            newWeights[0][k][j] = weights[0][k][j] + activations[0][k] * preSum;
         }
      }



      /**
      for (int j = 0; j < weights[1].length; j++)
      {
         dotsK = 0;
         for (int K = 0; K < weights[0].length; K++)
         {
            dotsK = activations[0][K] * weights[0][K][j];
         }

         sumI = 0;
         for (int I = 0; I < outputs.length; I++)
         {
            dotsJ = 0;
            for (int J = 0; J < weights[1].length; J++)
            {
               dotsJ += activations[1][J] * weights[1][J][I];
            }

            sumI += (targets[I] - outputs[I]) * dOutFunc(dotsJ) * weights[1][j][I];
         }

         dDotsK_sumI = dOutFunc(dotsK) * sumI;
         for (int k = 0; k < weights[0].length; k++)
         {
            newWeights[0][k][j] = weights[0][k][j] + lambdaFactor * activations[0][k] * dDotsK_sumI;
         }
      }
      */

      return newWeights;
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
      double diff;
      double nodeError;
      double caseError = 0;
      double error = 0;
      double[][] outputs = new double[targets.size()][targets.get(0).length];
      double[] finLayer;
      double[] caseTargets;


      for (int i = 0; i < targets.size(); i++)
      {
         finLayer = run(inputs.get(i));
         caseTargets = targets.get(i);
         caseError = 0;
         for (int j = 0; j < caseTargets.length; j++) //0 0 0 1
         {
            diff = caseTargets[j] - finLayer[j];
            nodeError = diff * diff;
            caseError += nodeError * nodeError / 2;
         }
         error += caseError * caseError;
      }

      return Math.sqrt(error);
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
      double error = 0;

      for (int i = 0; i < target.length; i++)
      {
         diff = target[i] - finLayer[i];
         error += diff * diff;
      }

      return Math.sqrt(error) / 2;
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
