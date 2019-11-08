/**
 * Created by Henry Wiese
 * 10.3.19
 */

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * This class contains methods that help load training constants and train a network.
 *
 * Minimizer()     - constructs a Minimizer
 * loadConstants() - loads training constants relevant toward training
 * minimize()      - trains the network based on the training constants
 */
public class Minimizer
{
   private Network network;
   private double lambda;
   private int MAX_EPOCHS;
   private double lambdaFactor;
   private double MIN_LAMBDA;
   private double ERROR_THRESHOLD;
   private int SUPER_EPOCHS;
   private String CONSTANTS_PATH = "/Users/henry/Documents/2019-2020/NeuralNets/data/training/constants.csv";
   private String TRAINING_PATH;
   ArrayList<double[]> inputs = new ArrayList<>();
   ArrayList<double[]> targets = new ArrayList<>();
   String[] values;


   /**
    * Creates a Minimizer that can be used to train networks
    * @param network the network to be trained
    * @throws IOException
    */
   public Minimizer(Network network) throws IOException
   {
      this.network = network;
      loadConstants();
   }

   /**
    * Loads constants relevant to minimizing the network.
    * @throws IOException
    */
   private void loadConstants() throws IOException
   {
      File files = new File(CONSTANTS_PATH);
      BufferedReader br = new BufferedReader(new FileReader(files));

      //unused lines are only for visual clarity
      String unusedLine = br.readLine();
      lambda = Double.parseDouble(br.readLine());
      unusedLine = br.readLine();
      MAX_EPOCHS = Integer.parseInt(br.readLine());
      unusedLine = br.readLine();
      lambdaFactor = Double.parseDouble(br.readLine());
      unusedLine = br.readLine();
      MIN_LAMBDA = Double.parseDouble(br.readLine());
      unusedLine = br.readLine();
      ERROR_THRESHOLD = Double.parseDouble(br.readLine());
      unusedLine = br.readLine();
      SUPER_EPOCHS = Integer.parseInt(br.readLine());
      unusedLine = br.readLine();
      TRAINING_PATH = br.readLine();

      br = new BufferedReader(new FileReader(new File(TRAINING_PATH)));

      String line = br.readLine();
      int count = 0;

      while (line != null && line.compareTo("") != 0)
      {
         values = line.split(",");
         inputs.add(new double[values.length]);
         for (int i = 0; i < values.length; i++)
         {
            inputs.get(count)[i] = Double.parseDouble(values[i]);
         }

         line = br.readLine();
         values = line.split(",");
         targets.add(new double[values.length]);
         for (int i = 0; i < values.length; i++)
         {
            targets.get(count)[i] = Double.parseDouble(values[i]);
         }


         line = br.readLine();
         count++;
      }
      br.close();

   }

   /**
    * Minimizes the error of a network by training the network for a number of super epochs.
    * A super epoch is an attempt to train the network with a random set of initial weights.
    * It prints the error of every Super Epoch.
    * It prints the smallest error returned of all the super epochs, runs the neural network with that
    * super epochs's weights, and prints the values in the final layer.
    *
    * @throws IOException
    */
   public double[][][] minimize() throws IOException
   {
      System.out.println("minimize");
      double minError = Double.MAX_VALUE;
      String bestEndCondition = "";
      double epochError = 0;
      String epochEndCondition = "";


      for (int i = 0; i < SUPER_EPOCHS; i++)
      {
         network.randWeights();
         epochEndCondition = network.train(inputs, targets, lambda, MAX_EPOCHS, lambdaFactor, MIN_LAMBDA, ERROR_THRESHOLD);
         epochError = network.getError(inputs, targets);
         System.out.println("SUPER EPOCH ERROR: " + epochError);

         if (epochError < minError)
         {
            minError = epochError;
            bestEndCondition = epochEndCondition;
         }
      }

      System.out.println("\nMIN ERROR: " + minError);
      for (double input[] : inputs)
      {
         System.out.println(Arrays.toString(network.run(input)));
      }

      System.out.println("\n" + bestEndCondition);

      return network.getWeights();
   }

   public void toTestBMP(String outFilePath, double[][][] weights) throws IOException
   {
      ImageWrapper testWrapper = new ImageWrapper( "/Users/henry/Documents/2019-2020/NeuralNets/data/training/test2.bmp");
      network.setWeights(weights);
      double[] output = network.run(inputs.get(0));
      testWrapper.toBMP(outFilePath);
   }
}
