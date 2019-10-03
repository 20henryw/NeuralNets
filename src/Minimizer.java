/**
 * Created by Henry Wiese
 * 10.3.19
 */

import java.io.*;
import java.util.ArrayList;

/**
 * This class contains methods that help load training constants and train a network.
 *
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
   ArrayList<Double> targets = new ArrayList<>();
   String[] values;


   public Minimizer(Network network) throws IOException
   {
      this.network = network;
      loadConstants();
      System.out.print("");
   }

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
         targets.add(Double.parseDouble(values[0]));

         line = br.readLine();
         count++;
      }
      br.close();

   }

   public void minimize() throws IOException
   {
      double minError = Double.MAX_VALUE;
      for (int i = 0; i < SUPER_EPOCHS; i++)
      {
         network.randWeights();
         network.train(inputs, targets, lambda, MAX_EPOCHS, lambdaFactor, MIN_LAMBDA, ERROR_THRESHOLD);
         double epochError = network.getError(inputs, targets);
         if (epochError < minError)
         {
            minError = epochError;
         }
      }

      System.out.println("\nMIN ERROR: " + minError);
      for (double input[] : inputs)
      {
         System.out.println(input[0] + " " + input[1] + "| " + network.run(input)[0]);
      }

   }
}
