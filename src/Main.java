/**
 * Created by Henry Wiese
 * 9.4.19
 */

import java.io.IOException;
import java.util.Arrays;

/**
 * The Main class of the project which ONLY contains the main method.
 *
 * All other code pertaining to the network is in Network.java
 */
public class Main
{

   /**
    * Creates a network based on a command line file path argument
    * If no argument is given, the default file path is used.
    *
    * Then, trains the network.
    */
   public static void main(String[] args) throws IOException
   {
      String filepath = "/Users/henry/Documents/2019-2020/NeuralNets/data/files.csv"; //default settings file path
      if (args.length != 0)
      {
         filepath = args[0];
      }

      Network network = new Network(filepath);
      Minimizer mini = new Minimizer(network);

      mini.minimize();

//      System.out.println("hi");
//
//      ImageWrapper wrapper = new ImageWrapper( "/Users/henry/Documents/2019-2020/NeuralNets/data/training/test1.bmp");
//      wrapper.createTrainingFile("/Users/henry/Documents/2019-2020/NeuralNets/data/training/bmpTraining.csv");
//      wrapper.toBMP("/Users/henry/Documents/2019-2020/NeuralNets/data/training/test2.bmp");

//      mini.toTestBMP("/Users/henry/Documents/2019-2020/NeuralNets/data/training/chaituOut.bmp", network.getWeights());
//      mini.toTestBMP("/Users/henry/Documents/2019-2020/NeuralNets/data/training/minimizeOut.bmp", mini.minimize());
   }

}
