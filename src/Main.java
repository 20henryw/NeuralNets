/**
 * Created by Henry Wiese
 * 9.4.19
 */

import java.io.IOException;

/**
 * The Main class of the project that contains the main method
 */
public class Main {

   /**
    * Creates a network, propagates it, and prints every activation of the output layer.
    */
   public static void main(String[] args) throws IOException {

      Network network = new Network();
      double[] out = network.propagate();
      for (double activation : out) {
         System.out.println(activation);
      }

   }

}
