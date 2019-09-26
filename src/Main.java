/**
 * Created by Henry Wiese
 * 9.4.19
 */

import java.io.IOException;

/**
 * The Main class of the project which ONLY contains the main method.
 *
 * All other code pertaining to the network is in Network.java
 */
public class Main
{

   /**
    * Creates a network and trains it.
    */
   public static void main(String[] args) throws IOException
   {

      Network network = new Network();
      network.train(10000, .001);

//      double[] out = network.run();
//      for (double activation : out)
//      {
//         System.out.println(activation);
//      }

   }

}
