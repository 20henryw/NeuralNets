/**
 * Created by Henry Wiese
 * 9.4.19
 */

public class Main {

   public static void main(String[] args) {

      Network network = new Network();
      System.out.println(network.propagate(new double[]{1, 1})[0]);

   }

}