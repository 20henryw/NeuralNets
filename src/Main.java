import java.io.IOException;

/**
 * Created by Henry Wiese
 * 9.4.19
 */

public class Main {

   public static void main(String[] args) throws IOException {

      Network network = new Network();
      System.out.println(network.propagate()[0]);
   }

}
