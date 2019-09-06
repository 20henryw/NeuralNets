/**
 * Created by Henry Wiese
 * 9.4.19
 */

public class Network
{
   private int[] layers;
   private int numLayers;
   private int MAX_LAYER_SIZE = 2;
   private double[][] activations;
   private double[][][] weights;

   public Network() {

      layers = new int[]{2, 2, 1};

      numLayers = layers.length;
      activations = new double[numLayers][MAX_LAYER_SIZE];
      weights = new double[numLayers][MAX_LAYER_SIZE][MAX_LAYER_SIZE];
      fillWeights();
   }

   private void fillWeights() {

      // numLayers -1 to avoid index out of bound errors, since you don't calculate weights from the output layer
      for (int layer = 0; layer < numLayers - 1; layer++) {
         for (int i = 0; i < layers[layer]; i++) {
            for (int j = 0; j < layers[layer + 1]; j++) {
               weights[layer][i][j] = Math.random() - 0.5;
            }
         }
      }

      System.out.println("pause");
   }
   public double[] propagate(double[] inputs) {
      double[] output = new double[layers[numLayers -1]];

      for (int i = 0; i < inputs.length; i++) {
         activations[0][i] = inputs[i];
      }

      for (int layer = 1; layer < numLayers; layer++) {

      }
      output[0] = 0;
      return output;
   }

}
