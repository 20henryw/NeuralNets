/**
 * Created on 10.24.19 by Henry Wiese
 */

/**
 *
 */
public class ImageWrapper
{
   public int[][] imageArray;

   public ImageWrapper(String imagePath)
   {
      imageArray = DibDump.bmpToArray(imagePath);
   }

   public void toBMP(String outFileName)
   {
      DibDump.imageArrayToBMP(imageArray, outFileName);
   }

   public void setImageArray(double[] pels)
   {
      int height = imageArray.length;
      for (int i = 0; i < height; i++)
      {
         for (int j = 0; j < imageArray[i].length; j++)
         {
            imageArray[i][j] = (int) pels[i * height + j];
         }
      }
   }
}
