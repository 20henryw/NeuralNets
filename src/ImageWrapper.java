/**
 * Created on 10.24.19 by Henry Wiese
 */

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 *
 */
public class ImageWrapper
{
   public int[][] imageArray;

   public ImageWrapper(String imagePath)
   {
      imageArray = DibDump.bmpToArray(imagePath);
      int height = imageArray.length;
      int max = 0;
      for (int i = 0; i < height; i++)
      {
         for (int j = 0; j < imageArray[i].length; j++)
         {
            imageArray[i][j] &= 0x00FFFFFF;
            if (imageArray[i][j] > max) max = imageArray[i][j];
         }
      }

      System.out.printf("%08x ", max);
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
            imageArray[i][j] = (int) (pels[i * height + j]);
         }
      }
   }

   /**
    * Writes the array to a training file in the format used by my project.
    * @param outFilePath File path to the new file that will contain the training data
    * @throws IOException
    */
   public void createTrainingFile(String outFilePath) throws IOException
   {
      BufferedWriter writer = new BufferedWriter(new FileWriter(outFilePath));
      String outString = "";
      int max = 0;

      int height = imageArray.length;
      for (int i = 0; i < height; i++)
      {
         for (int j = 0; j < imageArray[i].length; j++)
         {
            if (imageArray[i][j] > max)
            {
               max = imageArray[i][j];
            }
            outString += (imageArray[i][j] / 16777216.0) + ",";
         }
      }

      outString = outString.substring(0, outString.length() - 1);
      writer.write(outString + "\n");
      writer.write(outString);
      writer.close();

   }

}
