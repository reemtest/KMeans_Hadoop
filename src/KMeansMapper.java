import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;

public class KMeansMapper extends Mapper<LongWritable, Text,IntWritable, Text> {

    private static double[][] centroids;

    public static double[][] getCentroids() {
        return centroids;
    }

    public static void setCentroids(double[][] newCentroids) {
        centroids = newCentroids;
    }

    
    protected void setup(Context context) throws IOException {
        // Read centroids from HDFS file
        Path path = new Path("/centroids.txt");
        FileSystem fs = FileSystem.get(context.getConfiguration());
        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));

        List<double[]> centroidList = new ArrayList<>();
        String line;
        while ((line = reader.readLine()) != null) {
            String[] tokens = line.trim().split(",");
            double[] centroid = new double[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                centroid[i] = Double.parseDouble(tokens[i]);
            }
            centroidList.add(centroid);
        }
        reader.close();
        centroids = centroidList.toArray(new double[0][]);
    }

  
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        String line = value.toString().trim();
        if (line.isEmpty()) return;

        String[] parts = line.split(",");
        int featureCount = parts.length - 1; 
        double[] dataPoint = new double[featureCount];
        for (int i = 0; i < featureCount; i++) {
            dataPoint[i] = Double.parseDouble(parts[i]);
        }

        
       

        // Find closest centroid
        int closestCentroidId = -1;
        double minDistance = Double.MAX_VALUE;

        for (int i = 0; i < centroids.length; i++) {
            double distance = 0.0;
            for (int j = 0; j < dataPoint.length; j++) {
            	  distance += Math.pow(dataPoint[j] - centroids[i][j], 2);
            }
            distance = Math.sqrt(distance);

        
          
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroidId = i;
            }
        }

        //Closest Centroid
        context.write(new IntWritable(closestCentroidId), new Text(Arrays.toString(dataPoint)));
    }
}
