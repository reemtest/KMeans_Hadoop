import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;


public class KMeansReducer extends Reducer<IntWritable, Text, IntWritable, Text> {

    public void reduce(IntWritable key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
    	
    	
    	double[] sum = null;  
        int count = 0;       

        for (Text value : values) {
            String clean = value.toString().replaceAll("\\[|\\]", ""); // Remove brackets
            String[] parts = clean.split(",");
            
        if (sum == null) {
                sum = new double[parts.length];
            }

        for (int i = 0; i < parts.length; i++) {
                sum[i] += Double.parseDouble(parts[i].trim());
            }

            count++; 
        }

        for (int i = 0; i < sum.length; i++) {
            sum[i] /= count;
        }
        
        // Convert array to string
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sum.length; i++) {
        	sb.append(String.format("%.4f", sum[i]));
            if (i < sum.length - 1) sb.append(",");
        }

        // Emit new centroid
        context.write(key, new Text(sb.toString()));
 
            
}
}