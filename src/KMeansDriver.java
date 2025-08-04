import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class KMeansDriver extends Configured implements Tool {

    
	public int run(String[] args) throws Exception {
	    Configuration conf = getConf();
	    FileSystem fs = FileSystem.get(conf);

	    int maxIterations = 3;
	    int iteration = 0;

	    String datasetPath = args[0];        
	    String baseOutputPath = args[1];     

	   
	    if (!fs.exists(new Path("/centroids.txt"))) {
	        System.err.println("Initial centroid file /centroids.txt not found in HDFS.");
	        return -1;
	    }

	    while (iteration < maxIterations) {
	        String currentOutputPath = baseOutputPath + iteration;

	        // Delete previous output if it exists
	        Path currentOutput = new Path(currentOutputPath);
	        if (fs.exists(currentOutput)) {
	            fs.delete(currentOutput, true);
	        }

	        // Run the KMeans job
	        boolean success = runKMeansJob(conf, datasetPath, currentOutputPath, iteration);
	        if (!success) {
	            System.err.println("Iteration " + iteration + " failed.");
	            return -1;
	        }

	        // Read the reducer output and update /centroids.txt
	        Path resultFile = new Path(currentOutputPath + "/part-r-00000");
	        Path centroidFile = new Path("/centroids.txt");

	        // Clean old centroid file
	        if (fs.exists(centroidFile)) {
	            fs.delete(centroidFile, true);
	        }

	        // Extract centroid vectors (without the cluster index)
	        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(resultFile)));
	        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fs.create(centroidFile)));

	        String line;
	        while ((line = reader.readLine()) != null) {
	            String[] parts = line.trim().split("\t");
	            if (parts.length == 2) {
	                writer.write(parts[1]);
	                writer.newLine();
	            }
	        }

	        reader.close();
	        writer.close();

	        iteration++;
	    }

	    // Save the final centroids separately
	    Path finalOutput = new Path("/final_centroids.txt");
	    if (fs.exists(finalOutput)) {
	        fs.delete(finalOutput, true);
	    }
	    fs.rename(new Path("/centroids.txt"), finalOutput);

	    return 0;
	}
    	


    
    
   private boolean runKMeansJob(Configuration conf, String inputPath, String outputPath, int iteration) throws Exception {
        Job job = Job.getInstance(conf, "KMeans Iteration " + iteration);
        job.setJarByClass(KMeansDriver.class);

        job.setMapperClass(KMeansMapper.class);
        job.setReducerClass(KMeansReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        return job.waitForCompletion(true);
    }

    public static void main(String[] args) throws Exception {
    	
        int exitCode = ToolRunner.run(new KMeansDriver(), args);
        System.exit(exitCode);
    }
}
