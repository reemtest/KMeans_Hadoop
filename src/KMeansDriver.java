import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
	
	
	
	private List<double[]> readCentroids(FileSystem fs, Path path) throws IOException {
	    List<double[]> centroids = new ArrayList<>();
	    BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));
	    String line;
	    while ((line = reader.readLine()) != null) {
	        line = line.trim();
	       
	        String[] tokens = line.contains("\t") ? line.split("\t")[1].split(",") : line.split(",");
	        double[] centroid = Arrays.stream(tokens).mapToDouble(Double::parseDouble).toArray();
	        centroids.add(centroid);
	    }
	    reader.close();
	    return centroids;
	}

	private boolean hasConverged(List<double[]> oldC, List<double[]> newC, double epsilon) {
	    for (int i = 0; i < oldC.size(); i++) {
	        double distance = 0.0;
	        for (int j = 0; j < oldC.get(i).length; j++) {
	            distance += Math.pow(oldC.get(i)[j] - newC.get(i)[j], 2);
	        }
	        if (Math.sqrt(distance) > epsilon) {
	            return false;
	        }
	    }
	    return true;
	}

    
	public int run(String[] args) throws Exception {
		
		 long startTime = System.currentTimeMillis();
	    Configuration conf = getConf();
	    FileSystem fs = FileSystem.get(conf);

	    int maxIterations = Integer.parseInt(args[2]);  
	    double threshold = Double.parseDouble(args[3]); 

	    int iteration = 0;
	    int stableCount = 0;

	    String datasetPath = args[0];        
	    String baseOutputPath = args[1]; 

	    Path centroidFile = new Path("/centroids.txt");

	    if (!fs.exists(centroidFile)) {
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

	        // Read old centroids
	        List<double[]> oldCentroids = readCentroids(fs, centroidFile);

	        // Run the KMeans job
	        boolean success = runKMeansJob(conf, datasetPath, currentOutputPath, iteration);
	        if (!success) {
	            System.err.println("Iteration " + iteration + " failed.");
	            return -1;
	        }

	        // Read new centroids from reducer output
	        Path resultFile = new Path(currentOutputPath + "/part-r-00000");

	        // Clean old centroid file
	        if (fs.exists(centroidFile)) {
	            fs.delete(centroidFile, true);
	        }

	        List<double[]> newCentroids = new ArrayList<>();

	        BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(resultFile)));
	        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fs.create(centroidFile)));

	        String line;
	        while ((line = reader.readLine()) != null) {
	            String[] parts = line.trim().split("\t");
	            if (parts.length == 2) {
	                writer.write(parts[1]);
	                writer.newLine();

	                // Parse the new centroid for convergence check
	                String[] tokens = parts[1].split(",");
	                double[] vector = new double[tokens.length];
	                for (int i = 0; i < tokens.length; i++) {
	                    vector[i] = Double.parseDouble(tokens[i]);
	                }
	                newCentroids.add(vector);
	            }
	        }

	        reader.close();
	        writer.close();

	        // Check convergence
	        if (hasConverged(oldCentroids, newCentroids, threshold)) {
	            stableCount++;
	            if (stableCount >= 2) {
	                System.out.println("Convergence reached after 2 stable iterations at iteration " + iteration);
	                break;
	            }
	        } else {
	            stableCount = 0; // Reset if centroids changed
	        }

	        iteration++;  // Move inside the loop
	    }

	    // Save final centroids to separate file
	    Path finalOutput = new Path("/final_centroids.txt");
	    if (fs.exists(finalOutput)) {
	        fs.delete(finalOutput, true);
	    }
	    fs.rename(new Path("/centroids.txt"), finalOutput);
	    
	    long endTime = System.currentTimeMillis();  // ⏱️ End timing
	    System.out.println("Total execution time: " + (endTime - startTime) + " ms");

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
