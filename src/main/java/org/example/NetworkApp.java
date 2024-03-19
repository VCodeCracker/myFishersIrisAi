package org.example;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class NetworkApp {

    public static void main(String[] args) throws IOException {

        URL url = new URL("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");
        url.openConnection();
        InputStream inputStream = url.openStream();
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, "utf-8"));
        List<double[][]> trainingData = new ArrayList<>();

        String line;
        int k = 0;
        while ((line = reader.readLine()) != null || k == 150) {
            k++;
            if (line.trim().isEmpty()) continue; // Skip empty lines

            String[] split = line.split(",");
            double[][] io = new double[2][];
            double[] x = new double[4];
            double[] y = new double[3];

            for (int i = 0; i < 4; i++) {
                x[i] = Double.parseDouble(split[i]);
            }

            switch(split[4]) {
                case "Iris-setosa":
                    y[0] = 1;
                    y[1] = 0;
                    y[2] = 0;
                    break;
                case "Iris-versicolor":
                    y[0] = 0;
                    y[1] = 1;
                    y[2] = 0;
                    break;
                case "Iris-virginica":
                    y[0] = 0;
                    y[1] = 0;
                    y[2] = 1;
                    break;
            }

            io[0] = x;
            io[1] = y;
            trainingData.add(io);
        }


        SigmoidNetwork net = new SigmoidNetwork(4,5, 3);

        net.SGD(trainingData, 10000, 5, 1, trainingData);

        //training data is correct
    }
}
