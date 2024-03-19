package org.example;

import org.jblas.DoubleMatrix;

import java.io.*;

public class Main {

    public static final String FILE_SERIALIZATION = "net.ser";

    public static Object deserialize() throws IOException, ClassNotFoundException {
        Object obj;
        FileInputStream fileInputStream = new FileInputStream(FILE_SERIALIZATION);
        try (ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream)) {
            obj = objectInputStream.readObject();
        }
        return obj;
    }
    public static void serialize(Object obj) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(FILE_SERIALIZATION);
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream)) {
            objectOutputStream.writeObject(obj);
            objectOutputStream.flush();
        }
        System.out.println("Serialized");
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        double sepal_length = 7.9;
        double sepal_width = 3.1;
        double petal_length = 7.5;
        double petal_width = 1.8;
        SigmoidNetwork net = (SigmoidNetwork) deserialize();

        DoubleMatrix networkInput = new DoubleMatrix(new double[]{sepal_length,sepal_width,petal_length,petal_width});
        System.out.println( doubleMatrixToString( net.feedForward ( networkInput ) ) );
    }

    public static String doubleMatrixToString(DoubleMatrix dm) {
        StringBuilder sb = new StringBuilder();
        for (double d : dm.toArray()) {
            sb.append(d >= 0.5 ? 1 : 0).append(' ');
        }
        return sb.toString();
    }
}