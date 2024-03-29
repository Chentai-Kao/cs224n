package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {

    public static void main(String[] args) throws IOException {
        if (args.length < 7) {
            System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
            return;
        }

        // this reads in the train and test datasets
        List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
        List<Datum> testData = FeatureFactory.readTestData(args[1]);
        
        // parameters
        double lambda = Double.parseDouble(args[2]);
        double alpha = Double.parseDouble(args[3]);
        int hiddenSize = Integer.parseInt(args[4]);
        int windowSize = Integer.parseInt(args[5]);
        int epochs = Integer.parseInt(args[6]);

        //	read the train and test data
        //TODO: Implement this function (just reads in vocab and word vectors)
        FeatureFactory.initializeVocab("../data/vocab.txt");
        SimpleMatrix allVecs = FeatureFactory.readWordVectors("../data/wordVectors.txt");

        // initialize model
        //BaselineModel model = new BaselineModel();
        //WindowModel model = new WindowModel(5, 100, 0.001);
        //WindowModel model = new WindowModel(3, 100, 0.001);
        WindowModel model = new WindowModel(lambda, alpha, hiddenSize, windowSize, epochs);
        model.initWeights();

        //TODO: Implement those two functions
        model.train(trainData);
        model.test(testData);
    }
}
