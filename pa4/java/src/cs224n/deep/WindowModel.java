package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

    protected SimpleMatrix U, W, Wout;
    //
    public int windowSize, wordSize, hiddenSize, numGrams;

    public WindowModel(int _windowSize, int _hiddenSize, double _lr){
        //TODO
        wordSize = 50;
        numGrams = 3;
        windowSize = _windowSize;
        hiddenSize = _hiddenSize;
    }

    /**
     * Initializes the weights randomly.
     */
    public void initWeights(){
        //TODO
        // initialize with bias inside as the last column
        double epsilonW = Math.sqrt(6) / Math.sqrt(wordSize * numGrams + hiddenSize);
        W = SimpleMatrix.random(hiddenSize, wordSize * numGrams, -epsilonW, epsilonW, new Random());
        // U for the score
        double epsilonU = Math.sqrt(6) / Math.sqrt(hiddenSize + windowSize);
        U = SimpleMatrix.random(windowSize, hiddenSize, -epsilonU, epsilonU, new Random());
    }


    /**
     * Simplest SGD training
     */
    public void train(List<Datum> _trainData){
        // TODO
    }


    public void test(List<Datum> testData){
        // TODO
    }

}
