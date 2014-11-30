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
        numGrams = 3; // tri-gram model
        windowSize = _windowSize;
        hiddenSize = _hiddenSize;
    }

    /**
     * Initializes the weights randomly.
     */
    public void initWeights(){
        //TODO
        // initialize with bias inside as the last column
        // W for the hidden layer
        W = initMatrix(wordSize * numGrams, hiddenSize);
        // U for the score
        U = initMatrix(hiddenSize, windowSize);
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
    
    private SimpleMatrix initMatrix(int fanIn, int fanOut) {
        double epsilon = Math.sqrt(6) / Math.sqrt(fanIn + fanOut);
        SimpleMatrix V = SimpleMatrix.random(fanIn, fanOut + 1, -epsilon, epsilon, new Random());
        // initialize bias to zero
        for (int i = 0; i < fanIn; ++i) {
            V.set(i, fanOut - 1, 0);
        }
        return V;
    }

}
