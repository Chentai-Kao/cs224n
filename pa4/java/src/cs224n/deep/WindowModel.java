package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

    protected SimpleMatrix U, W, Wout;
    //
    public int windowSize, wordSize, hiddenSize, numGrams, wordVectorSize;

    public WindowModel(int _windowSize, int _hiddenSize, double _lr){
        //TODO
        wordSize = 50;
        numGrams = 3; // tri-gram model
        wordVectorSize = wordSize * numGrams;
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
        W = initMatrix(wordVectorSize, hiddenSize);
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
    
    private SimpleMatrix feedForward(List<String> words) {
        SimpleMatrix x = buildInputVector(words);
        // h = f(Wx + b1) (element-wise)
        SimpleMatrix z = W.mult(x);
        SimpleMatrix h = new SimpleMatrix(hiddenSize, 1);
        for (int i = 0; i < hiddenSize; ++i) {
            h.set(i, 0, Math.tanh(z.get(i, 0)));
        }
        // p = g(Uh + b2) (element-wise)
        SimpleMatrix c = U.mult(h);
        SimpleMatrix p = new SimpleMatrix(windowSize, 1);
        for (int i = 0; i < windowSize; ++i) {
            p.set(i, 0, Math.exp(c.get(i, 0))); // normalize later
        }
        double norm = 0; // normalization
        for (int i = 0; i < windowSize; ++i) {
            norm += p.get(i);
        }
        p.scale(1 / norm);
        return p;
    }

    private SimpleMatrix buildInputVector(List<String> words) {
        // concatenate input vector by [x_{i-1} x_i x_{i+1}]
        SimpleMatrix x = new SimpleMatrix(wordSize * words.size() + 1, 1);
        for (int i = 0; i < words.size(); ++i) {
            String word = words.get(i);
            if (FeatureFactory.wordToNum.containsKey(word)) {
                int wordIndex = FeatureFactory.wordToNum.get(word);
                int offset = i * wordSize;
                for (int j = 0; j < wordSize; ++j) {
                    x.set(offset + j, 0, FeatureFactory.allVecs.get(wordIndex, j));
                }
            }
        }
        return x;
    }
}
