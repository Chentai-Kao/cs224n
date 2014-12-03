package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

    protected SimpleMatrix U, W, Wout;
    private SimpleMatrix p, q, h, z; // all row-vector
    
    public int windowSize, wordSize, hiddenSize, classSize, wordVectorSize;

    public WindowModel(int _windowSize, int _hiddenSize, double _lr){
        //TODO
        wordSize = 50;
        classSize = 5; // output prediction of 5 classes
        windowSize = _windowSize;
        hiddenSize = _hiddenSize;
        wordVectorSize = wordSize * windowSize;
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
        U = initMatrix(hiddenSize, classSize);
    }


    /**
     * Simplest SGD training
     */
    public void train(List<Datum> _trainData){
        // TODO
        for (int i = 0; i < _trainData.size(); ++i) {
            System.out.println(_trainData.get(i).toString());
        }
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

    // perform feed forward of data x, update class variables p, q, ..., etc
    private void feedForward(SimpleMatrix x) {
        // h = f(Wx + b1) (element-wise)
        z = W.mult(x);
        h = new SimpleMatrix(hiddenSize, 1);
        for (int i = 0; i < hiddenSize; ++i) {
            h.set(i, 0, Math.tanh(z.get(i, 0)));
        }
        // p = g(Uh + b2) (element-wise)
        q = U.mult(h);
        p = new SimpleMatrix(classSize, 1);
        for (int i = 0; i < classSize; ++i) {
            p.set(i, 0, Math.exp(q.get(i, 0))); // normalize later
        }
        double norm = 0; // normalization
        for (int i = 0; i < classSize; ++i) {
            norm += p.get(i);
        }
        p.scale(1 / norm);
    }

    private SimpleMatrix buildX(List<String> words) {
        assert words.size() == windowSize; 
        // concatenate input vector by [x_{i-1} x_i x_{i+1}]
        SimpleMatrix x = new SimpleMatrix(windowSize, 1);
        for (int i = 0; i < windowSize; ++i) {
            String word = words.get(i);
            int wordIndex = FeatureFactory.wordToNum.get(word);
            int offset = i * wordSize;
            for (int j = 0; j < wordSize; ++j) {
                x.set(offset + j, 0, FeatureFactory.allVecs.get(wordIndex, j));
            }
        }
        return x;
    }
    
    // given data (x, y), update U by SGD of dJ_R / dU.
    // labmda: parameter of regularized term.
    // alpha: SGD learning rate.
    private void updateU(SimpleMatrix x, SimpleMatrix y, double lambda, double alpha) {
        // calculate dJR / dU
        SimpleMatrix dJRdU = new SimpleMatrix(U.numRows(), U.numCols());
        dJRdU.zero();
        double reg = lambda * U.elementSum(); // regularized term, lambda * sum_j sum_k U_jk
        for (int k = 0; k < dJRdU.numCols(); ++k) {
            // part of unregularized term, "sum_j y_j(1-p_j)"
            double partialSum = 0;
            for (int j = 0; j < dJRdU.numRows(); ++j) {
                partialSum += y.get(j, 0) * (1 - p.get(j, 0));
            }
            double unreg = -partialSum * h.get(k, 0);
            // update elements of dJRdU_jk
            for (int j = 0; j < dJRdU.numRows(); ++j) {
                dJRdU.set(j, k, unreg + reg);
            }
        }
        // update U by SGD
        U = U.minus(dJRdU.scale(alpha));
    }
}