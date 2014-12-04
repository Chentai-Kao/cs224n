package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

    protected SimpleMatrix U, W, Wout;
    private SimpleMatrix p, q, h, z, b1, b2, delta1, delta2; // row-vector, updated by feedForward() 
    private HashMap<String, SimpleMatrix> labelToY; // mapping from label to y 
    
    public int windowSize, wordSize, hiddenSize, classSize, wordVectorSize;

    public WindowModel(int _windowSize, int _hiddenSize, double _lr){
        //TODO
        wordSize = 50;
        classSize = 5; // output prediction of 5 classes
        windowSize = _windowSize;
        hiddenSize = _hiddenSize;
        wordVectorSize = wordSize * windowSize;
        String[] labels = {"O", "LOC", "MISC", "ORG", "PER"};
        labelToY = new HashMap<String, SimpleMatrix>();
        for (int i = 0; i < 5; ++i) {
            labelToY.put(labels[i], ones(5, 1));
        }
    }

    /**
     * Initializes the weights randomly.
     */
    public void initWeights(){
        //TODO
        // initialize with bias inside as the last column
        // W for the hidden layer
        W = initMatrix(hiddenSize, wordVectorSize);
        // U for the score
        U = initMatrix(classSize, hiddenSize);
        // intercept term
        b1 = initMatrix(hiddenSize, 1);
        b2 = initMatrix(classSize, 1);
    }


    /**
     * Simplest SGD training
     */
    public void train(List<Datum> _trainData){
        // TODO
        List<List<Datum>> sentences = extractSentences(_trainData);
        for (List<Datum> sentence : sentences) {
            int halfWindowSize = windowSize / 2;
            for (int i = 0; i < sentence.size() - windowSize + 1; ++i) {
                SimpleMatrix x = buildX(sentence, i);
                Datum centerWord = sentence.get(i + halfWindowSize);
                SimpleMatrix y = labelToY.get(centerWord.label);
                feedForward(x);
                buildDelta(x, y);
                backPropagation(x, y);
            }
        }
    }


    public void test(List<Datum> testData){
        // TODO
    }
    
    private SimpleMatrix initMatrix(int fanOut, int fanIn) {
        double epsilon = Math.sqrt(6) / Math.sqrt(fanIn + fanOut);
        SimpleMatrix V = SimpleMatrix.random(fanOut, fanIn, -epsilon, epsilon, new Random());
        return V;
    }
    
    // from the start index, find a complete sentence from trainData
    private List<List<Datum>> extractSentences(List<Datum> trainData) {
        List<List<Datum>> sentences = new ArrayList<List<Datum>>();
        for (int i = 0; i < trainData.size(); ++i) {
            // find a complete sentence ["<s>", "word", ..., "</s>"] 
            if (!trainData.get(i).word.equals("<s>")) {
                continue;
            }
            List<Datum> sentence = null;
            for (int j = i; j < trainData.size(); ++j) {
                // find the end of sentence
                if (trainData.get(j).word.equals("</s>")) {
                    sentence = trainData.subList(i, j + 1);
                    break;
                }
            }
            if (sentence != null) {
                sentences.add(sentence);
            }
        }
        return sentences;
    }

    // perform feed forward of data x, update class variables p, q, ..., etc
    private void feedForward(SimpleMatrix x) {
        // h = f(Wx + b1) (element-wise)
        z = W.mult(x).plus(b1);
        h = new SimpleMatrix(hiddenSize, 1);
        for (int i = 0; i < hiddenSize; ++i) {
            h.set(i, 0, Math.tanh(z.get(i, 0)));
        }
        // p = g(Uh + b2) (element-wise)
        q = U.mult(h).plus(b2);
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
    
    private void backPropagation(SimpleMatrix x, SimpleMatrix y) {
        // TODO
    }

    private SimpleMatrix buildX(List<Datum> sentence, int start) { 
        // concatenate input vector by [x_{i-1} x_i x_{i+1}]
        SimpleMatrix x = new SimpleMatrix(windowSize * wordSize, 1);
        int pos = 0;
        for (int i = 0; i < windowSize; ++i) {
            String word = sentence.get(start + i).word;
            if (!FeatureFactory.wordToNum.containsKey(word)) {
                // word not found in vocabulary
                word = "UUUNKKK";
            }
            int wordIndex = FeatureFactory.wordToNum.get(word);
            for (int j = 0; j < wordSize; ++j) {
                x.set(pos, 0, FeatureFactory.allVecs.get(wordIndex, j));
                ++pos;
            }
        }
        return x;
    }
    
    private void buildDelta(SimpleMatrix x, SimpleMatrix y) {
        // delta 2
        delta2 = p.minus(y);
        // delta 1
        SimpleMatrix temp = new SimpleMatrix(hiddenSize, 1);
        for (int i = 0; i < hiddenSize; ++i) {
            temp.set(i, 0, 1 - Math.pow(Math.tanh(z.get(i, 0)), 2));
        }
        delta1 = U.transpose().elementMult(temp);
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
    
    // return a one vector of given dimension
    private SimpleMatrix ones(int numRows, int numCols) {
        SimpleMatrix m = new SimpleMatrix(numRows, numCols);
        m.set(1);
        return m;
    }
}
