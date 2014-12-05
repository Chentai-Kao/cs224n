package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

    protected SimpleMatrix U, W, Wout;
    private SimpleMatrix x, y; // current data
    private SimpleMatrix p, q, h, z, b1, b2, delta1, delta2; // row-vector, updated by feedForward()
    private double alpha, lambda; // learning rate alpha, regularization lambda
    private HashMap<String, SimpleMatrix> labelToY; // mapping from label to y 
    
    public int windowSize, wordSize, hiddenSize, classSize, wordVectorSize;

    public WindowModel(int _windowSize, int _hiddenSize, double _lr){
        //TODO
        wordSize = 50;
        classSize = 5; // output prediction of 5 classes
        windowSize = _windowSize;
        hiddenSize = _hiddenSize;
        wordVectorSize = wordSize * windowSize;
        alpha = 0.001;
        lambda = 1;
        String[] labels = {"O", "LOC", "MISC", "ORG", "PER"};
        assert labels.length == classSize;
        labelToY = new HashMap<String, SimpleMatrix>();
        for (int i = 0; i < classSize; ++i) {
            SimpleMatrix m = new SimpleMatrix(classSize, 1);
            m.zero();
            m.set(i, 0, 1);
            labelToY.put(labels[i], m);
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
            for (int i = 0; i < sentence.size() - windowSize + 1; ++i) {
                buildXY(sentence, i);
                feedForward();
                buildDelta();
                backPropagation();
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
    private void feedForward() {
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
    
    private void backPropagation() {
        // TODO
    }

    private void buildXY(List<Datum> sentence, int start) { 
        // build x. concatenate input vector by [x_{i-1} x_i x_{i+1}]
        x = new SimpleMatrix(windowSize * wordSize, 1);
        int pos = 0;
        for (int i = 0; i < windowSize; ++i) {
            String word = sentence.get(start + i).word.toLowerCase();
            if (!FeatureFactory.wordToNum.containsKey(word)) {
                // word not found in vocabulary
                word = "UUUNKKK";
            } else if (word.matches("[.0-9]+")) {
                // contains only digit and period, convert each digit to "DG"
                word = word.replaceAll("[0-9]", "DG");
            }
            int wordIndex = FeatureFactory.wordToNum.get(word);
            for (int j = 0; j < wordSize; ++j) {
                x.set(pos, 0, FeatureFactory.allVecs.get(wordIndex, j));
                ++pos;
            }
        }
        // build y
        Datum centerWord = sentence.get(start + windowSize / 2);
        y = labelToY.get(centerWord.label);
    }
    
    private void buildDelta() {
        // delta 2
        delta2 = p.minus(y);
        // delta 1
        SimpleMatrix temp = new SimpleMatrix(hiddenSize, 1);
        for (int i = 0; i < hiddenSize; ++i) {
            temp.set(i, 0, 1 - Math.pow(Math.tanh(z.get(i, 0)), 2));
        }
        delta1 = U.transpose().mult(delta2).elementMult(temp);
    }
    
    // given data (x, y), update U by SGD of dJ_R / dU.
    private void updateU() {
        // dJR / dU
        SimpleMatrix dJdU = calc_unreg_dJdU();
        // add regularized term, lambda * sum_j sum_k U_jk
        elementAdd(dJdU, lambda * U.elementSum());
        // update U by SGD
        U = U.minus(dJdU.scale(alpha));
    }
    
    private SimpleMatrix calc_unreg_dJdU() {
        return delta2.mult(h.transpose()); // TODO not sure whether * (1 / m)?
    }
    
    private SimpleMatrix calc_unreg_dJdW() {
        return delta2.mult(h.transpose()); // TODO not sure whether * (1 / m)?
    }
    
    // return a one vector of given dimension
    private SimpleMatrix ones(int numRows, int numCols) {
        SimpleMatrix m = new SimpleMatrix(numRows, numCols);
        m.set(1);
        return m;
    }
    
    // perform elementwise add on the matrix
    private void elementAdd(SimpleMatrix m, double v) {
        for (int i = 0; i < m.numRows(); ++i) {
            for (int j = 0; j < m.numCols(); ++j) {
                m.set(i, j, v + m.get(i, j));
            }
        }
    }
}
