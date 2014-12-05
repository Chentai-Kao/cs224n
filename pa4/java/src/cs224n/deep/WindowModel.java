package cs224n.deep;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

    protected SimpleMatrix U, W, Wout;
    private SimpleMatrix x, y; // current data
    private List<Integer> wordIndices; // Index of selected words. (for reference allVecs)
    private SimpleMatrix p, q, h, z, b1, b2, delta1, delta2; // row-vector, updated by feedForward()
    private double alpha, lambda; // learning rate alpha, regularization lambda
    
    private HashMap<String, SimpleMatrix> labelToY; // mapping from label to y
    private String[] labels = {"O", "LOC", "MISC", "ORG", "PER"};
    
    private boolean gradientCheck; // true to perform gradient check
    private int gradientCheckCount; // sample size for gradient check
    private double gradientCheckEpsilon; // epsilon for gradient check
       
    private boolean outputCost; // true to output cost in SGD
    
    public int windowSize, wordSize, hiddenSize, classSize, wordVectorSize, epochs;

    public WindowModel(double _lambda, double _alpha, int _hiddenSize, int _windowSize, int _epochs){
        //TODO
        wordSize = 50;
        classSize = 5; // output prediction of 5 classes
        windowSize = _windowSize;
        hiddenSize = _hiddenSize;
        wordVectorSize = wordSize * windowSize;
        alpha = _alpha;
        lambda = _lambda;
        epochs = _epochs;
        
        // gradient check
        gradientCheck = false;
        gradientCheckCount = 0;
        gradientCheckEpsilon = 0.0001;

        // labels lookup
        assert labels.length == classSize;
        labelToY = new HashMap<String, SimpleMatrix>();
        for (int i = 0; i < classSize; ++i) {
            SimpleMatrix m = new SimpleMatrix(classSize, 1);
            m.zero();
            m.set(i, 0, 1);
            labelToY.put(labels[i], m);
        }
        
        outputCost = false;
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
        // output cost change in SGD
        PrintWriter costWriter = null;
        if (outputCost) {
            try {
                costWriter = new PrintWriter("../cost.out");
            } catch (FileNotFoundException e) {
                System.err.println("File not found error.");
            }
        }
        
        List<List<Datum>> sentences = extractSentences(_trainData);
        for (int _ = 0; _ < epochs; ++_) {
            if (gradientCheck) {
                for (List<Datum> sentence : sentences) {
                    for (int i = 0; i < sentence.size() - windowSize + 1; ++i) {
                        if (gradientCheckCount >= 10) {
                            return;
                        }
                        buildXY(sentence, i);
                        gradientCheck();
                        ++gradientCheckCount;
                    }
                }
            } else {
                Collections.shuffle(sentences);
                for (List<Datum> sentence : sentences) {
                    for (int i = 0; i < sentence.size() - windowSize + 1; ++i) {
                        buildXY(sentence, i);
                        feedForward();
                        buildDelta();
                        backPropagation();
                        if (costWriter != null) {
                            costWriter.println(calcCost());
                        }
                    }
                }
            }
        }
        if (costWriter != null) {
            costWriter.close();
        }
    }

    public void test(List<Datum> testData){
        // TODO
        List<List<Datum>> sentences = extractSentences(testData);
        try {
            PrintWriter writer = new PrintWriter("../window_model.out");
            for (List<Datum> sentence : sentences) {
                for (int i = 0; i < sentence.size() - windowSize + 1; ++i) {
                    Datum centerWord = sentence.get(i + windowSize / 2);
                    buildXY(sentence, i);
                    feedForward();
                    String label = decideLabel();
                    writer.println(centerWord.word + '\t' + centerWord.label + '\t' + label);
                }
            }
            writer.close();
        } catch (FileNotFoundException e) {
            System.err.println("File not found error.");
        }
    }
    
    // decide label from the current prediction p
    private String decideLabel() {
        int labelIndex = 0;
        double maxValue = -1;
        for (int i = 0; i < p.numRows(); ++i) {
            double v = p.get(i, 0);
            if (v > maxValue) {
                labelIndex = i;
                maxValue = v;
            }
        }
        return labels[labelIndex];
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
        p = p.scale(1 / p.elementSum()); // normalization
    }
    
    private void backPropagation() {
        SimpleMatrix dJdU = calcDJdU();
        SimpleMatrix dJdW = calcDJdW();
        SimpleMatrix dJdB1 = calcDJdB1();
        SimpleMatrix dJdB2 = calcDJdB2();
        SimpleMatrix dJdL = calcDJdL();
        U = U.plus(dJdU.scale(-alpha));
        W = W.plus(dJdW.scale(-alpha));
        b1 = b1.plus(dJdB1.scale(-alpha));
        b2 = b2.plus(dJdB2.scale(-alpha));
        x = x.plus(dJdL.scale(-alpha));
        updateAllVecs();
    }

    private void buildXY(List<Datum> sentence, int start) { 
        // build x. concatenate input vector by [x_{i-1} x_i x_{i+1}]
        x = new SimpleMatrix(windowSize * wordSize, 1);
        wordIndices = new ArrayList<Integer>();
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
            wordIndices.add(wordIndex);
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
        // dJ / dU
        SimpleMatrix dJdU = calcDJdU();
        // add regularized term, lambda * sum_j sum_k U_jk
        elementAdd(dJdU, lambda * U.elementSum());
        // update U by SGD
        U = U.minus(dJdU.scale(alpha));
    }
    
    private SimpleMatrix calcDJdU() {
        return delta2.mult(h.transpose()).plus(U.scale(lambda));
    }
    
    private SimpleMatrix calcDJdW() {
        return delta1.mult(x.transpose()).plus(W.scale(lambda));
    }
    
    private SimpleMatrix calcDJdB1() {
        return delta1;
    }
    
    private SimpleMatrix calcDJdB2() {
        return delta2;
    }

    private SimpleMatrix calcDJdL() {
        return W.transpose().mult(delta1);
    }
    
    private double calcCost() {
        double sum = 0;
        for (int i = 0; i < classSize; ++i) {
            sum -= y.get(i, 0) * Math.log(p.get(i, 0));
        }
        sum += lambda / 2 * (Math.pow(W.normF(), 2) + Math.pow(U.normF(), 2));
        return sum;
    }
    
    // perform elementwise add on the matrix
    private void elementAdd(SimpleMatrix m, double v) {
        for (int i = 0; i < m.numRows(); ++i) {
            for (int j = 0; j < m.numCols(); ++j) {
                m.set(i, j, v + m.get(i, j));
            }
        }
    }
    
    private void gradientCheck() {
        // check U
        feedForward();
        buildDelta();
        SimpleMatrix dJdU = calcDJdU();
        SimpleMatrix diffU = buildDiff(U);
        System.out.println("gradient check U: " + (dJdU.normF() - diffU.normF()));
        // check W
        feedForward();
        buildDelta();
        SimpleMatrix dJdW = calcDJdW();
        SimpleMatrix diffW = buildDiff(W);
        System.out.println("gradient check W: " + (dJdW.normF() - diffW.normF()));
        // check b1
        feedForward();
        buildDelta();
        SimpleMatrix dJdB1 = calcDJdB1();
        SimpleMatrix diffB1 = buildDiff(b1);
        System.out.println("gradient check b1: " + (dJdB1.normF() - diffB1.normF()));
        // check b2
        feedForward();
        buildDelta();
        SimpleMatrix dJdB2 = calcDJdB2();
        SimpleMatrix diffB2 = buildDiff(b2);
        System.out.println("gradient check b2: " + (dJdB2.normF() - diffB2.normF()));
        // check L
        feedForward();
        buildDelta();
        SimpleMatrix dJdL = calcDJdL();
        SimpleMatrix diffL = buildDiff(x);
        System.out.println("gradient check L: " + (dJdL.normF() - diffL.normF()));
    }
    
    private SimpleMatrix buildDiff(SimpleMatrix m) {
        SimpleMatrix diff = new SimpleMatrix(m.numRows(), m.numCols());
        for (int i = 0; i < m.numRows(); ++i) {
            for (int j = 0; j < m.numCols(); ++j) {
                double value = m.get(i, j);
                // positive
                m.set(i, j, value + gradientCheckEpsilon);
                feedForward();
                double pos = calcCost();
                // negative
                m.set(i, j, value - gradientCheckEpsilon);
                feedForward();
                double neg = calcCost();
                diff.set(i, j, (pos - neg) / (2 * gradientCheckEpsilon));
                // recover U
                m.set(i, j, value);
            }
        }
        return diff;
    }
    
    // Update allVecs by the new x.
    private void updateAllVecs() {
        int pos = 0;
        for (Integer wordIndex : wordIndices) {
            for (int i = 0; i < wordSize; ++i) {
                FeatureFactory.allVecs.set(wordIndex, i, x.get(pos, 0));
                ++pos;
            }
        }
    }
}
