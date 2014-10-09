package cs224n.wordaligner;  

import cs224n.util.*;

import java.util.List;
import java.util.Random;

/**
 * Simple word alignment baseline model that maps source positions to target 
 * positions along the diagonal of the alignment grid.
 * 
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 * 
 * @author Dan Klein
 * @author Spence Green
 */
public class IBMM2 implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  
  // TODO: Use arrays or Counters for collecting sufficient statistics
  // from the training data.
  private CounterMap<String,String> sourceTargetCounts;
  private CounterMap<String,String> sourceTargetProb;
  private CounterMap<String,String> alignmentCounts; // count(j,i,n,m)
  private CounterMap<String,String> alignmentProb; // q

  public Alignment align(SentencePair sentencePair) {
    // Placeholder code below. 
    // TODO Implement an inference algorithm for Eq.1 in the assignment
    // handout to predict alignments based on the counts you collected with train().
    Alignment alignment = new Alignment();
    int numSourceWords = sentencePair.getSourceWords().size();
    int numTargetWords = sentencePair.getTargetWords().size();

    for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
      String targetWord = sentencePair.getTargetWords().get(tgtIndex);
      double maxP = 0.0;
      int maxSrcIndex = -1;
      for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
        String sourceWord = sentencePair.getSourceWords().get(srcIndex);
        double p = sourceTargetProb.getCount(sourceWord, targetWord);
        if (p > maxP) {
          maxP = p;
          maxSrcIndex = srcIndex;
        }
      }
      alignment.addPredictedAlignment(tgtIndex, maxSrcIndex);
    }
    return alignment;
  }

  public void train(List<SentencePair> trainingPairs) {
    // Initialize with Model 1.
    IBMM1 model1 = new IBMM1();
    model1.train(trainingPairs);
    sourceTargetProb = model1.getSourceTargetProb();
    // Initialize q with random number.
    alignmentProb = new CounterMap<String,String>();
    Random random = new Random();
    for(SentencePair pair : trainingPairs){
      List<String> sourceWords = pair.getSourceWords();
      List<String> targetWords = pair.getTargetWords();
      int sourceLength = sourceWords.size();
      int targetLength = targetWords.size();
      for(String target : targetWords){
        String code = encodeKey(target, sourceLength, targetLength);
        for(String source : sourceWords){
          alignmentProb.setCount(code, source, random.nextFloat());
        }
      }
    }
    // Iterate EM until convergence.
    int numIter = 20;
    for(int iter = 0; iter < numIter; ++iter){
      System.out.println(Integer.toString(iter));
      sourceTargetCounts = new CounterMap<String,String>();
      alignmentCounts = new CounterMap<String,String>();
      // E-step.
      for(SentencePair pair : trainingPairs){
        List<String> sourceWords = pair.getSourceWords();
        List<String> targetWords = pair.getTargetWords();
        int sourceLength = sourceWords.size();
        int targetLength = targetWords.size();
        for(String target : targetWords){
          String code = encodeKey(target, sourceLength, targetLength);
          double sumP = 0.0;
          for(String source : sourceWords){
            sumP += alignmentProb.getCount(code, source) *
                    sourceTargetProb.getCount(source, target);
          }
          for(String source : sourceWords){
            double delta = alignmentProb.getCount(code, source) *
                           sourceTargetProb.getCount(source, target) /
                           sumP;
            sourceTargetCounts.incrementCount(source, target, delta);
            alignmentCounts.incrementCount(code, source, delta);
          }
        }
      }
      // M-step.
      sourceTargetProb = Counters.conditionalNormalize(sourceTargetCounts);
      alignmentProb = Counters.conditionalNormalize(alignmentCounts);
    }
  }

  private String encodeKey(String source, int sourceLength, int targetLength) {
    return source + "_" + sourceLength + "_" + targetLength;
  }
}
