package cs224n.wordaligner;  

import cs224n.util.*;

import java.util.List;

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
public class IBMM1 implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  
  // TODO: Use arrays or Counters for collecting sufficient statistics
  // from the training data.
  private CounterMap<String,String> sourceTargetCounts;
  private CounterMap<String,String> sourceTargetProb;

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
    // Start with P(f|e) uniform, including p(f|NULL).
    sourceTargetCounts = new CounterMap<String,String>();
    sourceTargetProb = new CounterMap<String,String>();
    for(SentencePair pair : trainingPairs){
      List<String> sourceWords = pair.getSourceWords();
      List<String> targetWords = pair.getTargetWords();
      double uniformValue = 1.0 / (targetWords.size() + 1);
      for(String source : sourceWords){
        for(String target : targetWords){
          sourceTargetProb.setCount(source, target, uniformValue);
        }
        sourceTargetProb.setCount(source, IBMM1.NULL_WORD, uniformValue);
      }
    }
    // Iterate EM until convergence.
    int numIter = 50;
    for(int iter = 0; iter < numIter; ++iter){
      // E-step.
      for(SentencePair pair : trainingPairs){
        List<String> sourceWords = pair.getSourceWords();
        List<String> targetWords = pair.getTargetWords();
        for(String target : targetWords){
          double sumP = 0.0;
          for(String k : sourceTargetProb.keySet()){
            sumP += sourceTargetProb.getCount(k, target);
          }
          for(String source : sourceWords){
            double p = sourceTargetProb.getCount(source, target) / sumP;
            sourceTargetCounts.incrementCount(source, target, p);
          }
        }
      }
      // M-step.
      sourceTargetProb = Counters.conditionalNormalize(sourceTargetCounts);
    }
  }
}
