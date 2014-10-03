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
public class PMIModel implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  
  // TODO: Use arrays or Counters for collecting sufficient statistics
  // from the training data.
  private Counter<String> sourceCounts;
  private Counter<String> targetCounts;
  private CounterMap<String,String> sourceTargetCounts;

  public Alignment align(SentencePair sentencePair) {
    // Placeholder code below. 
    // TODO Implement an inference algorithm for Eq.1 in the assignment
    // handout to predict alignments based on the counts you collected with train().
    Alignment alignment = new Alignment();
    int numSourceWords = sentencePair.getSourceWords().size();
    int numTargetWords = sentencePair.getTargetWords().size();

    for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
      String targetWord = sentencePair.getTargetWords().get(tgtIndex);
      double pTgt = targetCounts.getCount(targetWord) / targetCounts.totalCount();
      double maxP = 0.0;
      int maxSrcIndex = -1;
      for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
        String sourceWord = sentencePair.getSourceWords().get(srcIndex);
        double pSrcTgt = sourceTargetCounts.getCount(sourceWord, targetWord) / sourceTargetCounts.totalCount();
        double pSrc = sourceCounts.getCount(sourceWord) / sourceCounts.totalCount();
        double p = java.lang.Math.log(pSrcTgt) - java.lang.Math.log(pSrc) - java.lang.Math.log(pTgt);
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
    sourceCounts = new Counter<String>();
    targetCounts = new Counter<String>();
    sourceTargetCounts = new CounterMap<String,String>();
    for(SentencePair pair : trainingPairs){
      List<String> sourceWords = pair.getSourceWords();
      List<String> targetWords = pair.getTargetWords();
      for(String source : sourceWords){
        sourceCounts.incrementCount(source, 1.0);
      }
      for(String target : targetWords){
        targetCounts.incrementCount(target, 1.0);
      }
      for(String source : sourceWords){
        for(String target : targetWords){
          // TODO: Warm-up. Your code here for collecting sufficient statistics.
          sourceTargetCounts.incrementCount(source, target, 1.0);
        }
      }
    }
  }
}
