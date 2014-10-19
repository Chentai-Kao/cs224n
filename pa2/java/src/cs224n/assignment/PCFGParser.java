package cs224n.assignment;

import cs224n.ling.Tree;
import cs224n.util.Counter;
import cs224n.util.Pair;
import cs224n.util.Triplet;

import java.util.*;

/**
 * The CKY PCFG Parser you will implement.
 */
public class PCFGParser implements Parser {
    private Grammar grammar;
    private Lexicon lexicon;
    // {
    //   "(begin_idx, end_idx)": {
    //     A: (score, Triple(split, B, C))
    //   }
    // }
    // Example (in a cell(i, j)): A -> BC (score: 0.5)
    //                            "(i, j)": { A: (0.5, Triple(split, B, C)) }
    private HashMap<String,
                    HashMap<String,
                            Pair<Double,
                                 Triplet<Integer, String, String>>>> data;

    public void train(List<Tree<String>> trainTrees) {
        // TODO: before you generate your grammar, the training trees
        // need to be binarized so that rules are at most binary
        for (Tree<String>tree : trainTrees) {
          tree = TreeAnnotations.annotateTree(tree);
        }
        lexicon = new Lexicon(trainTrees);
        grammar = new Grammar(trainTrees);
        data = new HashMap<String,
                           HashMap<String,
                                   Pair<Double, Triplet<Integer, String, String>>>>();
    }

    public Tree<String> getBestParse(List<String> sentence) {
        // TODO: implement this method
    	int numWords = sentence.size();
        // Initialize the data.
    	for (int i = 0; i < numWords + 1; ++i) {
    	    for (int j = 0; j < numWords + 1; ++j) {
    	        data.put(getIndexKey(i, j),
    	                 new HashMap<String,
                                     Pair<Double, Triplet<Integer, String, String>>>());
    	    }
    	}
    	for (int i = 0; i < numWords; ++i) {
    	    String word = sentence.get(i);
            // The set of pre-terminals (will be updated later).
            Set<String> preTerminals = new HashSet<String>();
            // Build leaf layer.
    	    for (Grammar.UnaryRule rule : grammar.getUnaryRulesByChild(word)) {
    	        String tag = rule.getParent();
    	        String key = getIndexKey(i, i + 1);
    	        double score = rule.getScore();
    	        data.get(key).put(tag, createScoreAndBackPair(
    	                score, null, null, null));
    	        preTerminals.add(tag); // update the pre-terminals
    	    }
    	    // Handle unaries.
    	    Boolean added = true;
    	    while (added) {
    	        added = false;
    	        for (String B : preTerminals) {
    	            for (Grammar.UnaryRule ruleA : grammar.getUnaryRulesByChild(B)) {
    	                String A = ruleA.getParent();
    	                double prob = ruleA.getScore() * getScoreFromData(i, i + 1, B);
    	                if (prob > getScoreFromData(i, i + 1, A)) {
    	                    String key = getIndexKey(i, i + 1);
    	                    data.get(key).put(A, createScoreAndBackPair(
    	                            prob, null, B, null));
    	                    added = true;
    	                    preTerminals.add(A); // update the pre-terminals
    	                }
    	            }
    	        }
    	    }
    	}
        return null;
    }

    private String getIndexKey(Integer i, Integer j) {
        return "(" + i.toString() + ", " + j.toString() + ")";
    }

    Pair<Double, Triplet<Integer, String, String>> createScoreAndBackPair(
            double score, Integer split, String B, String C) {
        // If this rule is pointing to the actual word, set triplet to null.
        if (split == null && B == null && C == null) {
            return new Pair<Double, Triplet<Integer, String, String>>(
                    score, null);
        }
        // A normal rule (unary or binary).
        Triplet<Integer, String, String> back =
                new Triplet<Integer, String, String>(
                        split, B, C);
        return new Pair<Double, Triplet<Integer, String, String>>(
                        score, back);
    }
    
    double getScoreFromData(Integer begin_idx, Integer end_idx, String A) {
        String key = getIndexKey(begin_idx, end_idx);
        if (!data.get(key).containsKey(A)) {
            return 0.0;
        }
        return data.get(key).get(A).getFirst();
    }
}
