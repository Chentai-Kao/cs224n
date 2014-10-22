package cs224n.assignment;

import cs224n.ling.Tree;
import cs224n.ling.Trees;
import cs224n.util.Counter;
import cs224n.util.Interner;
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
    private IdentityHashMap<String,
    IdentityHashMap<String,
    Pair<Double,
    Triplet<Integer, String, String>>>> data;
    private Interner<String> interner;

    public void train(List<Tree<String>> trainTrees) {
        // TODO: before you generate your grammar, the training trees
        // need to be binarized so that rules are at most binary
        List<Tree<String>> binarizedTrees = new ArrayList<Tree<String>>();
        for (Tree<String> tree : trainTrees) {
            Tree<String> binarizedTree = TreeAnnotations.annotateTree(tree);
            binarizedTrees.add(binarizedTree);
        }
        lexicon = new Lexicon(binarizedTrees);
        grammar = new Grammar(binarizedTrees);
        interner = new Interner<String>();
    }

    public Tree<String> getBestParse(List<String> sentence) {
        // TODO: implement this method
        int numWords = sentence.size();
        // Initialize the data.
        data = new IdentityHashMap<String,
                IdentityHashMap<String,
                Pair<Double, Triplet<Integer, String, String>>>>();
        for (int i = 0; i < numWords + 1; ++i) {
            for (int j = 0; j < numWords + 1; ++j) {
                data.put(getIndexKey(i, j),
                        new IdentityHashMap<String,
                        Pair<Double, Triplet<Integer, String, String>>>());
            }
        }
        // Leaf layer of the matrix.
        for (int i = 0; i < numWords; ++i) {
            String word = sentence.get(i);
            // The set of unaries to fix later.
            Set<String> unariesToFix = new HashSet<String>();
            // Build leaf layer.
            for (String A : lexicon.getAllTags()) {
                double score = lexicon.scoreTagging(word, A);
                if (score > 0.0) {
                    setScoreToData(i, i + 1, A, score, null, null, null);
                    unariesToFix.add(A); // collect the unaries to fix.
                }
            }
            // Handle unaries.
            while (!unariesToFix.isEmpty()) {
                Set<String> newFix = new HashSet<String>();
                for (String B : unariesToFix) {
                    for (Grammar.UnaryRule ruleAB : grammar.getUnaryRulesByChild(B)) {
                        String A = ruleAB.getParent();
                        double prob = ruleAB.getScore() * getScoreFromData(i, i + 1, B);
                        if (prob > getScoreFromData(i, i + 1, A)) {
                            setScoreToData(i, i + 1, A, prob, null, B, null);
                            newFix.add(A); // update the unaries to fix in the next round.
                        }
                    }
                }
                unariesToFix = newFix;
            }
        }
        // Fill all remaining cells of the matrix.
        for (int span = 2; span <= numWords; ++span) {
            for (int begin = 0; begin <= numWords - span; ++begin) {
                int end = begin + span;
                // The set of non-terminals (will be updated later).
                Set<String> unariesToFix = new HashSet<String>();
                // Binary rules.
                for (int split = begin + 1; split <= end - 1; ++split) {
                    for (String B : data.get(getIndexKey(begin, split)).keySet()) {
                        // Iterate all rules (with left child B), do things if C
                        // is right child of the rule. That is, A->BC.
                        for (Grammar.BinaryRule r : grammar.getBinaryRulesByLeftChild(B)) {
                            String C = interner.intern(r.getRightChild());
                            if (data.get(getIndexKey(split, end)).containsKey(C)) {
                                double prob = getScoreFromData(begin, split, B) *
                                              getScoreFromData(split, end, C) *
                                              r.getScore();
                                String A = r.getParent();
                                if (prob > getScoreFromData(begin, end, A)) {
                                    setScoreToData(begin, end, A, prob, split, B, C);
                                    unariesToFix.add(A);
                                }
                            }
                        }
                    }
                }
                // Handle unaries.
                while (!unariesToFix.isEmpty()) {
                    Set<String> newFix = new HashSet<String>();
                    for (String B : unariesToFix) {
                        for (Grammar.UnaryRule ruleAB : grammar.getUnaryRulesByChild(B)) {
                            String A = ruleAB.getParent();
                            double prob = ruleAB.getScore() * getScoreFromData(begin, end, B);
                            if (prob > getScoreFromData(begin, end, A)) {
                                setScoreToData(begin, end, A, prob, null, B, null);
                                newFix.add(A); // update the unaries to fix in the next round.
                            }
                        }
                    }
                    unariesToFix = newFix;
                }
            }
        }
        // Build tree. Find the maximum in the root cell, then build tree.
        String key = getIndexKey(0, numWords);
        String maxA = null;
        Double maxValue = -Double.MAX_VALUE;
        for (String A : data.get(key).keySet()) {
            double score = getScoreFromData(0, numWords, A);
            if (score > maxValue) {
                maxValue = score;
                maxA = A;
            }
        }
        Tree<String> tree = buildTree(0, numWords, maxA, sentence);
        return TreeAnnotations.unAnnotateTree(
                new Tree<String>("ROOT", Collections.singletonList(tree)));
    }

    private String getIndexKey(Integer i, Integer j) {
        return interner.intern("" + i + "_" + j);
    }

    private Pair<Double, Triplet<Integer, String, String>> createScoreAndBackPair(
            double score, Integer split, String B, String C) {
        // If this rule is pointing to the actual word, set triplet to null.
        if (split == null && B == null && C == null) {
            return new Pair<Double, Triplet<Integer, String, String>>(
                    score, null);
        }
        // A normal rule (unary or binary).
        Triplet<Integer, String, String> back =
                new Triplet<Integer, String, String>(split, B, C);
        return new Pair<Double, Triplet<Integer, String, String>>(score, back);
    }

    private Double getScoreFromData(Integer begin_idx, Integer end_idx, String A) {
        String key = getIndexKey(begin_idx, end_idx);
        A = interner.intern(A);
        if (!data.get(key).containsKey(A)) {
            return 0.0;
        }
        return data.get(key).get(A).getFirst();
    }

    private void setScoreToData(Integer begin_idx, Integer end_idx, String A, double score, Integer split, String B, String C) {
        String key = getIndexKey(begin_idx, end_idx);
        A = interner.intern(A);
        data.get(key).put(A, createScoreAndBackPair(score, split, B, C));
    }

    private Triplet<Integer, String, String> getBackPointerFromData(
            Integer begin_idx, Integer end_idx, String A) {
        String key = getIndexKey(begin_idx, end_idx);
        A = interner.intern(A);
        if (!data.get(key).containsKey(A)) {
            return null;
        }
        return data.get(key).get(A).getSecond();
    }

    private Tree<String> buildTree(Integer begin_idx, Integer end_idx, String A, List<String> sentence) {
        Tree<String> tree = new Tree<String>(A);
        List<Tree<String>> children = new ArrayList<Tree<String>>();
        Triplet<Integer, String, String> back = getBackPointerFromData(begin_idx, end_idx, A);
        if (back == null) {
            // Pre-terminal, one layer up from leaf node.
            String word = sentence.get(begin_idx);
            children.add(new Tree<String>(word));
        } else {
            // Get useful variables.
            Integer split = back.getFirst();
            String B = back.getSecond();
            String C = back.getThird();        
            if (split == null && C == null) {
                // Unary rule. Recursively call this function.
                children.add(buildTree(begin_idx, end_idx, B, sentence));
            } else {
                // Binary rule.
                children.add(buildTree(begin_idx, split, B, sentence));
                children.add(buildTree(split, end_idx, C, sentence));
            }
        }
        // Set children and return the tree.
        tree.setChildren(children);
        return tree;
    }
}
