package cs224n.corefsystems;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.coref.Mention;
import cs224n.util.Pair;
import cs224n.util.StringUtils;

public class RuleBased implements CoreferenceSystem {

    @Override
    public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
        // TODO Auto-generated method stub

    }

    @Override
    public List<ClusteredMention> runCoreference(Document doc) {
        // TODO Auto-generated method stub
        // Initialize all mention as singleton.
        Map <Mention, ClusteredMention> clusters = new HashMap<Mention, ClusteredMention>();
        for (Mention m : doc.getMentions()) {
            clusters.put(m, m.markSingleton());
        }
        // Create all combination pair of mentions. Easy to loop through.
        List<Pair<Mention, Mention>> mentionPairs = new ArrayList<Pair<Mention, Mention>>();
        for (Mention a : doc.getMentions()) {
            for (Mention b : doc.getMentions()) {
                mentionPairs.add(new Pair<Mention, Mention>(a, b));
            }
        }
        // Multi-pass sieve (NLP 10')
        pass1(mentionPairs, clusters);
        pass2(mentionPairs, clusters);
        // Return the mentions
        return new ArrayList<ClusteredMention>(clusters.values()); 
    }

    // Check whether two mentions are already marked coreferent.
    private boolean isCoreferent(Map <Mention, ClusteredMention> clusters,
            Mention a, Mention b) {
        return clusters.get(a).entity == clusters.get(b).entity;
    }

    // Mark a and b as coreferent.
    private void updateCoreferent(Map <Mention, ClusteredMention> clusters,
            Mention a, Mention b) {
        if (isCoreferent(clusters, a, b)) {
            return;
        }
        // Merge "all mentions with the same entity as b" to a.
        Set<Mention> mentionsToMerge = clusters.get(b).entity.mentions;
        for (Mention m : mentionsToMerge) {
            m.removeCoreference();
            clusters.put(m, m.markCoreferent(clusters.get(a)));
        }
    }

    private void pass1(List<Pair<Mention, Mention>> mentionPairs,
            Map <Mention, ClusteredMention> clusters) {
        for (Pair<Mention, Mention> mentionPair : mentionPairs) {
            Mention a = mentionPair.getFirst();
            Mention b = mentionPair.getSecond();
            if (isCoreferent(clusters, a, b)) {
                continue;
            }
            if (a.headToken().isNoun() && b.headToken().isNoun() &&
                    a.gloss().equals(b.gloss())) {
                updateCoreferent(clusters, a, b);
            }
        }
    }

    private void pass2(List<Pair<Mention, Mention>> mentionPairs,
            Map <Mention, ClusteredMention> clusters) {
        for (Pair<Mention, Mention> mentionPair : mentionPairs) {
            Mention a = mentionPair.getFirst();
            Mention b = mentionPair.getSecond();
            if (isCoreferent(clusters, a, b)) {
                continue;
            }
            if (a.headToken().posTag().equals("RP")) {
                System.out.println(a.gloss());
            }
            if (false || // TODO apposition
                isPredicateNominative(a, b) || // TODO predicate nominative
                false || // TODO role appositive
                isRelativePronoun(a, b) || // TODO relative pronoun
                false || // TODO acronym
                false) { // TODO demonym
                updateCoreferent(clusters, a, b);
            }
        }
    }
    
    private boolean isPredicateNominative(Mention a, Mention b) {
        return (a.sentence == b.sentence) &&
               (a.headToken().isNoun() && b.headToken().isNoun()) &&
               (b.beginIndexInclusive - a.endIndexExclusive == 1) &&
               a.sentence.words.get(a.endIndexExclusive).equals("is");
    }
    
    private boolean isRelativePronoun(Mention a, Mention b) {
        return (a.headToken().isNoun() && b.headToken().isNoun());
    }
}
