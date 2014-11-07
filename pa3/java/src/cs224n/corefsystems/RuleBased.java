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
        a.removeCoreference();
        clusters.put(a, a.markCoreferent(clusters.get(b)));
    }

    private void pass1(List<Pair<Mention, Mention>> mentionPairs,
            Map <Mention, ClusteredMention> clusters) {
        for (Pair<Mention, Mention> mentionPair : mentionPairs) {
            Mention a = mentionPair.getFirst();
            Mention b = mentionPair.getSecond();
            if (isCoreferent(clusters, a, b)) {
                continue;
            }
            if (a.gloss().equals(b.gloss())) {
                updateCoreferent(clusters, a, b);
            }
        }
    }

    private void pass2(List<Pair<Mention, Mention>> mentionPairs,
            Map <Mention, ClusteredMention> clusters) {
        for (Pair<Mention, Mention> mentionPair : mentionPairs) {
            Mention a = mentionPair.getFirst();
            Mention b = mentionPair.getSecond();
            if (a.gloss().equals("Firestone")) {
                System.out.println("(FIRESTONE)" + a.gloss() + " ***** " + b.gloss());
            }
            if (isCoreferent(clusters, a, b)) {
                continue;
            }
            System.out.println("NOT COREFERENT");
            if (false || // TODO apposition
                isPredicateNominative(a, b) || // TODO predicate nominative
                false || // TODO role appositive
                false || // TODO relative pronoun
                false || // TODO acronym
                false) { // TODO demonym
                System.out.println(a.gloss() + " ***** " + b.gloss() + " ##### " + isCoreferent(clusters, a, b));
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
}
