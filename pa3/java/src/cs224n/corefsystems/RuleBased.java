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
        // A hash of set: entity => [coreferent mentions of this entity]
        // Initialized as m.entity => [m], each mention has its own entity.
//        Map<Entity, Set<Mention>> entityMentionMap = new HashMap<Entity, Set<Mention>>();
//        for (Mention m : doc.getMentions()) {
//            Entity entity = clusters.get(m).entity;
//            Set<Mention> mentionSet = new HashSet<Mention>();
//            mentionSet.add(m);
//            entityMentionMap.put(entity, mentionSet);
//        }
        // Create all combination pair of mentions. Easy to loop through.
        Set<Mention> seenMentions = new HashSet<Mention>();
        List<Pair<Mention, Mention>> mentionPairs = new ArrayList<Pair<Mention, Mention>>();
        for(Mention mi : doc.getMentions()) {
            seenMentions.add(mi);
            for(Mention mj : doc.getMentions()) {
                if (!seenMentions.contains(mj)) {
                    mentionPairs.add(new Pair<Mention, Mention>(mi, mj));
                }
            }
        }
        // Multi-pass sieve (NLP 10')
        pass1(mentionPairs, clusters);

        // Create and return the mentions
        List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();
        for (ClusteredMention cluster : clusters.values()) {
            mentions.add(cluster);
        }
        return mentions;
    }
    
    // Move src to the same bin as dst's entity, meaning that they're coreferent.   
    private void updateCoreferent(Map <Mention, ClusteredMention> clusters, Mention dst, Mention src) {
        src.removeCoreference();
        clusters.put(src, src.markCoreferent(clusters.get(dst)));
    }

    private void pass1(List<Pair<Mention, Mention>> mentionPairs,
                       Map <Mention, ClusteredMention> clusters) {
        for (Pair<Mention, Mention> mentionPair : mentionPairs) {
            Mention mi = mentionPair.getFirst();
            Mention mj = mentionPair.getSecond();
            if (mi.gloss().equals(mj.gloss())) {
                updateCoreferent(clusters, mi, mj);
            }
        }
    }
}
