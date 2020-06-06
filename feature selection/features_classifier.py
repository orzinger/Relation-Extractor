import pickle
import random
import re

import networkx as nx
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

nlp = spacy.load('en_core_web_sm')


##########################################################################
########################## Entities extraction ###########################
##########################################################################


# Extract entity from given entity name
def label_ent(ent, doc):
    for e in doc.ents:
        if ent in e.text or e.text in ent:
            return e


# Return span between given entities
# --- Ent1 w1 w2 ... wn Ent2 ---
# span = w1 w2 ... wn
def span_entities(doc, ent1, ent2):
    if ent1.start < ent2.start:
        return doc[ent1.start + 1:ent2.start]
    return doc[ent2.start + 1:ent1.start]


#########################################################################
########################## Syntactic features ###########################
#########################################################################

# Return graph of text
def build_deps_graph(doc):
    edges = []
    for token in doc:
        for child in token.children:
            edges.append((token, child))
    return nx.Graph(edges)


# Return shortest path in tree between two entities
def find_shortestpath_deps(graph, entity1, entity2):
    return nx.shortest_path(graph, source=entity1.root, target=entity2.root)


# Return shortest path and length in dependicie tree between two entities
def dependicies_tree_path(entities, graph):
    return find_shortestpath_deps(graph, entities[0], entities[1])


# Return base syntactic chunk
def base_syntactic_chunks(entities, graph):
    return list(nx.all_simple_paths(graph, entities[0].root, entities[1].root))


# Return maximun spanning tree
def mst_(graph):
    return sorted(nx.maximum_spanning_tree(graph).edges(data=True))


# Build syntactic features
def syntactic_builder(entities, doc):
    syntatic_dict = {}
    graph = build_deps_graph(doc)
    try:
        dtp = dependicies_tree_path(entities, graph)
        for i, d in enumerate(dtp):
            syntatic_dict['dep' + str(i)] = d.dep_
    except Exception:
        pass

    return syntatic_dict


##########################################################################
########################## Between entities features #####################
##########################################################################

# Build the frame (feature vector) of the target words
# --- frame: [bow,pos heads,span ents, number of span ents] ---
def frame_builder(entities, doc):
    frame_dict = {}
    ent1, ent2 = entities
    span = span_entities(doc, ent1, ent2)

    pos_span_heads = pos_heads(span)
    for i, pos_head in enumerate(pos_span_heads):
        frame_dict['pos_head' + str(i)] = pos_head

    span_ents, len_span_ents = back_entities(span)
    frame_dict['span_len'] = len_span_ents

    return frame_dict


# Return POS of heads of words between target entities span
def pos_heads(span):
    return [s.head.tag_ for s in span]


# Return other entities between the target entities
def back_entities(span):
    return [e.label_ for e in span.ents], len(span.ents)


# Return synonyms of a sample word in span
# --- i.e: 'rejected' = synonyms['reject', 'refuse', 'reject', 'pass_up', 'turn_down', ...] ---
def synset_wordnet(span):
    tokens = list(span)
    word = random.sample(tokens, 1)
    tokens.remove(word[0])
    z = [s.lemma_names() for s in wn.synsets(str(word[0]))]
    return list(filter(lambda x: x != 'by', [j for i in z for j in i])), tokens


##########################################################################
########################## Entities features #############################
##########################################################################

# Return internal entities feaures
def entitiy_internal_builder(entities, doc):
    internal_dict = {}
    ent1, ent2 = entities

    # Get entity type
    entities_types = [ent1.label_, ent2.label_]
    for i, entity_type in enumerate(entities_types):
        internal_dict['ent_type' + str(i)] = entity_type

    # Get entity head POS
    entities_heads_pos = [ent1.root.pos_, ent2.root.pos_]
    for i, entity_head_pos in enumerate(entities_heads_pos):
        internal_dict['entity_head_pos' + str(i)] = entity_head_pos

    # Get entity head dependecy
    entities_heads_deps = [ent1.root.dep_, ent2.root.dep_]
    for i, entity_head_dep in enumerate(entities_heads_deps):
        internal_dict['entity_head_dep' + str(i)] = entity_head_dep

    # Get iob of entity
    entities_iob = [ent1.root.ent_iob_ + ent2.root.ent_iob_]

    internal_dict['entities_iob'] = entities_iob[0]

    return internal_dict


def processing_line(line):
    processed_line = re.sub(r'\s+', ' ', line)
    processed_line = re.sub(r'\.$', '', processed_line)
    processed_line = re.sub(r'\-|_', '', processed_line)
    processed_line = re.sub(r'\s+,\s+', ',', processed_line)
    processed_line = re.sub(r'LRB', '(', processed_line)
    processed_line = re.sub(r'RRB', ')', processed_line)
    processed_line = processed_line.strip()
    return processed_line


def processing_data(annot):
    df = []
    for _, line in annot.iterrows():
        pline = processing_line(line["Text"][1:-2].strip())
        line = [line["Sent"], line["Ent1"], line["Rel"], line["Ent2"], pline]
        df.append(line)
    return pd.DataFrame(df, columns=["Sent", "Ent1", "Rel", "Ent2", "Text"])


# Function building features of piar of entities
# --- [Features of the named entities/arguments involved] [Features derived from the words between and
# ---  round the named entities] [Features derived from the syntactic environment that governs the two entities]
def features_builder(doc, ent1, ent2):
    entities = [ent1, ent2]
    # Syntactic features
    syntatic_dict = syntactic_builder(entities, doc)

    # Between entities features
    frame_dict = frame_builder(entities, doc)

    # internal features
    internal_dict = entitiy_internal_builder(entities, doc)

    features_dict = {}
    features_dict.update(syntatic_dict)
    features_dict.update(frame_dict)
    features_dict.update(internal_dict)

    return features_dict


def save_model(model, model_file):
    pickle.dump(model, open(model_file, 'wb'))


def convert_features_map_to_vectorized_form(features_map):
    vec = DictVectorizer()
    pos_vectorized = vec.fit_transform(features_map)

    save_model(vec, 'vec.sav')

    return pos_vectorized, vec


def train_model(annot):
    features_dicts = []
    labels = []
    for index, row in annot.iterrows():
        ent1Label = row["Ent1"]
        relLabel = row["Rel"]
        ent2Label = row["Ent2"]

        doc = nlp(processing_line(row["Text"][1:-2].strip()))

        ent1 = label_ent(processing_line(ent1Label), doc)
        ent2 = label_ent(processing_line(ent2Label), doc)

        if ent1 and ent2:
            features_dict = features_builder(doc, ent1, ent2)
            features_dicts.append(features_dict)

            if relLabel == 'Live_In':
                labels.append(True)
            else:
                labels.append(False)

    f, vec = convert_features_map_to_vectorized_form(features_dicts)
    logreg = LogisticRegression(solver='lbfgs',
                                multi_class='multinomial',
                                tol=0.01,
                                random_state=0,
                                max_iter=40000)
    logreg.fit(f, labels)
    print(logreg.score(f, labels))

    save_model(logreg, 'logreg.sav')

    return logreg, vec


def test_model(dev_annot, logreg, vec):
    dev_features_dicts = []
    dev_labels = []

    for index, row in dev_annot.iterrows():
        ent1Label = row["Ent1"]
        relLabel = row["Rel"]
        ent2Label = row["Ent2"]

        doc = nlp(processing_line(row["Text"][1:-2].strip()))

        ent1 = label_ent(processing_line(ent1Label), doc)
        ent2 = label_ent(processing_line(ent2Label), doc)

        if ent1 and ent2:
            features_dict = features_builder(doc, ent1, ent2)
            activated_features = vec.transform(features_dict).indices
            feature_vector = np.zeros(len(vec.get_feature_names()))

            for index in activated_features:
                feature_vector[index] = 1

            dev_features_dicts.append(feature_vector)

            if relLabel == 'Live_In':
                dev_labels.append(True)
            else:
                dev_labels.append(False)

    true_pos = 0
    false_pos = 0
    false_neg = 0
    for f_v, relation in zip(dev_features_dicts, dev_labels):
        prediction = logreg.predict(f_v.reshape(1, -1))[0]
        if prediction == True and relation == True:
            true_pos += 1

        if prediction == True and relation == False:
            false_pos += 1

        if prediction == False and relation == True:
            false_neg += 1

    # accuracy
    print(logreg.score(dev_features_dicts, dev_labels))
    # precession
    precession = true_pos / (true_pos + false_pos)
    print(precession)
    # recall
    recall = true_pos / (true_pos + false_neg)
    print(recall)
    # f1
    print(2 * ((precession * recall) / (precession + recall)))


if __name__ == '__main__':
    annot = pd.read_csv("data/TRAIN.annotations", sep='\t', names=["Sent", "Ent1", "Rel", "Ent2", "Text"], header=None)
    dev_annot = pd.read_csv("data/DEV.annotations", sep='\t', names=["Sent", "Ent1", "Rel", "Ent2", "Text"], header=None)

    logreg, vec = train_model(annot)

    test_model(dev_annot, logreg, vec)
