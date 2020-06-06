import spacy
import pandas as pd
import itertools as it
import networkx as nx
import dynet as dy
from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
import re
from collections import defaultdict
from random import sample 
import numpy as np
import itertools
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm')



def read_data(filex):

    return pd.read_csv(filex, sep=r'\t', names=["Sent","Ent1","Rel","Ent2","Text"], header=None)



word2vecIndex = defaultdict()
word2vec = None

def proccesing_line(line):
    
    # line = line[1:-1].strip()
    processed_line = re.sub(r'\s+', ' ', line)
    processed_line = re.sub(r"\s+\`\s+\`", '', processed_line)
    processed_line = re.sub(r"\s+\'\s+\'", '', processed_line)
    processed_line = re.sub(r'\.$', '', processed_line)
    processed_line = re.sub(r'\'', '', processed_line)
    # processed_line = re.sub(r'\.\s+', ' ', processed_line)
    # processed_line = re.sub(r'\'s', '', processed_line)
    # processed_line = re.sub('[^a-zA-Z0-9,.\']', ' ', line)
    processed_line = re.sub(r"\(",r"( ", processed_line) 
    processed_line = re.sub(r"\)", r") ", processed_line)
    processed_line = re.sub(r'LRB', '(', processed_line)
    processed_line = re.sub(r'RRB', ')', processed_line)
    processed_line = processed_line.strip()
    return processed_line



def augmentation(annot):

    df = pd.DataFrame(columns=("Sent","Ent1","Rel","Ent2","Text"))
    for index, row in annot.iterrows():
        ent1Label = proccesing_line(row["Ent1"])
        relLabe = row["Rel"]
        ent2Label = proccesing_line(row["Ent2"])
        doc = nlp(row["Text"])

        ent1 = label_ent(ent1Label, doc)
        ent2 = label_ent(ent2Label, doc)

        if ent1 is None or ent2 is None:
            continue

        if ent1.end < ent2.start:
            span = doc[ent1.end:ent2.start]
        else:
            span = doc[ent2.end:ent1.start]
        sen = span.text.split(" ")
        chosen_word = sample(sen, 1)[0]
        i = sen.index(chosen_word)
        synosym = syns_word(chosen_word)
        if synosym is not None:
            tmp_doc = doc.text.split(" ")
            j = tmp_doc.index(chosen_word)
            tmp_doc[j] = synosym
            doc = " ".join(tmp_doc)
            df = df.append({'Ent1' : ent1Label , 'Rel' : relLabe, "Ent2" : ent2Label, "Text" : doc} , ignore_index=True)

    return df

def syns_word(chosen_word):

    synonyms = []

    for syn in wn.synsets(chosen_word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    while(len(synonyms) > 0):

        chosen_synosym = sample(synonyms, 1)[0]
        i = synonyms.index(chosen_synosym)

        if chosen_synosym != chosen_word:
            return chosen_synosym
        else:
            del synonyms[i]
    return None
            


def proccesing_data(annot, type_ = "train"):
    
    global word2vecIndex
    global word2vec
    sentences = []
    df = []
    
    for _, line in annot.iterrows():
        pline = proccesing_line(line["Text"][1:-1].strip() if type_ == "train" else line["Text"][0:-1].strip())
        sentences.append(pline)
        line = [line["Sent"], line["Ent1"], line["Rel"], line["Ent2"], pline] if type_ == "train" else [line["Sent"], pline]
        df.append(line)
    annot = pd.DataFrame(df, columns = ["Sent","Ent1","Rel","Ent2","Text"]) if type_ == "train" else pd.DataFrame(df, columns = ["Sent","Text"])


    if type_ == "train":

        df = read_wikipedia()

        for _, line in df.iterrows():
            sentences.append(line["Text"])

        annot = pd.concat([annot, df])

        all_words = [sent.split(' ') for sent in sentences]
        word2vec = Word2Vec(all_words, size=300, window=5, min_count=1, workers=4)

        for i in range(len(word2vec.wv.vectors)):
            word2vecIndex[word2vec.wv.index2word[i]] = i
        word2vecIndex["UNK"] = i + 1

    return annot

def read_wikipedia():
    
    df = pd.DataFrame(columns=("Sent","Ent1","Rel","Ent2","Text"))
    

    with open(r"data/wikipedia.train",'r', encoding="utf-8") as rf:  
        
        for line in rf.readlines():
            
            if line.startswith("url"):
                ent1 = proccesing_line(line.strip().split("/")[-1])
                continue
            if line == "\n":
                continue
            parsed_line = BeautifulSoup(line)
            relations = []
            ents = []
            for script in parsed_line.findAll("a"):
                if script.has_attr('relation'):
                    relations.append(script['relation'])
                    if script.has_attr('title'):
                        ents.append(script['title'])
            body = parsed_line.body
            for tag in body.select('script'):
                tag.decompose()
            for tag in body.select('style'):
                tag.decompose()
            parsed_line = body.get_text(separator='')
            parsed_line = re.sub(","," ,",parsed_line)
            parsed_line = re.sub("\."," .",parsed_line)
            parsed_line = re.sub(r'\"', '\" ', parsed_line)
            parsed_line = proccesing_line(parsed_line)
            if len(relations) == 0:
                continue

            entities = list(zip(relations, ents))
            for pair in entities:
                df = df.append({"Ent1" : ent1, "Rel" : pair[0], "Ent2" : pair[1], "Text" : parsed_line}, ignore_index=True)

            # entities = []
            # entities = list(itertools.product(nlp(parsed_line).ents, nlp(parsed_line).ents))
            # pref_entities = list(filter(lambda x: x[0].label_=="PERSON" and x[1].label_=="GPE", entities))
            # if len(pref_entities) == 0:
            #     entities = list(filter(lambda x: x[0].text != x[1].text, entities))
            #     pref_entities = entities

            
            # word_tokens = parsed_line.spilt(" ")
            # parsed_line = " ".join([w for w in word_tokens if not w in stop_words])
            # for pair in pref_entities:
            #     df = df.append({"Ent1" : pair[0].text, "Rel" : label, "Ent2" : pair[1].text, "Text" : parsed_line}, ignore_index=True)

        return df

def label_ent(ent, doc):
    
    # for e in list(doc.noun_chunks):
    #     if ent in e.text or e.text in ent:
    #         return e
    for e in doc.ents:
        if ent in e.text or e.text in ent:
            return e

# def label_ent(ent, doc):

#     doc = re.split("/| |-",doc.text)
#     ent = re.split("/| |-",ent)
#     for w in doc:
#         if ent[-1] in w or w in ent[-1]:
#             return doc.index(ent[-1])

def span_entities(doc, ent1, ent2):
    
    if ent1.start < ent2.start:
        return doc[ent1.start:ent2.end]
    return doc[ent2.start:ent1.end]


class LSTM2:

    def __init__(self, model):

        global word2vec
        self.LSTM_NUM_OF_LAYERS = 1
        self.VOCAB_SIZE = len(word2vec.wv.vocab)
        self.EMBEDDINGS_SIZE = 300
        self.STATE_SIZE = 32
        self.enc_fwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, model)
        self.enc_bwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, model)
        self.accecptor_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.STATE_SIZE*2, self.STATE_SIZE, model)
        self.w = model.add_parameters((200, self.STATE_SIZE))
        self.b1 = model.add_parameters(200)
        self.u = model.add_parameters((2,200))
        self.b2 = model.add_parameters(2)
        self.input_lookup = model.add_lookup_parameters((self.VOCAB_SIZE + 1, self.EMBEDDINGS_SIZE))
        self.input_lookup.init_from_array(np.concatenate((word2vec.wv.vectors, np.zeros(self.EMBEDDINGS_SIZE, dtype=float).reshape(1,-1)),axis=0))
        
    
    def encoded_sentence(self, embedded):

        embedded_rev = list(reversed(embedded))
        fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), embedded)
        bwd_vectors = self.run_lstm(self.enc_bwd_lstm.initial_state(), embedded_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def embed_sentence(self, sentence):
        global word2vecIndex
        vector = []
        for w in sentence.split(" "):
            try:
                w = self.input_lookup[word2vecIndex[w]]
            except:
                w = self.input_lookup[word2vecIndex["UNK"]]
            vector.append(w)
        return vector

    def run_lstm(self, init_state, input_vecs):

        s = init_state
        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors

    def get_loss(self, input_sentence, label):

        dy.renew_cg()

        w = dy.parameter(self.w)

        b1 = dy.parameter(self.b1)

        u = dy.parameter(self.u)

        b2 = dy.parameter(self.b2)

        embedded = self.embed_sentence(input_sentence)

        encoded = self.encoded_sentence(embedded)

        acc_lstm = self.run_lstm(self.accecptor_lstm.initial_state(), encoded)

        mlp_input = acc_lstm[-1]

        h = dy.tanh((w * mlp_input) + b1)

        y_pred = dy.softmax((u * h) + b2)

        loss = -dy.log(dy.pick(y_pred, label))

        return loss
    

    def predict(self, input_sentence):

        dy.renew_cg()

        w = dy.parameter(self.w)

        b1 = dy.parameter(self.b1)

        u = dy.parameter(self.u)

        b2 = dy.parameter(self.b2)

        embedded = self.embed_sentence(input_sentence)

        encoded = self.encoded_sentence(embedded)

        encoded = self.encoded_sentence(embedded)

        acc_lstm = self.run_lstm(self.accecptor_lstm.initial_state(), encoded)

        mlp_input = acc_lstm[-1]

        h = dy.tanh((w * mlp_input) + b1)

        y_pred = dy.softmax((u * h) + b2).vec_value()

        return y_pred.index(max(y_pred))


class LSTM:

    def __init__(self, word2vec, model):
        self.LSTM_NUM_OF_LAYERS = 2
        self.VOCAB_SIZE = len(word2vec.wv.vocab)
        self.EMBEDDINGS_SIZE = 300
        self.ATTENTION_SIZE = 32
        self.STATE_SIZE = 32
        self.model = model
        self.attention_w1 = model.add_parameters( (self.ATTENTION_SIZE, self.STATE_SIZE*2))
        self.attention_w2 = model.add_parameters( (self.ATTENTION_SIZE, self.STATE_SIZE*self.LSTM_NUM_OF_LAYERS*2))
        self.attention_v = model.add_parameters( (1, self.ATTENTION_SIZE))
        self.enc_fwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, model)
        self.enc_bwd_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.EMBEDDINGS_SIZE, self.STATE_SIZE, model)
        self.input_lookup = model.add_lookup_parameters((self.VOCAB_SIZE + 1, self.EMBEDDINGS_SIZE))
        self.input_lookup.init_from_array(np.concatenate((word2vec.wv.vectors, np.zeros(self.EMBEDDINGS_SIZE, dtype=float).reshape(1,-1)),axis=0))
        self.decoder_w = model.add_parameters( (self.VOCAB_SIZE, self.STATE_SIZE))
        self.decoder_b = model.add_parameters( (self.VOCAB_SIZE))
        self.dec_lstm = dy.LSTMBuilder(self.LSTM_NUM_OF_LAYERS, self.STATE_SIZE*2+self.EMBEDDINGS_SIZE, self.STATE_SIZE, model)
        self.mlp_w = model.add_parameters((200, self.STATE_SIZE))
        self.mlp_b1 = model.add_parameters(200)
        self.mlp_u = model.add_parameters((2,200))
        self.mlp_b2 = model.add_parameters(2)

    
    
    def encoded_sentence(self, embedded):

        embedded_rev = list(reversed(embedded))
        fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), embedded)
        bwd_vectors = self.run_lstm(self.enc_bwd_lstm.initial_state(), embedded_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def embed_sentence(self, sentence):
        global word2vecIndex
        vector = []
        for w in sentence.split(" "):
            try:
                w = self.input_lookup[word2vecIndex[w]]
            except:
                w = self.input_lookup[word2vecIndex["UNK"]]
            vector.append(w)
        return vector


    def run_lstm(self, init_state, input_vecs):

        s = init_state
        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors

    
    def attend(self, input_mat, state, w1dt):

        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)
        w2dt = w2*dy.concatenate(list(state.s()))
        scores = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(scores)
        context = input_mat * att_weights
        return context

    def decode(self, vectors, sentence, label):

        # w = dy.parameter(self.decoder_w)
        # b = dy.parameter(self.decoder_b)
        global word2vecIndex
        w1 = dy.parameter(self.attention_w1)
        w_mlp = dy.parameter(self.mlp_w)
        u_mlp = dy.parameter(self.mlp_u)
        b2_mlp = dy.parameter(self.mlp_b2)
        b1_mlp = dy.parameter(self.mlp_b1)


        input_mat = dy.concatenate_cols(vectors)
        w1dt = None
        
        try:
            last_output_embeddings = self.input_lookup[sentence.split(" ")[0]]
        except:
            last_output_embeddings = self.input_lookup[word2vecIndex["UNK"]]
        
        # last_output_embeddings = self.input_lookup[word2vecIndex[first_word]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.STATE_SIZE*2), last_output_embeddings]))
        for word in sentence.split(" "):
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            try:
                word = self.input_lookup[word2vecIndex[word]]
            except:
                word = self.input_lookup[word2vecIndex["UNK"]]
            last_output_embeddings = self.input_lookup[word2vecIndex[word]]
        mlp_input = s.output()
        h_mlp = dy.tanh((w_mlp * mlp_input) + b1_mlp)
        # y_pred = dy.logistic((u_mlp*h_mlp) + b2_mlp)
        # loss = dy.binary_log_loss(y_pred, dy.scalarInput(label))
        y_pred = dy.softmax((u_mlp * h_mlp) + b2_mlp)
        loss = -dy.log(dy.pick(y_pred, label))

        return loss

    
    def predict(self, input_sentence):

        dy.renew_cg()

        embedded = self.embed_sentence(input_sentence)

        encoded = self.encoded_sentence(embedded)

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)
        global word2vecIndex
        w1 = dy.parameter(self.attention_w1)
        w_mlp = dy.parameter(self.mlp_w)
        u_mlp = dy.parameter(self.mlp_u)
        b2_mlp = dy.parameter(self.mlp_b2)
        b1_mlp = dy.parameter(self.mlp_b1)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None
        try:
            last_output_embeddings = self.input_lookup[sentence.split(" ")[0]]
        except:
            last_output_embeddings = self.input_lookup[word2vecIndex["UNK"]]
        # last_output_embeddings = self.input_lookup[word2vecIndex[first_word]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.STATE_SIZE*2), last_output_embeddings]))

        for word in input_sentence.split(" "):
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = dy.tanh(w * s.output() + b)
            probs = dy.softmax(out_vector).vec_value()
            next_word = probs.index(max(probs))
            word = self.input_lookup[next_word]
            last_output_embeddings = self.input_lookup[word2vecIndex[word]]
        mlp_input = s.output()
        h_mlp = dy.tanh((w_mlp * mlp_input) + b1_mlp)
        y_pred = dy.softmax((u_mlp * h_mlp) + b2_mlp).vec_value()
        return y_pred.index(max(y_pred))



    def get_loss(self, input_sentence, label):

        dy.renew_cg()

        embedded = self.embed_sentence(input_sentence)

        encoded = self.encoded_sentence(embedded)

        return self.decode(encoded, input_sentence, label)



def evaluation(lstm):
    
    annot = pd.read_csv(r"data/Corpus.DEV.txt", sep=r"\t", header = None, names = ["Sent", "Text"])

    annot = proccesing_data(annot, "test")

    with open("predicts.txt", 'w') as wf:

        for _, row in annot.iterrows():

            doc = nlp(row["Text"])

            targets = []

            non_tragets = []

            # test_compare_list = defaultdict(dict)

            entities = list(itertools.product(doc.ents, doc.ents))

            entities = list(filter(lambda x: x[0].text != x[1].text, entities))

            
            for pair in entities:

                span = span_entities(doc,pair[0],pair[1])

                rel = lstm.predict(span.text)

                if rel == 1:
                    targets.append(str(pair))
                else:
                    non_tragets.append(str(pair))

            # test_compare_list[index]["targets"] = targets
            # test_compare_list[index]["non_targets"] = non_tragets
        
            wf.write("{}{}{}.\n)".format(",".join(targets)+r'\t' if len(targets) !=0 else "" , ",".join(non_tragets)+r'\t', doc.text))
            



################################################################

def evaluation2(lstm):

    annot = read_data(r"data/DEV.annotations")

    annot = proccesing_data(annot, "test")

    examples = 0
    nones = []
    test_list = []
    golden_list = []
    for _,row in annot.iterrows():
        golden_list.append(1 if row["Rel"] == "Live_In" else 0)
    
    good = 0
    bad = 0

    with open("predicts2.txt", 'w') as wf:

        for index, row in annot.iterrows():

            ent1Label = proccesing_line(row["Ent1"])
            ent2Label = proccesing_line(row["Ent2"])
            doc = nlp(row["Text"])

            ent1 = label_ent(ent1Label, doc)
            ent2 = label_ent(ent2Label, doc)


            if ent1 is None or ent2 is None:
                nones.append({"Index" : index, "Golden" : golden_list[index]})
                bad += 1
                continue

            examples += 1
            
            span = span_entities(doc,ent1,ent2)

            rel = lstm.predict(span.text)

            persons = []
            gpe = []
            
            if rel == 1:
                for e in doc.ents:
                    if e.label_ == "PERSON":
                        persons.append(e)
                    elif e.label_ == "GPE":
                        gpe.append(e)
                pair = sample(list(zip(persons, gpe)), 1)
            else:
                entities = list(itertools.product(doc.ents, doc.ents))
                entities = list(filter(lambda x: x[0].text != x[1].text, entities))
                pair = sample(entities, 1)

            wf.write("{} {} {} ( {}.)".format(str(pair[0]), "Live_In" if rel==1 else "Live_out", str(pair[1]), doc.text))
            if rel == golden_list[index]:

                good += 1
                test_list.append(1)

            else:

                bad += 1
                test_list.append(0)
        

    tp = 0

    fp = 0

    fn = 0

    for i,j in zip(golden_list, test_list):

        if j == i == 1:
            tp += 1
        if j == 1 and i == 0:
            fp += 1
        if j == 0 and i == 1:
            fn += 1


    print("Precision: {}".format(tp / (tp + fp)))

    print("Recall: {}".format(tp / (tp + fn)))


if __name__ == "__main__":

    annot = read_data(r"data/old_TRAIN.annotations")

    annot = proccesing_data(annot)
    
    model = dy.Model()

    trainer = dy.SimpleSGDTrainer(model)

    lstm = LSTM2(model)

    examples = 0

    total_loss = 0.0
    nones = 0

    
    for index, row in annot.iterrows():

        ent1Label = proccesing_line(row["Ent1"])
        relLabe = 1 if row["Rel"] == "Live_In" or row["Rel"] == "birth_place" else 0
        ent2Label = proccesing_line(row["Ent2"])
        doc = nlp(row["Text"])

        ent1 = label_ent(ent1Label, doc)
        ent2 = label_ent(ent2Label, doc)

        if ent1 is None or ent2 is None:
            nones += 1
            continue

        
        examples += 1

        span = span_entities(doc,ent1,ent2)

        loss = lstm.get_loss(span.text, relLabe)

        # loss = lstm.get_loss(span.text, relLabe)

        total_loss += loss .value()

        loss.backward()

        trainer.update()


        if examples % 100 == 0:
            print("loss: {}".format(total_loss / examples))

    print("Nones: {}".format(nones))
    print("Total examples: {}".format(examples))

    
    evaluation(lstm)

    


