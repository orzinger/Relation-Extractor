import pickle
import sys

import numpy as np
import pandas as pd
import spacy

from features_classifier import processing_line, features_builder, label_ent

nlp = spacy.load('en_core_web_sm')


def extract(input_file_name, output_file_name):
    if input_file_name.split('.')[-1] != 'txt':
        print('please insert .txt file to run the program as intended')
        exit(-1)

    vec = pickle.load(open('vec.sav', 'rb'))
    logreg = pickle.load(open('logreg.sav', 'rb'))
    input = pd.read_csv(input_file_name, sep='\t', names=["Sent_NO", "Sent"], header=None)

    with open(output_file_name, 'w') as output:
        for index, row in input.iterrows():
            sent_no = row['Sent_NO']
            doc = nlp(processing_line(row["Sent"][1:-2].strip()))

            ents = [(e.text, e.label_, e.kb_id_) for e in doc.ents]
            persons = [ent[0] for ent in ents if ent[1] == 'PERSON']
            gpes = [ent[0] for ent in ents if ent[1] == 'GPE']

            if persons and gpes:
                trues = []
                for person in persons:
                    for gpe in gpes:
                        ent1 = label_ent(processing_line(person), doc)
                        ent2 = label_ent(processing_line(gpe), doc)

                        if ent1 and ent2:
                            features_dict = features_builder(doc, ent1, ent2)
                            activated_features = vec.transform(features_dict).indices
                            feature_vector = np.zeros(len(vec.get_feature_names()))

                        for index in activated_features:
                            feature_vector[index] = 1

                        if logreg.predict(feature_vector.reshape(1, -1))[0] == True:
                            trues.append((person, gpe))

                if trues:
                    output.write(
                        sent_no + '\t' + trues[0][0] + '\t' + 'Live_In' + '\t' + trues[0][1] + '\t' + '( ' + row[
                            "Sent"] + ' )' + '\n')
                else:
                    output.write(
                        sent_no + '\t' + persons[0] + '\t' + 'NOT_Live_In' + '\t' + gpes[0] + '\t' + '( ' + row[
                            "Sent"] + ' )' + '\n')
            else:
                output.write(
                    sent_no + '\t' + 'no person and gpe detected' + '\t' + 'NOT Live_In' + '\t' + 'no person and gpe detected' + '\t' + '( ' +
                    row["Sent"] + ' )' + '\n')


if __name__ == '__main__':
    extract(sys.argv[1], sys.argv[2])
