# Relation-Extractor


A NLP tool for relations extraction from a training set, including
over 1000 of sentences.
From each sentence, we extract one kind of relation: "Living In",
mean, that there is some realtion between a person and a living place.

The tool consist two methods to deal with this problem:

1) Deep Learning style - an attentions network was builded with 1 layer of LSTM and 
a regular feed-forrowed neural network to predict the relation (existing or non-existent)
One of the problems that showed up is how do we train a deep neural network with only
1000 training examples. So, we used an augumented data

2) Feature Selection style - a simple SVM using a vector of features to predict the relation (existing or non-existent).

Including report

