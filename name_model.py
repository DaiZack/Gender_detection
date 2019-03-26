from nltk.corpus import names
import random,nltk,pickle
import numpy as np

male_name = names.words('male.txt')
female_name = names.words('female.txt')


def name_count(name):
    arr = np.zeros(52+26*26)
    # Iterate each character
    for ind, x in enumerate(name):
        arr[ord(x)-ord('a')] += 1
        arr[ord(x)-ord('a')+26] += ind+1
    # Iterate every 2 characters
    for x in range(len(name)-1):
        if name[x].isalpha():
            ind = (ord(name[x])-ord('a'))*26 + (ord(name[x+1])-ord('a'))
            arr[ind] += 1
    return arr


def gender_features(word):
    word = word.lower()
    feature= {'last_l': word[-1], 'last_2': word[-2]}

    arr = name_count(word)

    for i in range(len(arr)):
        feature[i] = arr[i]
    return feature


labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])

random.shuffle(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

train_set, test_set = featuresets[:3000], featuresets[7000:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

# print(classifier.classify(gender_features('ab')))
# print(nltk.classify.accuracy(classifier, featuresets))

with open('gender_detect.pickle', 'wb') as file:
    pickle.dump(classifier, file)