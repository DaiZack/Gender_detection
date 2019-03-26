import pandas as pd
import pickle
import numpy as np

data = pd.read_csv('person_untagged.csv')
names = data['person'].astype('str')
with open('gender_detect.pickle', 'rb') as file:
    detector = pickle.load(file)


def parse_names(names):
    name_small = [n.lower() for n in names]

    lastnames = []
    firstnames = []
    middlenames = []

    for n in name_small:
        n = n.strip()
        if len(n.split(',')) == 1:
            lastnames.append(n.split(' ')[-1])
            firstnames.append(n.split(' ')[0])
            if len(n.split(' '))>2:
                middlenames.append(' '.join(n.split(' ')[1:-1]))
            else:
                middlenames.append('')
        else:
            lastnames.append(n.split(',')[0])
            firstnames.append(n.split(',')[-1].strip().split(' ')[0])
            if len(n.split(',')[-1].split(' '))>2:
                middlenames.append(' '.join(n.split(',')[-1].strip().split(' ')[1:]))
            else:
                middlenames.append('')
    return lastnames, firstnames, middlenames


def name_count(name):
    name = name.lower()
    arr = np.zeros(52+26*26)
    # Iterate each character
    for ind, x in enumerate(name):
        if ord('a') <= ord(x) <= ord('z'):
            arr[ord(x)-ord('a')] += 1
            arr[ord(x)-ord('a')+26] += ind+1
    # Iterate every 2 characters
    for x in range(len(name)-1):
        if ord('a')<=ord(name[x])<=ord('z') and ord('a')<=ord(name[x+1])<=ord('z'):
            ind = (ord(name[x])-ord('a'))*26 + (ord(name[x+1])-ord('a'))
            arr[ind] += 1
    return arr


def gender_features(word):
    word = word.lower()
    if len(word)>1:
        feature= {'last_l': word[-1], 'last_2': word[-2]}
    else:
        feature = {'last_l': word[-1], 'last_2': word[-1]}

    arr = name_count(word)

    for i in range(len(arr)):
        feature[i] = arr[i]
    return feature


lastnames, firstnames, middlenames = parse_names(names)

data['lastname'] = lastnames
data['firstname'] = firstnames
data['middlename'] = middlenames

# data['predict_gender'] = [detector.classify(gender_features(n)) for n in data['firstname'] if n]

genders = []
for n in data['firstname']:
    print(n,' : ',detector.classify(gender_features(n)))
    genders.append(detector.classify(gender_features(n)))

data['predict_gender'] = genders

