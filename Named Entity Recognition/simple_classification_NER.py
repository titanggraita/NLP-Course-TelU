# Ade Romadhony
# Fakultas Informatika, Telkom University

from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# baca file train dan test
train_lines = []
test_lines = []
with open('kalimat_POS_NE_train.txt', 'r') as f:
    train_lines = f.readlines()
with open('kalimat_POS_NE_test.txt', 'r') as f:
    test_lines = f.readlines()

# fungsi untuk konversi label NE
def convert_label(raw_ne):
    new_label = 0
    if (raw_ne=='B-PER'): new_label = 1
    if (raw_ne=='I-PER'): new_label = 2
    if (raw_ne=='B-ORG'): new_label = 3
    if (raw_ne=='I-ORG'): new_label = 4
    if (raw_ne=='B-LOC'): new_label = 5
    if (raw_ne=='I-LOC'): new_label = 6
    return new_label

# inisialisasi kode token dan postag
token_dict = {}
prefix1_dict = {}
prefix2_dict = {}
prefix3_dict = {}
suffix1_dict = {}
suffix2_dict = {}
suffix3_dict = {}
postag_dict = {}
counter_token = 0
counter_postag = 0
counter_prefix1 = 0
counter_prefix2 = 0
counter_prefix3 = 0
counter_suffix1 = 0
counter_suffix2 = 0
counter_suffix3 = 0

# baca data token, postag, dan label NE
train_sents = []
test_sents = []
sent = []
counter = 0
# data train
for line in train_lines:
    line = line.rstrip('\n')
    curr_tuple = ()
    if len(line)>1:
        line_part = line.split(" ")
        t = (line_part[0], line_part[1], convert_label(line_part[2]))
        if line_part[0].lower() not in token_dict.keys():
            token_dict[line_part[0].lower()] = counter_token
            counter_token = counter_token+1
            if len(line_part[0].lower())>=3:
                #prefix
                if line_part[0].lower()[0] not in prefix1_dict.keys():
                    prefix1_dict[line_part[0].lower()[0]] = counter_prefix1
                    counter_prefix1 = counter_prefix1+1
                if line_part[0].lower()[0:2] not in prefix2_dict.keys():
                    prefix2_dict[line_part[0].lower()[0:2]] = counter_prefix2
                    counter_prefix2 = counter_prefix2+1
                if line_part[0].lower()[0:3] not in prefix3_dict.keys():
                    prefix3_dict[line_part[0].lower()[0:3]] = counter_prefix3
                    counter_prefix3 = counter_prefix3+1
                #suffix
                if line_part[0].lower()[-1] not in suffix1_dict.keys():
                    suffix1_dict[line_part[0].lower()[-1]] = counter_suffix1
                    counter_suffix1 = counter_suffix1+1
                if line_part[0].lower()[-2:] not in suffix2_dict.keys():
                    suffix2_dict[line_part[0].lower()[-2:]] = counter_suffix2
                    counter_suffix2 = counter_suffix2+1
                if line_part[0].lower()[-3:] not in suffix3_dict.keys():
                    suffix3_dict[line_part[0].lower()[0:3]] = counter_suffix3
                    counter_suffix3 = counter_suffix3+1
            elif len(line_part[0].lower())>=2:
                #prefix
                if line_part[0].lower()[0] not in prefix1_dict.keys():
                    prefix1_dict[line_part[0].lower()[0]] = counter_prefix1
                    counter_prefix1 = counter_prefix1+1
                if line_part[0].lower()[0:2] not in prefix2_dict.keys():
                    prefix2_dict[line_part[0].lower()[0:2]] = counter_prefix2
                    counter_prefix2 = counter_prefix2+1
                prefix3_dict["novalues"] = 9999
                #suffix
                if line_part[0].lower()[-1] not in suffix1_dict.keys():
                    suffix1_dict[line_part[0].lower()[-1]] = counter_suffix1
                    counter_suffix1 = counter_suffix1+1
                if line_part[0].lower()[-2:] not in suffix2_dict.keys():
                    suffix2_dict[line_part[0].lower()[-2:]] = counter_suffix2
                    counter_suffix2 = counter_suffix2+1
                suffix3_dict['novalues'] = 9999
            else:
                #prefix
                if line_part[0].lower()[0] not in prefix1_dict.keys():
                    prefix1_dict[line_part[0].lower()[0]] = counter_prefix1
                    counter_prefix1 = counter_prefix1+1
                prefix2_dict["novalues"] = 9999
                prefix3_dict["novalues"] = 9999
                #suffix
                if line_part[0].lower()[-1] not in suffix1_dict.keys():
                    suffix1_dict[line_part[0].lower()[-1]] = counter_suffix1
                    counter_suffix1 = counter_suffix1+1
                suffix2_dict["novalues"] = 9999
                suffix3_dict["novalues"] = 9999
        if line_part[1] not in postag_dict.keys():
            postag_dict[line_part[1]] = counter_postag
            counter_postag = counter_postag+1
        #print(t)
        sent.append(t)
    else:
        print("train sent = ")
        print(sent)
        train_sents.append(sent)
        sent = []
        counter = counter+1

# data test
counter = 0
for line in test_lines:
    line = line.rstrip('\n')
    curr_tuple = ()
    if len(line)>1:
        line_part = line.split(" ")
        t = (line_part[0], line_part[1], convert_label(line_part[2]))
        #print(t)
        sent.append(t)
    else:
        print("test sent = ")
        print(sent)
        test_sents.append(sent)
        sent = []
        counter = counter+1


# kode untuk token/kata dan postag yang tidak muncul di data training, namun muncul di data testing
token_dict['unk'] = 9999
postag_dict['unk'] = 9999
prefix1_dict['novalues'] = 9999
prefix2_dict['novalues'] = 9999
prefix3_dict['novalues'] = 9999
suffix1_dict['novalues'] = 9999
suffix2_dict['novalues'] = 9999
suffix3_dict['novalues'] = 9999

# fungsi untuk ekstraksi fitur dari sebuah kalimat
def word2features(sent, i):  
    word = sent[i][0]
    postag = sent[i][1]
    if word.lower() not in token_dict.keys(): 
        word = 'unk'
    if postag not in postag_dict.keys():
        postag = 'unk'
    if len(word) >= 3:
        #prefix
        if word[0] not in prefix1_dict.keys():
            prefix1 = "novalues"
        else:
            prefix1 = word[0]
        if word[0:2] not in prefix2_dict.keys():
            prefix2 = "novalues"
        else:
            prefix2 = word[0:2]
        if word[0:3] not in prefix3_dict.keys():
            prefix3 = "novalues"
        else:
            prefix3 = word[0:3]
        #suffix
        if word[-1] not in suffix1_dict.keys():
            suffix1 = "novalues"
        else:
            suffix1 = word[-1]
        if word[-2:] not in suffix2_dict.keys():
            suffix2 = "novalues"
        else:
            suffix2 = word[-2:]
        if word[-3:] not in suffix3_dict.keys():
            suffix3 = "novalues"
        else:
            suffix3 = word[-3:]
    elif len(word) >=2:
        #prefix
        if word[0] not in prefix1_dict.keys():
            prefix1 = "novalues"
        else:
            prefix1 = word[0]
        if word[0:2] not in prefix2_dict.keys():
            prefix2 = "novalues"
        else:
            prefix2 = word[0:2]
        prefix3 = "novalues"
        #suffix
        if word[-1] not in suffix1_dict.keys():
            suffix1 = "novalues"
        else:
            suffix1 = word[-1]
        if word[-2:] not in suffix2_dict.keys():
            suffix2 = "novalues"
        else:
            suffix2 = word[-2:]
        suffix3 = "novalues"
    else:
        #prefix
        if word[0] not in prefix1_dict.keys():
            prefix1 = "novalues"
        else:
            prefix1 = word[0]
        prefix2 = "novalues"
        prefix3 = "novalues"
        #suffix
        if word[-1] not in suffix1_dict.keys():
            suffix1 = "novalues"
        else:
            suffix1 = word[-1]
        suffix2 = "novalues"
        suffix3 = "novalues"
    features = [
        token_dict[word.lower()], # fitur kata dalam bentuk lowercase
        word.isupper(), # fitur apakah karakter pertama token merupakan huruf kapital
        word.istitle(), # fitur apakah token merupakan title
        word.isdigit(), # fitur apakah token merupakan digit
        prefix1_dict[prefix1], #fitur 1 karakter pertama token
        prefix2_dict[prefix2], #fitur 2 karakter pertama token
        prefix3_dict[prefix3], #fitur 3 karakter pertama token
        suffix1_dict[suffix1], #fitur 1 karakter terakhir token
        suffix2_dict[suffix2], #fitur 2 karakter terakhir token
        suffix3_dict[suffix3], #fitur 3 karakter terakhir token
        postag_dict[postag] # fitur kode postag token
    ]
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]  

print('postag dictonary')
print(postag_dict)
print('token dictonary')
print(token_dict)

# ekstraksi fitur data train
X_train = []
y_train = []
for s in train_sents:
    for i in range(len(s)):
        X_train.append(word2features(s,i))
        y_train.append(s[i][2])
# ekstraksi fitur data test
X_test = []
y_test = []
for s in test_sents:
    for i in range(len(s)):
        X_test.append(word2features(s,i))
        y_test.append(s[i][2])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

# train classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Coba test, satu kata
print('xtest')
print(X_test[0])

print(clf.predict(X_test[0].reshape(1,-1)))

# Coba test, keseluruhan data test
print('hasil klasifikasi data test:')
print(clf.predict(X_test))

print("\nakurasi : ", round(clf.score(X_test,y_test)*100,2), "%")
print("\n")
print(classification_report(y_test,clf.predict(X_test)))
