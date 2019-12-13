# Ade Romadhony
# Contoh sederhana NER berbasis aturan yang didefinisikan secara manual

# read the file
lines = []
with open('kalimat_POSTag.txt', 'r') as f:
    lines = f.readlines()

counter_line = 0
tokens = []
postags = []
labels = ["B-PER","B-ORG","B-LOC","I-PER","I-ORG","I-LOC"]
for line in lines:
    line = line.rstrip('\n')
    if len(line)>1:
        line_part = line.split(" ")
        tokens.append(line_part[0])
        postags.append(line_part[1])
    else:
        print(tokens)
        print(postags)
        NE_labels = []
        counter_token = 0
        prev_NE_label = ""
        prev_token = ""
        for token in tokens:
            if token[0].isupper() and postags[counter_token]=='NNP':
                if prev_NE_label in labels:
                    if prev_NE_label=="B-PER" or prev_NE_label=="I-PER":
                        NE_labels.append("I-PER")
                    elif prev_NE_label=="B-ORG" or prev_NE_label=="I-ORG":
                        NE_labels.append("I-ORG")
                    else: 
                        NE_labels.append("I-LOC")
                else:
                    if prev_token == "di" or prev_token == "ke":
                        NE_labels.append("B-LOC")
                    elif token.isupper():
                        NE_labels.append("B-ORG")
                    else: 
                        NE_labels.append("B-PER")
            else:
                NE_labels.append("O")
            prev_NE_label = NE_labels[counter_token]
            counter_token += 1
        print(tokens)
        print(NE_labels)
        tokens = []
        postags = []
