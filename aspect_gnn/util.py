import collections
from nltk.tokenize import TweetTokenizer
#from tensorflow.contrib import learn

def readACL(filename,binary=False):
    with open(filename,'r',errors='ignore') as fin:
        texts = []
        targets = []
        labels = []
        for index,line in enumerate(fin):
            if index%3==0:
                texts.append(line.strip().lower())
            elif index%3==1:
                targets.append(line.strip().lower())
            else:
                labels.append(int(line.strip()))
                if binary:
                    if labels[-1] == 0:
                        labels.pop()
                        texts.pop()
                        targets.pop()
                    else:
                        if labels[-1] == 1:
                            labels[-1] = 0
    return texts,targets,labels

class vocab():
    def __init__(self,max_document_length,min_frequency,token = str.split):
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        self.token = token
        return
    def fit(self,texts):
        counter = collections.Counter({'__BLANK__':9999})
        for text in texts:
            counter.update(str.split(text))
        self.freq = counter
        
        
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))
        
'''
def build_vocab(texts,max_document_length,min_frequency=0,tokenizer_fn=None):
    
    vocab = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency=min_frequency,tokenizer_fn=tokenizer_fn)
    vocab.fit(texts)
    return vocab

'''

'''
words = []
for text in texts:
    words += text.split()
counter = collections.Counter(words)
count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

words, _ = list(zip(*count_pairs))
word_to_id = dict(zip(words, range(len(words))))
return word_to_id
'''


