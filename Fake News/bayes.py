import collections
import itertools
import math, os, pickle, re
import pandas as pd


class Bayes_Classifier:
    def __init__(self, faketrain, realtrain):
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text.'''
        self.trainDirectory = faketrain, realtrain
        self.positive = collections.defaultdict(int)
        self.negative = collections.defaultdict(int)
        self.negative_text = 0
        self.positive_text = 0
        self.truepositive = 0
        self.falsepositive = 0
        self.truenegative = 0
        self.falsenegative = 0
        if os.path.isfile('positive.pk1') and os.path.isfile('negative.pk1'):
            self.positive = self.load('positive.pk1')
            self.negative = self.load('negative.pk1')
            self.negative_text = self.load('negative_text.pk1')
            self.positive_text = self.load('positive_text.pk1')
        else:
            self.train()
        self.total_positive = sum(self.positive.values())
        self.total_negative = sum(self.negative.values())

    def train(self):
        '''Trains the Naive Bayes Sentiment Classifier.'''
        self.positive_text = 0
        self.negative_text = 0
        for ft, rt in itertools.zip_longest(fake_train, real_train, fillvalue=''):
            faketrain = [ft]
            realtrain = [rt]
            for data in faketrain:
                if data != '':
                    self.negative_text += 1
                    textstring = self.tokenize(data)
                    for word in textstring:
                        self.negative[str(word.lower())] += 1
            for data in realtrain:
                if data != '':
                    self.positive_text += 1
                    textstring = self.tokenize(data)
                    for word in textstring:
                        self.positive[str(word.lower())] += 1
        self.save(self.positive, 'positive.pk1')
        self.save(self.negative, 'negative.pk1')
        self.save(self.positive_text, 'positive_text.pk1')
        self.save(self.negative_text, 'negative_text.pk1')

    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''
        pos = math.log10(1.0 * self.positive_text / (self.positive_text + self.negative_text))
        neg = math.log10(1.0 * self.negative_text / (self.positive_text + self.negative_text))
        for w in self.tokenize(sText):
            w = w.lower()
            pos += math.log10((self.positive[w] + 1.0) / self.total_positive)
            neg += math.log10((self.negative[w] + 1.0) / self.total_negative)
        if pos > neg:
            return "real"
        elif neg > pos:
            return "fake"

    def save(self, dObj, sFilename):
        '''Given an object and a file name, write the object to the file using pickle.'''
        f = open(sFilename, "wb")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        '''Given a file name, load and return the object stored in the file.'''
        f = open(sFilename, "rb")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        '''Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order).'''
        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))
        if sToken != "":
            lTokens.append(sToken)
        return lTokens

    def performance(self, test, faketest, truetest):
        self.truepositive = 0
        self.falsepositive = 0
        self.truenegative = 0
        self.falsenegative = 0
        for text in test:
            result = self.classify(text)
            for data in faketest:
                if data == text:
                    if result == 'fake':
                        self.truenegative += 1
                    else:
                        self.falsenegative += 1
            for data in truetest:
                if data == text:
                    if result == 'real':
                        self.truepositive += 1
                    else:
                        self.falsepositive += 1
        print("Performance Meausres")
        print('True Positive : %d ' % self.truepositive)
        print('False Positive : %d ' % self.falsepositive)
        print('True Negative : %d ' % self.truenegative)
        print('False Negative : %d ' % self.falsenegative)
        self.calculationFunction()

    def calculationFunction(self):
        Accuracy = (self.truepositive + self.truenegative) / (
                self.truepositive + self.falsepositive + self.falsenegative + self.truenegative)
        print('Accuracy [TP+TN+TNu/TP+FP+FN+TN+FNu+TNu] = %.3f ' % Accuracy)
        Precision = self.truepositive / (self.truepositive + self.falsepositive)
        print('Precision [TP/TP+FP] =  %.3f ' % Precision)
        Recall = self.truepositive / (self.truepositive + self.falsenegative)
        print('Recall [TP/TP+FN+FNu] =  %.3f ' % Recall)
        F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
        print('F1 Score [2 * (Recall * Precision)/(Recall + Precision)] =  %.3f ' % F1_Score)


fakenews = pd.read_csv('C:\\Users\\nijoj\\Desktop\\New_Detection\\data\\Fake.csv')
truenews = pd.read_csv('C:\\Users\\nijoj\\Desktop\\New_Detection\\data\\True.csv')

fake_train = fakenews['title'][4697:]
real_train = truenews['title'][4284:]
test_fake = fakenews['title'][:4697]
test_real = truenews['title'][:4284]
bc = Bayes_Classifier(fake_train, real_train)
test = []
faketest = []
truetest = []
for ft, rt in itertools.zip_longest(test_fake, test_real, fillvalue=''):
    if ft != '':
        test.append(ft)
        faketest.append(ft)
    if rt != '':
        test.append(rt)
        truetest.append(rt)
results = {"fake": 0, "real": 0}
for text in test:
    result = bc.classify(text)
    results[result] += 1
print("\nResults Summary:");
for r in results:
    print("%s: %d" % (r, results[r]));
print("\n")
bc.performance(test, faketest, truetest)
