import re
import spacy
import pandas as pd
nlp = spacy.load('en')
from difflib import SequenceMatcher
#print('Language:',nlp.lang)
#print('Vocabulary size:',nlp.vocab.length)
    
def sortRules (ruleList):
  
    ruleList.sort(key = len, reverse = True)
    sortedList = []
    for rule in ruleList:
        s = rule.strip().split('\t')
        splitTrig = s[0].split()
        trig = r'\s+'.join(splitTrig)
        pattern = r'\b(' + trig + r')\b'
        s.append(re.compile(pattern, re.IGNORECASE))
        sortedList.append(s)
    return sortedList


    
def convert_to_doc(text):
    doc =nlp(text)
    return doc
    
class infTagger(object):
    def __init__(self, sentence = '', phrases = None, rules = None, negP = True):
        self.__sentence = sentence
        self.__raw_sentence=sentence
        self.__phrases = phrases
        self.__rules = rules
        self.__negTaggedSentence = ''
        self.__scopesToReturn = []
        self.__negationFlag = None
        self.__test_value=[]
        self.__data=[]
        
        
        filler = '_'
        
        for i, rule in enumerate(self.__rules):
            reformatRule = re.sub(r'\s+', filler, rule[0].strip())           
            self.__sentence = rule[3].sub (' ' + rule[2].strip()+ reformatRule + rule[2].strip() + ' ', self.__sentence)        
            
        for phrase in self.__phrases:
            phrase = re.sub(r'([.^$*+?{\\|()[\]])', r'\\\1', phrase)
            splitPhrase = phrase.split()
            joiner = r'\W+'
            joinedPattern = r'\b' + joiner.join(splitPhrase) +  r'\b'
            reP = re.compile(joinedPattern, re.IGNORECASE)
            m = reP.search(self.__sentence)
            if m:
                self.__sentence = self.__sentence.replace(m.group(0), '[PHRASE]' +re.sub(r'\s+', filler, m.group(0).strip())+ '[PHRASE]')


        
        other = "No"
        mySelf= "No"
        infectionExp = "No"
        newsExp = "No"
        compainEx = "No"
        concernExp = "No"
        negExp = "No"
        vaccExp="No"
        
        sentenceTokens = self.__sentence.split()
        
        for i in range(len(sentenceTokens)):
            if sentenceTokens[i][:6] == '[INFE]':                 
                infectionExp="Yes"
            if sentenceTokens[i][:6] == '[TSEL]':
                mySelf="Yes"
            if sentenceTokens[i][:6] == '[TOTH]':
                other="Yes"
            if sentenceTokens[i][:6] == '[NEWS]':
                newsExp="Yes"
            if sentenceTokens[i][:6] == '[CAMP]':
                compainEx="Yes"
            if sentenceTokens[i][:6] == '[CONC]':
                concernExp="Yes"
            if sentenceTokens[i][:6] == '[NEGA]':
                negExp="Yes"
            if sentenceTokens[i][:6] == '[VACC]':
                vaccExp="Yes"
            
                
               
        self.__data.append(infectionExp)
        self.__data.append(other)
        self.__data.append(mySelf)
        self.__data.append(newsExp)
        self.__data.append(compainEx)
        self.__data.append(concernExp)
        self.__data.append(vaccExp)
        self.__data.append(negExp)
        self.__data.append(self.__raw_sentence)
        
    def getVector(self):
        return self.__data