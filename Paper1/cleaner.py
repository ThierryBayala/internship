from normalizer import *
rfile = open(r'triggers/apostrophe.txt')
irules = sortRules(rfile.readlines())


def negation_tag(doc):
    data=[]
    doc.user_data['negated']=set()
    for sent in doc.sents:
        ph= set()
        for word in sent:
            if word.pos_!='ADP' and word.pos_!='PUNCT':
                ph.add(word.text)
        tagger = negTagger(sentence = sent.text, phrases = list(ph),rules = irules, negP=False)
        scopes=  tagger.getScopes()
        
        data.append(tagger.getNoApos())
        
        res = set()
        for scope in scopes:
            s = scope.replace('[NEGATED]','').replace('.','').replace(',','')
            if ' ' in s:
                for wd in s.split(' '):
                    res.add(wd)
            else:
                res.add(s)
                
        for word in sent:
            if word.text in res:
                doc.user_data['negated'].add(word.i)
    return data

#define a new pipleline including the negation_tag component
def custom_pipeline(nlp):
    return (nlp.tagger,nlp.parser,negation_tag)

#need to re-initlaize spaCy with the new pipeline
nlp_neg = spacy.load('en', create_pipeline=custom_pipeline)