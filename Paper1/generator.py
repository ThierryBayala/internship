from infection import *
from lxml import etree as ET
import xml.etree.cElementTree as ET


rfile = open(r'triggers/infection_triggers.txt')
irules = sortRules(rfile.readlines())
def generator_vec(doc):
    data_xml=[]
    #root = ET.Element("Data")
    doc.user_data=set()
    if len(list(doc.sents))==1:
        for sent in doc.sents:
            ph= set(doc.user_data)
            for word in sent:
                if word.pos_!='ADP' and word.pos_!='PUNCT':
                    ph.add(word.text)
            tagger = infTagger(sentence = sent.text, phrases = list(ph),rules = irules, negP=False)
            vec = tagger.getVector()

           # ET.SubElement(doc, "Tweet").text = 
            data_xml.append(vec[8])
            #ET.SubElement(doc, "Infection").text = 
            data_xml.append(vec[0])
            #ET.SubElement(doc, "Other").text =
            data_xml.append(vec[1])
            #ET.SubElement(doc, "Self").text = 
            data_xml.append(vec[2])
            #ET.SubElement(doc, "News").text = 
            data_xml.append(vec[3])
            #ET.SubElement(doc, "Campaign").text = 
            data_xml.append(vec[4])
            #ET.SubElement(doc, "Concern").text = 
            data_xml.append(vec[5])
            #ET.SubElement(doc, "Vaccine").text = 
            data_xml.append(vec[6])
            #ET.SubElement(doc, "Negation").text = 
            data_xml.append(vec[7])
    
    return data_xml

#define a new pipleline including the negation_tag component
def custom_pipeline(nlp):
    return (nlp.tagger,nlp.parser,generator_vec)

#need to re-initlaize spaCy with the new pipeline
nlp_neg = spacy.load('en', create_pipeline=custom_pipeline)



