#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re
import string
import warnings
warnings.filterwarnings('ignore')


# load ner model
model_ner = spacy.load('../3. Train Named Entity Recognition/output/model-best')

def cleanText(txt):
    whitespace = string.whitespace
    punctuation= "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tablewhitespace = str.maketrans('','',whitespace)
    tablepunctuation = str.maketrans('','',punctuation)

    text = str(txt)
    text = text.lower()
    remvoewhitespace = text.translate(tablewhitespace)
    removepunctuation = remvoewhitespace.translate(tablepunctuation)
    return str(removepunctuation)

class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
    
    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id +=1
            self.text = text
            return self.id




def parser(text, label):
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D', '', text)
        
    elif label == 'EMAIL':
        text = text.lower()
        allow_special = '@_.\-'
        text = re.sub(r'[^A-Za-z10-9{} ]'.format(allow_special),'',text)
    elif label == 'WEB':
        text = text.lower()
        allow_spe_cha = ':/.%#\-'
        text = re.sub(r'[^A-Za-z10-9{} ]'.format(allow_spe_cha),'',text)
    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]','',text)
        text = text.title()
    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]','',text)
        text = text.title()
    
    return text


grp_gen = groupgen()

def getPrediction(image):
    # extract data
    testData = pytesseract.image_to_data(image)

    #convert data into content
    test_list = list(map(lambda x: x.split('\t'), testData.split('\n')))
    df = pd.DataFrame(test_list[1:], columns=test_list[0])
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(cleanText)

    #convertdata into content
    dataClean = df.query("text != '' ")
    content = " ".join([w for w in dataClean['text']])

    # get predictions from NER model

    doc = model_ner(content)

    # converting doc into json
    docjosn = doc.to_json()

    doc_text = docjosn['text']

    # creating tokens
    dataframe_tokens = pd.DataFrame(docjosn['tokens'])
    dataframe_tokens['token'] = dataframe_tokens[['start', 'end']].apply(
        lambda x: doc_text[x[0]:x[1]], axis=1)


    right_table = pd.DataFrame(docjosn['ents'])[['start', 'label']]
    dataframe_tokens = pd.merge(dataframe_tokens, right_table, how='left', on='start')
    dataframe_tokens.fillna('O', inplace=True)

    # join label to dataClean dataframe
    dataClean['end'] = dataClean['text'].apply(lambda x : len(x)+1).cumsum() - 1
    dataClean['start'] = dataClean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)

    # inner join with start
    datarame_info = pd.merge(dataClean, dataframe_tokens[['start', 'token', 'label']], how='inner', on='start')

    # Bounding Box
    bb_df = datarame_info.query("label != 'O' ")

    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)


    # right and bottom of bounding box
    bb_df[['left', 'top', 'width', 'height']] = bb_df[['left', 'top', 'width', 'height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']

    # tagging: groupby group
    col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = bb_df[col_group].groupby(by='group')
    img_tagging = group_tag_img.agg({
        'left': min,
        'right': max,
        'top': min,
        'bottom': max,
        'label': np.unique,
        'token': lambda x: " ".join(x)
    })


    img_bb = image.copy()
    for l,r,t,b,label,token in img_tagging.values:
        cv2.rectangle(img_bb, (l,t), (r,b), (0,255,0),2)
        cv2.putText(img_bb, str(label), (l,t), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)



    info_array = datarame_info[['token', 'label']].values
    entities = dict(NAME=[], ORG=[],DES=[],PHONE=[],EMAIL=[],WEB=[])

    previous = 'O'
    for token, label in info_array:
        bio_tag = label[:1]
        label_tag = label[2:]
        # step -1 parse the token
        text = parser(token, label_tag)
        
        if bio_tag in ('B', 'I'):
            if previous != label_tag:
                entities[label_tag].append(text)
            else:
                if bio_tag == 'B':
                    entities[label_tag].append(text)
                else:
                    if label_tag in ("NAME", 'ORG', 'DES'):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
        
        previous = label_tag
                    
    return img_bb, entities

