# document-scanner-app-with-ocr-ner

**0. Project Setup**
   - Create Virtual env
   - install requirements.txt
   - install pytesseract: https://pypi.org/project/pytesseract/
   - install spacy: https://spacy.io/usage

**1. Data Preparation**
   - How pytesseract work
   - How BIO/IOB format works

**2. Data PreProcessing**
   - Convert the data into Spacy format
   - Spliting data into Training and test dataset

**3. Train Named Entity Recognition**
   - download base_config.cfg file: https://spacy.io/usage/training
   - setup config.cfg file from base_config.cfg 3rd line
   - prepare data -- train.spacy, test.spacy
   - python -m spacy train ./config.cfg --output ./output/ --paths.train ./spacy/train.spacy --paths.dev ./spacy/test.spacy

**4. Predictions**
   - Building predictions pipeline

**5. Improve Model Performance**
   - Cleaning text for better accuracy -> data_preprocessing
   6. Document Scanner

**6. Document Scanner**
   - 

**6. Document Scanner Web App**
   - 
