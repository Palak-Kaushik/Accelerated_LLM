from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

df=pd.read_csv('router_training_data.csv')

sentences = list(df['User Query']) #sentences to be embedded from training data

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5') 
model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')



model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    #print("Sentence embeddings:", sentence_embeddings)

list_embeddings = [embedding.numpy() for embedding in sentence_embeddings] # list of embeddings of all sentences
print("shape of sentence embeddings:", sentence_embeddings.shape,"\n\n\n") #tensor to numpy convert


# adding the classifier


# creating new embedding dataframe
embedded_df= pd.DataFrame({
    'query': df['User Query'],
    'embedding': list_embeddings,
    'label' : df['Label']
})


# encoding labels

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
embedded_df['label']=encoder.fit_transform(embedded_df['label'])



# applying classification model
# random forest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Prepare features and target
X = np.array(list(embedded_df['embedding']))
y = embedded_df['label']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))


import pickle

with open('rf_classifier.bin', 'wb') as f:
    pickle.dump(clf, f)


with open('label_encoder.bin', 'wb') as le_file:
    pickle.dump(encoder, le_file)



