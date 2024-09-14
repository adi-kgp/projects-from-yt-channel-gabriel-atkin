"""
# ## Disneyland Review Rating Prediction
# 
# Given *reviews of Disneyland*, let's try to predict the **rating** associated with a given review.
# 
# We will use a Tensorlflow/Keras text model with word embeddings to make our predictions.
# 
# Data source: https://www.kaggle.com/datasets/arushchillar/disneyland-reviews
"""

# ### Getting Started

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf


data = pd.read_csv("DisneylandReviews.csv", encoding='latin-1')

print(data.head())

print(data.info())


### Preprocessing

def get_sequences(texts, tokenizer, train=True, max_seq_length=None):
    sequences = tokenizer.texts_to_sequences(texts)
    
    if train == True:
        max_seq_length = np.max(list(map(len, sequences)))
        
    sequences = pad_sequences(sequences, maxlen = max_seq_length, padding='post')
    
    return sequences

def preprocess_inputs(df):
    df = df.copy()
    
    # Limit the data to only the review and rating columns
    y = df['Rating']
    X = df['Review_Text']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=0.7, 
                                                        shuffle=True,
                                                        random_state=1)
    # Fit tokenizer 
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)
    print("Vocab length:", len(tokenizer.word_index) + 1)
    
    # Convert texts to sequences 
    X_train = get_sequences(X_train, tokenizer, train=True)
    X_test = get_sequences(X_test, tokenizer, train=False, 
                           max_seq_length=X_train.shape[1])
    
    return X_train, X_test, y_train, y_test, tokenizer

X_train, X_test, y_train, y_test, t = preprocess_inputs(data)

## Training
X_train.shape

inputs = tf.keras.Input(shape=(X_train.shape[1],))
x = tf.keras.layers.Embedding(
        input_dim=37846,
        output_dim=64
    )(inputs)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
        optimizer = 'adam',
        loss='mse'
    )

history = model.fit(
        X_train,
        y_train,
        validation_split = 0.2,
        batch_size=32,
        epochs=100,
        callbacks=[
                tf.keras.callbacks.EarlyStopping(
                        monitor = 'val_loss',
                        patience=3,
                        restore_best_weights = True
                    )
            ]
    )


## Results
y_pred = np.squeeze(model.predict(X_test))


rmse = np.sqrt(np.mean((y_test - y_pred)**2))
r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))

print("RMSE : {:.2f}".format(rmse))
print("R^2 Score: {:.5f}".format(r2))





























