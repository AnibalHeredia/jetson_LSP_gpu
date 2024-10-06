import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from func import get_word_ids, get_sequences_and_labels
from constants import *

def training_model(model_path, epochs=100):
    word_ids = get_word_ids(WORDS_JSON_PATH) # ['word1', 'word2', 'word3]
    
    sequences, labels = get_sequences_and_labels(word_ids)
    
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float16')
    
    x = np.array(sequences)
    y = to_categorical(labels).astype(int) 
    
    #early_stopping = EarlyStopping(monitor='accuracy', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, random_state=42)
    
    model = get_model(int(MODEL_FRAMES), len(word_ids))
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=8, callbacks=[checkpoint])
    
    history_data = {key: np.array(value) for key, value in history.history.items()}
    np.save('models/history.npy', history_data)

    # Get training and validation metrics
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

   # Create loss metrics graphs
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Create accuracy metrics graphs
    plt.plot(train_acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Calculate predictions
    y_pred = model.predict(x_val)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_val, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=word_ids, columns=word_ids)
    np.save('models/confusion_matrix.npy', cm_df.to_numpy())

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, cmap="Blues")
    plt.xticks(rotation=45) 
    plt.xlabel("Predicted Label", labelpad=20, loc='center')
    plt.ylabel("True Label", labelpad=20)
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate precision, recall, and F1-score
    report = classification_report(y_test, y_pred, target_names=word_ids,output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('models/classification_report.csv', index=True,sep=';')  # index=True guarda el índice
    print("Classification Report:")
    print(report)

    model.summary()
    #model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)
    
