import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from func import get_word_ids, get_sequences_and_labels
from constants import *

def metrics(data_path):
    # Cargar el history
    with open('history.pkl', 'rb') as file:
        history = pickle.load(file)
   
    model = load_model(MODEL_PATH)

    word_ids = get_word_ids(WORDS_JSON_PATH) # ['word1', 'word2', 'word3]
    sequences, labels = get_sequences_and_labels(word_ids)
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float16')
    x = np.array(sequences)
    y = to_categorical(labels).astype(int) 
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, random_state=42)

    # Get training and validation metrics
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

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

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Calculate precision, recall, and F1-score
    report = classification_report(y_test, y_pred, target_names=word_ids)
    print("Classification Report:")
    print(report)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision:", precision)

    # Calculate recall (sensitivity)
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall (Sensitivity):", recall)

    # Calculate F1-score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-Score:", f1)

    #https://medium.com/@maxgrossman10/accuracy-recall-precision-f1-score-with-python-4f2ee97e0d6
    
if __name__ == "__main__":
    metrics(METRICS_PATH)
    