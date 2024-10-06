import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from func import get_word_ids
from constants import *

def metrics(data_path):

    history_data = np.load('models/history.npy', allow_pickle=True).item()

    # Extraer las métricas
    train_loss = history_data['loss']
    val_loss = history_data['val_loss']
    train_acc = history_data['categorical_accuracy']
    val_acc = history_data['val_categorical_accuracy']

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

    word_ids = get_word_ids(WORDS_JSON_PATH)
    cm = np.load('models/confusion_matrix.npy')
    cm_df = pd.DataFrame(cm, index=word_ids, columns=word_ids)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, cmap="Blues")
    plt.xticks(rotation=45) 
    plt.xlabel("Predicted Label", labelpad=20, loc='center')
    plt.ylabel("True Label", labelpad=20)
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate precision, recall, and F1-score
    report_df = pd.read_csv('models/classification_report.csv')
    print("Classification Report:")
    print(report_df)
    
'''
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Calculate precision,
    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision:", precision)

    # Calculate recall (sensitivity)
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall (Sensitivity):", recall)

    # Calculate F1-score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-Score:", f1)

    #https://medium.com/@maxgrossman10/accuracy-recall-precision-f1-score-with-python-4f2ee97e0d6
    '''

if __name__ == "__main__":
    metrics(METRICS_PATH)
    