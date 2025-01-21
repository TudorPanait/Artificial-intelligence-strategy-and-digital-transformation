import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# a function that shows a frequency table
# with both the counts and percentages

def freqtable(series, dropna = False):
    counts = series.value_counts(dropna=dropna)
    freqs = series.value_counts(normalize = True, dropna = dropna) * 100
    result = pd.DataFrame({
        'count': counts,
        'percent': freqs
     })
    return result
    
# a function that plots the ROC curve based on probabilities

def plot_roc_curve(true_y, y_prob):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    