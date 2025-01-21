import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
    
# a function that plots the ROC curve based on probabilities

def plot_roc_curve(true_y, y_prob):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    