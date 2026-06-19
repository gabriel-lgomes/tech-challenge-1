import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

LABEL_MAP = {0: 'Benigno', 1: 'Maligno'}


def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'Modelo':     name,
        'Accuracy':   round(accuracy_score(y_test, y_pred), 4),
        'Recall':     round(recall_score(y_test, y_pred), 4),
        'Precision':  round(precision_score(y_test, y_pred), 4),
        'F1-Score':   round(f1_score(y_test, y_pred), 4),
        'AUC-ROC':    round(roc_auc_score(y_test, y_proba), 4),
    }


def plot_confusion_matrix(model, X_test, y_test, title, ax):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=LABEL_MAP.values(),
                yticklabels=LABEL_MAP.values(),
                linewidths=1, linecolor='white')
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title(title)

    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(
        f'Previsto\n\nVP={tp}  FP={fp}  FN={fn}  VN={tn}\n'
        f'Recall Maligno = {tp/(tp+fn):.2%}'
    )


def plot_roc_curves(models_dict, X_test, y_test, ax):
    colors = ['#6c8ef7', '#4caf7d', '#f6c344', '#f06060']
    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, lw=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Aleatório')
    ax.set_xlabel('Taxa de Falso Positivo')
    ax.set_ylabel('Taxa de Verdadeiro Positivo (Recall)')
    ax.set_title('Curva ROC')
    ax.legend()
    ax.grid(True, alpha=0.3)
