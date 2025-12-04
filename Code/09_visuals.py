import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def run_all_visuals(
    y_test,
    y_test_prob_lr,
    y_test_prob_rf,
    y_test_prob_mlp,
    mlp_train_losses,
    mlp_val_losses
):
    plt.figure()
    plt.plot(mlp_train_losses, label="train")
    plt.plot(mlp_val_losses, label="val")
    plt.legend()
    plt.title("MLP Loss Curves")
    plt.show()

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_prob_rf)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_test_prob_mlp)

    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label="LR")
    plt.plot(fpr_rf, tpr_rf, label="RF")
    plt.plot(fpr_mlp, tpr_mlp, label="MLP")
    plt.legend()
    plt.title("ROC Curves")
    plt.show()

    prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_test_prob_lr)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_test_prob_rf)
    prec_mlp, rec_mlp, _ = precision_recall_curve(y_test, y_test_prob_mlp)

    plt.figure()
    plt.plot(rec_lr, prec_lr, label="LR")
    plt.plot(rec_rf, prec_rf, label="RF")
    plt.plot(rec_mlp, prec_mlp, label="MLP")
    plt.legend()
    plt.title("PR Curves")
    plt.show()

    cm_lr = confusion_matrix(y_test, (y_test_prob_lr >= 0.5).astype(int))
    cm_rf = confusion_matrix(y_test, (y_test_prob_rf >= 0.5).astype(int))
    cm_mlp = confusion_matrix(y_test, (y_test_prob_mlp >= 0.5).astype(int))

    print("Confusion Matrix LR")
    print(cm_lr)
    print("Confusion Matrix RF")
    print(cm_rf)
    print("Confusion Matrix MLP")
    print(cm_mlp)
