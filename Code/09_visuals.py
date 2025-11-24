def plot_confusion(ax, cm, title):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# Simple feature importance visualizers
def feature_importance_logreg(model, X, title):
    if hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
        idx = np.argsort(np.abs(coefs))[::-1][:10]
        plt.figure(figsize=(7, 4))
        plt.barh(X.columns[idx], coefs[idx])
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("Coefficient Weight")
        plt.grid(alpha=0.3)
        plt.show()
    else:
        print("Logistic regression model has no coefficients.")


def feature_importance_rf(model, X, title):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:10]
        plt.figure(figsize=(7, 4))
        plt.barh(X.columns[idx], imp[idx])
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("Gini Importance")
        plt.grid(alpha=0.3)
        plt.show()
    else:
        print("Random forest model missing feature_importances_ attribute.")


def feature_importance_mlp(model, X, y, title):
    from sklearn.inspection import permutation_importance
    try:
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=10,
            random_state=42,
            scoring="average_precision"
        )
        imp = result.importances_mean
        idx = np.argsort(imp)[::-1][:10]
        plt.figure(figsize=(7, 4))
        plt.barh(X.columns[idx], imp[idx])
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("Permutation Importance")
        plt.grid(alpha=0.3)
        plt.show()
    except Exception as e:
        print("Could not compute MLP permutation importance:", e)


#MLP Training Curves
try:
    plt.figure(figsize=(8, 5))
    plt.plot(mlp_train_losses, label="Train Loss")
    plt.plot(mlp_val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
except Exception as e:
    print("Training curves not available. If you logged train/val loss, "
          "add them as mlp_train_losses and mlp_val_losses.")
    print("Error:", e)


#Model performance side-by-side comparison
model_names = ["LogReg", "RandomForest", "MLP"]
test_ap = [
    average_precision_score(y_test, y_test_prob_lr),
    average_precision_score(y_test, y_test_prob_rf),
    average_precision_score(y_test, y_test_prob_mlp),
]

plt.figure(figsize=(6, 4))
plt.bar(model_names, test_ap)
plt.ylabel("Test Average Precision")
plt.title("Model Comparison (PR-AUC)")
plt.grid(alpha=0.3)
plt.show()


# ROC + PR Curves for all models side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC
for name, probs in zip(
    model_names,
    [y_test_prob_lr, y_test_prob_rf, y_test_prob_mlp]
):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0].set_title("ROC Curves")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].legend()
axes[0].grid(alpha=0.3)

# PR
for name, probs in zip(
    model_names,
    [y_test_prob_lr, y_test_prob_rf, y_test_prob_mlp]
):
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})")

axes[1].set_title("Precision-Recall Curves")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


#Threshold extraction for all models 
tau_logreg = taus.get("LogReg (scaled)", 0.5)
tau_rf = taus.get("Random Forest", 0.5)
tau_mlp = taus.get("MLP (scaled)", 0.5)


# Model-specific visuals
def full_model_visuals(model_name, probs, tau):
    preds = (probs >= tau).astype(int)
    cm = confusion_matrix(y_test, preds)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Confusion
    plot_confusion(axs[0], cm, f"{model_name} (τ={tau:.3f})")

    # Probability histogram
    axs[1].hist(probs, bins=30, alpha=0.75)
    axs[1].set_title(f"{model_name} Probability Distribution")
    axs[1].set_xlabel("Predicted Probability")

    # Calibration Curve
    true_means = []
    pred_means = []
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            true_means.append(y_test[mask].mean())
            pred_means.append(probs[mask].mean())

    axs[2].plot(pred_means, true_means, marker="o")
    axs[2].plot([0, 1], [0, 1], "--", color="gray")
    axs[2].set_title(f"{model_name} Calibration Curve")
    axs[2].set_xlabel("Predicted Mean")
    axs[2].set_ylabel("Observed Frequency")

    plt.tight_layout()
    plt.show()


#Run for all three models 
full_model_visuals("Logistic Regression", y_test_prob_lr, tau_logreg)
full_model_visuals("Random Forest", y_test_prob_rf, tau_rf)
full_model_visuals("MLP", y_test_prob_mlp, tau_mlp)

#Feature Importance Visuals
feature_importance_logreg(
    logreg_model, X_test, "Logistic Regression: Top Features"
)
feature_importance_rf(
    rf_model, X_test, "Random Forest: Top Features"
)
feature_importance_mlp(
    mlp_model, X_test, y_test.to_numpy(),
    "MLP: Top Features (Permutation)"
)

#Random Forest Ablation Summary
imp = rf_model.feature_importances_
idx = np.argsort(imp)[::-1][:10]

plt.figure(figsize=(7, 4))
plt.barh(X_test.columns[idx], imp[idx])
plt.gca().invert_yaxis()
plt.title("Random Forest: Top 10 Features (Ablation-Adjacent Summary)")
plt.xlabel("Gini Importance")
plt.grid(alpha=0.3)
plt.show()

print("All final-deliverable visualizations generated.")

