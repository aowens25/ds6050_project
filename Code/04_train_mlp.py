# Improved tabular MLP for CWD risk (scaled features, tuned, early stopping)

H1 = 64
H2 = 32
DROPOUT = 0.2
BATCH_SIZE = 256
LR = 1e-4
MAX_EPOCHS = 150
PATIENCE = 10
MAX_GRAD_NORM = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CWDRiskMLPImproved(nn.Module):
    def __init__(self, in_dim, h1=H1, h2=H2, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    # numeric conversion + impute
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_val = X_val.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.astype(np.float32).values)
    y_val_t = torch.tensor(y_val.astype(np.float32).values)
    y_test_t = torch.tensor(y_test.astype(np.float32).values)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # class weighting
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = min(neg / max(pos, 1.0), 10.0)
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)

    model = CWDRiskMLPImproved(in_dim=X_train_t.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_pr = -np.inf
    best_state = None
    epochs_no_improve = 0
    mlp_train_losses, mlp_val_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_train_loss = running_loss / len(train_ds)
        mlp_train_losses.append(avg_train_loss)

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(DEVICE))
            val_loss = criterion(val_logits, y_val_t.to(DEVICE)).item()
            mlp_val_losses.append(val_loss)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        val_pr_auc = average_precision_score(y_val, val_probs)
        val_roc_auc = roc_auc_score(y_val, val_probs)

        print(f"Epoch {epoch:03d}: TrainLoss={avg_train_loss:.4f}, "
              f"ValLoss={val_loss:.4f}, Val PR-AUC={val_pr_auc:.4f}, Val ROC-AUC={val_roc_auc:.4f}")

        if val_pr_auc > best_val_pr + 1e-6:
            best_val_pr = val_pr_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no PR-AUC improvement for {PATIENCE} epochs).")
                break

    # load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # F1-optimal threshold on validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t.to(DEVICE))
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)

    if len(f1_scores) > 0:
        best_idx = int(np.argmax(f1_scores))
        best_thresh = float(thresholds[best_idx])
    else:
        best_thresh = 0.5

    # test metrics
    with torch.no_grad():
        test_logits = model(X_test_t.to(DEVICE))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    test_pred = (test_probs >= best_thresh).astype(int)
    print(f"\nFinal test evaluation done with τ = {best_thresh:.3f}")

    return model, scaler, best_thresh, test_probs, mlp_train_losses, mlp_val_losses


# run MLP end-to-end
df = load_data(DATA_PATH)
X, y, train_df, val_df, test_df = prepare_features(df)

n = len(X)
train_end = int(n * 0.6)
val_end = int(n * 0.8)
X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

mlp_model, mlp_scaler, mlp_threshold, mlp_test_probs, mlp_train_losses, mlp_val_losses = train_mlp(
    X_train, y_train, X_val, y_val, X_test, y_test
)
