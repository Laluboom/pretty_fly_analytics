import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 2.1  Dataset
# ---------------------------------------------------------------------------

class PrettyFlyDataset(Dataset):
    def __init__(self, X_cat, X_num, y, task_type):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        if task_type == "classification":
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


# ---------------------------------------------------------------------------
# 2.2  Network
# ---------------------------------------------------------------------------

class PrettyFlyNet(nn.Module):
    def __init__(self, cat_vocab_sizes, n_num_features, task_type, n_classes=1, emb_dim=8):
        super().__init__()
        self.task_type = task_type

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, emb_dim)
            for vocab_size in cat_vocab_sizes.values()
        ])

        emb_output_dim = len(cat_vocab_sizes) * emb_dim
        input_dim = emb_output_dim + n_num_features

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        if task_type == "regression":
            self.head = nn.Linear(64, 1)
        elif task_type == "binary":
            self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        else:
            self.head = nn.Linear(64, n_classes)

    def forward(self, x_cat, x_num):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_num], dim=1)
        x = self.mlp(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# 2.3  Loss selector
# ---------------------------------------------------------------------------

def get_loss_fn(task_type):
    if task_type == "regression":
        return nn.MSELoss()
    elif task_type == "binary":
        return nn.BCELoss()
    else:
        return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# 2.4  Training loop
# ---------------------------------------------------------------------------

def train_model(
    X_cat, X_num, y,
    task_type, n_classes,
    feature_meta,
    epochs=50,
    batch_size=512,
    lr=1e-3,
    patience=5,
    device=None,
    seed=42,
):
    # M2: seed everything for reproducible weight init and dropout
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stratify = y if task_type in ("binary", "classification") else None
    # stratify only works when every class has ≥2 samples
    if stratify is not None and len(np.unique(stratify)) < 2:
        stratify = None

    idx = np.arange(len(y))
    try:
        train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=stratify)
    except ValueError:
        train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    def make_loader(indices, shuffle):
        ds = PrettyFlyDataset(X_cat[indices], X_num[indices], y[indices], task_type)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)

    model = PrettyFlyNet(
        cat_vocab_sizes=feature_meta["cat_vocab_sizes"],
        n_num_features=X_num.shape[1],
        task_type=task_type,
        n_classes=n_classes,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = get_loss_fn(task_type)

    best_val_loss = float("inf")
    best_weights  = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for x_c, x_n, yb in train_loader:
            x_c, x_n, yb = x_c.to(device), x_n.to(device), yb.to(device)
            optimiser.zero_grad()
            out = model(x_c, x_n)
            if task_type in ("regression", "binary"):
                loss = loss_fn(out.squeeze(), yb)
            else:
                loss = loss_fn(out, yb)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_idx)

        # validate
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_c, x_n, yb in val_loader:
                x_c, x_n, yb = x_c.to(device), x_n.to(device), yb.to(device)
                out = model(x_c, x_n)
                if task_type in ("regression", "binary"):
                    loss = loss_fn(out.squeeze(), yb)
                else:
                    loss = loss_fn(out, yb)
                val_loss += loss.item() * len(yb)
                if task_type == "binary":
                    all_preds.extend(out.squeeze().cpu().numpy())
                elif task_type == "classification":
                    all_preds.extend(out.argmax(dim=1).cpu().numpy())
                else:
                    all_preds.extend(out.squeeze().cpu().numpy())
                all_targets.extend(yb.cpu().numpy())
        val_loss /= len(val_idx)

        tqdm.write(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)

    # final val metric
    all_preds    = np.array(all_preds)
    all_targets  = np.array(all_targets)

    if task_type == "binary":
        val_metric = roc_auc_score(all_targets, all_preds)
        metric_name = "val_AUC"
    elif task_type == "classification":
        val_metric = accuracy_score(all_targets, all_preds)
        metric_name = "val_accuracy"
    else:
        val_metric = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))
        metric_name = "val_RMSE"

    print(f"\n{metric_name}: {val_metric:.4f}")
    return model, best_val_loss, val_metric, metric_name, val_idx, all_preds, all_targets
