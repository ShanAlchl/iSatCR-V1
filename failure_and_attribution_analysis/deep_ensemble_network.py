import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple, Union


def _resolve_category_sizes(
    num_categories: Union[int, Sequence[int]],
    num_discrete_features: int,
) -> List[int]:
    if isinstance(num_categories, int):
        return [int(num_categories)] * int(num_discrete_features)

    category_sizes = [int(size) for size in num_categories]
    if len(category_sizes) != int(num_discrete_features):
        raise ValueError("num_categories length must match num_discrete_features")
    return category_sizes


class EmbeddedMLP(nn.Module):
    """
    MLP with an embedding branch for one discrete feature plus multiple continuous features.
    """

    def __init__(
        self,
        num_continuous: int,
        num_categories: Union[int, Sequence[int]],
        num_discrete_features: int = 1,
        embedding_dim: int = 4,
    ):
        super().__init__()
        category_sizes = _resolve_category_sizes(num_categories, num_discrete_features)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim) for size in category_sizes]
        )
        self.num_discrete_features = len(category_sizes)

        input_dim = num_continuous + embedding_dim * self.num_discrete_features
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 80)
        self.fc3 = nn.Linear(80, 80)
        self.out = nn.Linear(80, 1)

    def forward(self, continuous_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        if discrete_x.dim() == 1:
            discrete_x = discrete_x.unsqueeze(1)
        if discrete_x.shape[1] != self.num_discrete_features:
            raise ValueError("discrete_x feature count does not match EmbeddedMLP configuration")

        embedded_parts = [
            embedding(discrete_x[:, idx].long())
            for idx, embedding in enumerate(self.embeddings)
        ]
        embedded_x = torch.cat(embedded_parts, dim=1)
        x = torch.cat([continuous_x, embedded_x], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.out(x))


class DeepNarrowMLP(nn.Module):
    """
    Deeper but narrower MLP specialized for continuous inputs.
    """

    def __init__(self, num_continuous: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_continuous, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 48),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(48, 32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, continuous_x: torch.Tensor, discrete_x: torch.Tensor = None) -> torch.Tensor:
        return self.net(continuous_x)


class TemporalGRU(nn.Module):
    """
    Lightweight GRU for sequence-aware failure scoring.
    """

    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(seq_x)
        last_step_out = gru_out[:, -1, :]
        return torch.sigmoid(self.fc(last_step_out))


class DeepEnsembleNetwork(nn.Module):
    """
    Heterogeneous ensemble that returns both failure score and coefficient-of-variation uncertainty.
    """

    def __init__(self, models: List[nn.Module], model_weights: List[float]):
        super().__init__()
        self.models = nn.ModuleList(models)
        weights_tensor = torch.tensor(model_weights, dtype=torch.float32)
        if weights_tensor.sum() > 0:
            weights_tensor = weights_tensor / weights_tensor.sum()
        self.register_buffer("weights", weights_tensor)

    def set_model_weights(self, model_weights: List[float]):
        weights_tensor = torch.tensor(model_weights, dtype=torch.float32, device=self.weights.device)
        if weights_tensor.sum() > 0:
            weights_tensor = weights_tensor / weights_tensor.sum()
        self.weights.copy_(weights_tensor)

    def forward(
        self,
        continuous_x: torch.Tensor,
        discrete_x: torch.Tensor,
        seq_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for model in self.models:
            if isinstance(model, EmbeddedMLP):
                pred = model(continuous_x, discrete_x)
            elif isinstance(model, DeepNarrowMLP):
                pred = model(continuous_x)
            elif isinstance(model, TemporalGRU):
                in_seq = seq_x if seq_x is not None else continuous_x.unsqueeze(1)
                pred = model(in_seq)
            else:
                raise ValueError("Unknown base-model type in DeepEnsembleNetwork")
            predictions.append(pred)

        all_preds = torch.stack(predictions, dim=0)
        weights_expanded = self.weights.view(-1, 1, 1).to(all_preds.device)
        ensemble_score = (all_preds * weights_expanded).sum(dim=0)
        std_preds = all_preds.std(dim=0, unbiased=False)
        mean_clamped = torch.clamp(ensemble_score, min=1e-6)
        cv_uncertainty = std_preds / mean_clamped
        return ensemble_score, cv_uncertainty

    def fit_incremental(
        self,
        continuous_x: torch.Tensor,
        discrete_x: torch.Tensor,
        targets: torch.Tensor,
        seq_x: Optional[torch.Tensor] = None,
        epochs: int = 12,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[torch.device] = None,
    ) -> Dict[str, List[float]]:
        """
        Fast incremental training for newly accumulated scenario-score samples.
        """

        if len(continuous_x) == 0:
            return {"model_rmse": [], "model_weights": self.weights.detach().cpu().tolist()}

        train_device = device or next(self.parameters()).device
        self.to(train_device)

        continuous_x = continuous_x.to(train_device)
        discrete_x = discrete_x.to(train_device)
        targets = targets.to(train_device).view(-1, 1)
        if seq_x is not None:
            seq_x = seq_x.to(train_device)

        num_samples = continuous_x.shape[0]
        effective_batch_size = max(1, min(int(batch_size), num_samples))
        loss_fn = nn.SmoothL1Loss(beta=0.1)

        for model in self.models:
            model.train()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            for _ in range(max(1, int(epochs))):
                permutation = torch.randperm(num_samples, device=train_device)
                for start_idx in range(0, num_samples, effective_batch_size):
                    batch_indices = permutation[start_idx:start_idx + effective_batch_size]
                    batch_c = continuous_x[batch_indices]
                    batch_d = discrete_x[batch_indices]
                    batch_y = targets[batch_indices]

                    if isinstance(model, EmbeddedMLP):
                        preds = model(batch_c, batch_d)
                    elif isinstance(model, DeepNarrowMLP):
                        preds = model(batch_c)
                    elif isinstance(model, TemporalGRU):
                        batch_seq = seq_x[batch_indices] if seq_x is not None else batch_c.unsqueeze(1)
                        preds = model(batch_seq)
                    else:
                        raise ValueError("Unknown base-model type in DeepEnsembleNetwork")

                    loss = loss_fn(preds, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        rmse_values: List[float] = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                if isinstance(model, EmbeddedMLP):
                    preds = model(continuous_x, discrete_x)
                elif isinstance(model, DeepNarrowMLP):
                    preds = model(continuous_x)
                elif isinstance(model, TemporalGRU):
                    in_seq = seq_x if seq_x is not None else continuous_x.unsqueeze(1)
                    preds = model(in_seq)
                else:
                    raise ValueError("Unknown base-model type in DeepEnsembleNetwork")

                mse = torch.mean((preds - targets) ** 2).item()
                rmse_values.append(float(np.sqrt(max(mse, 1e-12))))

        inverse_rmse = 1.0 / np.clip(np.array(rmse_values, dtype=np.float64), 1e-6, None)
        normalized_weights = (inverse_rmse / inverse_rmse.sum()).tolist()
        self.set_model_weights(normalized_weights)
        self.eval()

        return {
            "model_rmse": rmse_values,
            "model_weights": normalized_weights,
        }


def build_default_deep_ensemble(
    num_continuous: int,
    num_categories: Union[int, Sequence[int]],
    num_discrete_features: int = 1,
    embedding_dim: int = 4,
) -> DeepEnsembleNetwork:
    models = [
        EmbeddedMLP(
            num_continuous=num_continuous,
            num_categories=num_categories,
            num_discrete_features=num_discrete_features,
            embedding_dim=embedding_dim,
        ),
        DeepNarrowMLP(num_continuous=num_continuous),
        DeepNarrowMLP(num_continuous=num_continuous),
    ]
    return DeepEnsembleNetwork(models=models, model_weights=[1.0, 1.0, 1.0])


if __name__ == "__main__":
    num_continuous = 12
    num_categories = 5

    models_list = [
        EmbeddedMLP(num_continuous, num_categories),
        EmbeddedMLP(num_continuous, num_categories, embedding_dim=8),
        DeepNarrowMLP(num_continuous),
        DeepNarrowMLP(num_continuous),
        TemporalGRU(num_continuous + 1),
        TemporalGRU(num_continuous + 1, hidden_size=32),
    ]

    rmses = np.array([0.1, 0.12, 0.08, 0.09, 0.15, 0.14])
    model_weights = (1.0 / rmses / (1.0 / rmses).sum()).tolist()

    ensemble_net = DeepEnsembleNetwork(models_list, model_weights)

    batch_sz = 4
    dummy_continuous = torch.rand(batch_sz, num_continuous)
    dummy_discrete = torch.randint(0, num_categories, (batch_sz, 1))
    dummy_seq = torch.rand(batch_sz, 10, num_continuous + 1)
    scores, cvs = ensemble_net(dummy_continuous, dummy_discrete, dummy_seq)
    print("Ensemble Predicted Failure Scores:\n", scores.detach().numpy())
    print("Prediction CV (Uncertainty):\n", cvs.detach().numpy())
