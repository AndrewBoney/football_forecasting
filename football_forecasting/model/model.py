import torch.nn as nn
import torch.distributions as dist
import lightning.pytorch as pl

import torch

from torch.utils.data import DataLoader
from datasets import load_dataset
from torchmetrics.functional import accuracy, precision, recall, f1_score

# Define the Poisson / ELO module
class PoissonProbabilityModule(nn.Module):
    def __init__(self, max_goals: int = 10):
        """
        Converts expected goals (λ) into Poisson probabilities.
        
        Args:
            max_goals (int): Maximum number of goals to consider.
                             The module will output explicit probabilities for 
                             0, 1, ..., max_goals and one extra tail probability for outcomes > max_goals.
        """
        super().__init__()
        self.max_goals = max_goals

    def forward(self, expected_values: torch.Tensor):
        """
        Args:
            expected_values (Tensor): shape (batch_size,) of λ values.
        
        Returns:
            Tensor: shape (batch_size, max_goals + 2) containing probabilities for
                    goal counts 0, 1, …, max_goals and the tail probability for > max_goals.
        """
        goal_range = torch.arange(self.max_goals + 1, device=expected_values.device)
        # Compute probabilities for outcomes 0, 1, …, max_goals
        probabilities = dist.Poisson(expected_values.reshape(-1, 1)).log_prob(goal_range).exp()
        # Tail probability for outcomes > max_goals
        tail_prob = 1 - probabilities.sum(dim=1, keepdim=True)
        probabilities = torch.cat([probabilities, tail_prob], dim=1)
        return probabilities


class BaseModel(pl.LightningModule):
    """
    Base LightningModule that contains common functionality for model classes.
    Shared features:
    - a dummy trainable parameter so models always have parameters
    - generic negative log-likelihood helper that maps targets into a tail bin
    - generic metric computation for multiclass prediction over goal counts
    - generic optimizer construction using `self.hparams.lr` when available
    """
    def __init__(self, max_goals: int = 10, lr: float = 1e-3):
        super().__init__()
        # store defaults; child classes should call `self.save_hyperparameters()`
        # to persist constructor args into `self.hparams` if desired
        self.max_goals = max_goals
        self.lr = lr

    def adjust_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Map target goal counts greater than `self.max_goals` into the tail bin index.

        Args:
            targets (Tensor): integer goal counts, shape (batch,)

        Returns:
            Tensor: adjusted integer targets (same shape) where values > max_goals
                    are replaced with `max_goals + 1` (the tail bin index).
        """
        tail_index = self.max_goals + 1
        return torch.where(
            targets > self.max_goals,
            torch.full_like(targets, tail_index),
            targets
        )

    def _nll_loss(self, prob_dist: torch.Tensor, targets: torch.Tensor):
        targets_adjusted = self.adjust_targets(targets)
        batch_idx = torch.arange(prob_dist.size(0), device=prob_dist.device)
        eps = 1e-8  # to avoid log(0)
        nll = -torch.log(prob_dist[batch_idx, targets_adjusted] + eps)
        return nll.mean()

    def _compute_metrics(self, probs: torch.Tensor, targets: torch.Tensor, prefix: str):
        targets_adjusted = self.adjust_targets(targets)
        preds = torch.argmax(probs, dim=1)
        metrics = {}
        num_classes = self.max_goals + 2
        metrics[f"{prefix}_acc"] = accuracy(
            preds, targets_adjusted, task="multiclass", num_classes=num_classes
        )
        metrics[f"{prefix}_precision"] = precision(
            preds, targets_adjusted, task="multiclass", num_classes=num_classes, average="macro"
        )
        metrics[f"{prefix}_recall"] = recall(
            preds, targets_adjusted, task="multiclass", num_classes=num_classes, average="macro"
        )
        metrics[f"{prefix}_f1"] = f1_score(
            preds, targets_adjusted, task="multiclass", num_classes=num_classes, average="macro"
        )
        return metrics

    def configure_optimizers(self):
        lr = getattr(self.hparams, 'lr', getattr(self, 'lr', 1e-3))
        return torch.optim.Adam(self.parameters(), lr=lr)


class PoissonModel(BaseModel):
    def __init__(self, max_goals: int = 10, lr: float = 1e-3):
        """
        Model that uses Poisson probabilities to compute a negative log likelihood loss
        and evaluates standard classification metrics.

        Args:
            max_goals (int): Maximum number of goals to consider before using the tail bin.
            lr (float): Learning rate.
        """
        # Initialize base class (creates dummy param and stores max_goals and lr)
        super().__init__(max_goals=max_goals, lr=lr)
        # Persist hyperparameters (so `self.hparams.max_goals` and `self.hparams.lr` exist)
        self.save_hyperparameters()
        # Use the BaseModel-stored max_goals for the probability module
        self.poisson_module = PoissonProbabilityModule(max_goals=self.max_goals)
        self.dummy_param = nn.Parameter(torch.zeros(1))


    def forward(self, expected_home: torch.Tensor, expected_away: torch.Tensor):
        """
        Given expected goals for home and away teams, returns their Poisson probability distributions.

        Args:
            expected_home (Tensor): shape (batch_size,)
            expected_away (Tensor): shape (batch_size,)

        Returns:
            Tuple[Tensor, Tensor]: home and away probabilities, each of shape (batch_size, max_goals+2)
        """
        home_prob = self.poisson_module(expected_home)
        away_prob = self.poisson_module(expected_away)
        return home_prob, away_prob

    # Use BaseModel._nll_loss and BaseModel._compute_metrics directly

    def training_step(self, batch, batch_idx):
        expected_home = batch['expected_home']
        expected_away = batch['expected_away']
        home_goals = batch['home_goals']
        away_goals = batch['away_goals']

        home_prob, away_prob = self(expected_home, expected_away)
        loss_home = self._nll_loss(home_prob, home_goals)
        loss_away = self._nll_loss(away_prob, away_goals)
        loss = loss_home + loss_away + self.dummy_param.sum() * 0.0

        # Compute metrics for home predictions
        home_metrics = self._compute_metrics(home_prob, home_goals, prefix="train_home")
        away_metrics = self._compute_metrics(away_prob, away_goals, prefix="train_away")

        self.log("train_loss", loss, prog_bar=True)
        for name, value in {**home_metrics, **away_metrics}.items():
            self.log(name, value, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        expected_home = batch['expected_home']
        expected_away = batch['expected_away']
        home_goals = batch['home_goals']
        away_goals = batch['away_goals']

        home_prob, away_prob = self(expected_home, expected_away)
        loss_home = self._nll_loss(home_prob, home_goals)
        loss_away = self._nll_loss(away_prob, away_goals)
        loss = loss_home + loss_away + self.dummy_param.sum() * 0.0

        # Compute metrics for home and away predictions
        home_metrics = self._compute_metrics(home_prob, home_goals, prefix="val_home")
        away_metrics = self._compute_metrics(away_prob, away_goals, prefix="val_away")

        self.log("val_loss", loss, prog_bar=True)
        for name, value in {**home_metrics, **away_metrics}.items():
            self.log(name, value, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        expected_home = batch['expected_home']
        expected_away = batch['expected_away']
        home_goals = batch['home_goals']
        away_goals = batch['away_goals']

        home_prob, away_prob = self(expected_home, expected_away)
        loss_home = self._nll_loss(home_prob, home_goals)
        loss_away = self._nll_loss(away_prob, away_goals)
        loss = loss_home + loss_away + self.dummy_param.sum() * 0.0

        # Compute metrics for home and away predictions
        home_metrics = self._compute_metrics(home_prob, home_goals, prefix="test_home")
        away_metrics = self._compute_metrics(away_prob, away_goals, prefix="test_away")

        self.log("test_loss", loss, prog_bar=True)
        for name, value in {**home_metrics, **away_metrics}.items():
            self.log(name, value, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # configure_optimizers is provided by BaseModel


# -------------------------
# Main training and evaluation script
# -------------------------
def main():
    # Instantiate the DataModule and the LightningModule.
    batch_size = 256
    data_module = FootballDataModule(batch_size=batch_size)
    # Here we set max_goals=4 (so outcomes 0-4 are explicit and index 5 is the tail)
    model = PoissonLightningModule(max_goals=4, lr=1e-3)

    # Set up the trainer.
    trainer = pl.Trainer(
        max_epochs=5,
    )

    # Train the model.
    trainer.fit(model, datamodule=data_module)
    # Test the model.
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
