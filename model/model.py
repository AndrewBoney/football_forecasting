import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_dataset
from torchmetrics.functional import accuracy, precision, recall, f1_score
# -------------------------
# Define the Poisson module
# -------------------------
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

# -------------------------
# Define the Lightning Module
# -------------------------
class PoissonLightningModule(pl.LightningModule):
    def __init__(self, max_goals: int = 10, lr: float = 1e-3):
        """
        LightningModule that uses Poisson probabilities to compute a negative log likelihood loss
        and evaluates standard classification metrics.

        Args:
            max_goals (int): Maximum number of goals to consider before using the tail bin.
            lr (float): Learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.poisson_module = PoissonProbabilityModule(max_goals=self.hparams.max_goals)
        # Dummy parameter to ensure the model has a trainable parameter.
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

    def _nll_loss(self, prob_dist: torch.Tensor, targets: torch.Tensor):
        """
        Computes the negative log likelihood loss for a batch.
        Targets greater than max_goals are assigned to the extra tail bin.

        Args:
            prob_dist (Tensor): shape (batch_size, max_goals+2)
            targets (Tensor): shape (batch_size,) with integer goal counts.

        Returns:
            Tensor: scalar loss value.
        """
        tail_index = self.poisson_module.max_goals + 1
        targets_adjusted = torch.where(
            targets > self.poisson_module.max_goals,
            torch.full_like(targets, tail_index),
            targets
        )
        batch_idx = torch.arange(prob_dist.size(0), device=prob_dist.device)
        eps = 1e-8  # to avoid log(0)
        nll = -torch.log(prob_dist[batch_idx, targets_adjusted] + eps)
        return nll.mean()

    def _compute_metrics(self, probs: torch.Tensor, targets: torch.Tensor, prefix: str):
        """
        Computes classification metrics given probability distributions and targets.
        The prediction is taken as the argmax of the probabilities.
        Targets above max_goals are mapped to the extra tail bin.

        Args:
            probs (Tensor): shape (batch_size, max_goals+2)
            targets (Tensor): shape (batch_size,)
            prefix (str): prefix for metric names (e.g., "train", "val", "test").

        Returns:
            dict: a dictionary of metric names and their values.
        """
        tail_index = self.poisson_module.max_goals + 1
        # Adjust targets: any goal count above max_goals goes to tail index.
        targets_adjusted = torch.where(
            targets > self.poisson_module.max_goals,
            torch.full_like(targets, tail_index),
            targets
        )
        preds = torch.argmax(probs, dim=1)
        metrics = {}
        metrics[f"{prefix}_acc"] = accuracy(
            preds, targets_adjusted, task="multiclass", num_classes=self.poisson_module.max_goals+2
        )
        metrics[f"{prefix}_precision"] = precision(
            preds, targets_adjusted, task="multiclass", num_classes=self.poisson_module.max_goals+2, average="macro"
        )
        metrics[f"{prefix}_recall"] = recall(
            preds, targets_adjusted, task="multiclass", num_classes=self.poisson_module.max_goals+2, average="macro"
        )
        metrics[f"{prefix}_f1"] = f1_score(
            preds, targets_adjusted, task="multiclass", num_classes=self.poisson_module.max_goals+2, average="macro"
        )
        return metrics

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

# -------------------------
# Define the Data Module
# -------------------------
class FootballDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        """
        LightningDataModule for the AndyB/football_fixtures dataset.
        
        Args:
            batch_size (int): How many samples per batch.
        """
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # Download the dataset (only executed on 1 process)
        load_dataset("AndyB/football_fixtures")

    def setup(self, stage=None):
        # Load the dataset and perform per-example processing.
        self.dataset = load_dataset("AndyB/football_fixtures")

        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["val"]
        self.test_dataset = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

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
