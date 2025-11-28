import lightning.pytorch as lit
from torch.utils.data import DataLoader
from datasets import load_dataset

def encode(example, col : str, num_embeddings: int):
    example[col] = [hash(x) % num_embeddings for x in example[col]]
    return example

# Data Loading Module
class FootballDataModule(lit.LightningDataModule):
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