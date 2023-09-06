from abc import ABC
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from components.utils.model import ChatNet
from components.utils.dataset import ChatDataset

from sklearn.model_selection import train_test_split

from components.config import DLTrainingConfig
from components.utils.helpers import DataCollection, build_loss_fn
import consts.dl_model_training

class Train(ABC):
    """Base class for all trainings."""
    
    def __init__(self, cfg: DLTrainingConfig) -> None:
        """Initialize training."""
        self.cfg = cfg
        self.data_collection: DataCollection = self._set_data_collection()
    
    def _create_data(
        self,
        inputs: List[Path],
        train: bool
    ) -> Tuple(DataLoader, Dataset):
        """Create and return DataLoader and Dataset from given data.
        This method constructs and retrund DataLoader and Dataset from provided data.

        Args:
            inputs (List[Path]): An input intents paths.
            train (bool): Indicator if the DataLoader is used for training.

        Returns:
            Tuple[DataLoader, Dataset]: The DataLoader and Dataset objects used in data collection.
        """
        dataset = ChatDataset(inputs=inputs, cfg=self.cfg)
        data_loader = DataLoader(dataset, batch_size=self.cfg.batch_size, drop_last=True, shuffle=train)
        
        return (data_loader, dataset)
    
    def _set_data_collection(self) -> DataCollection:
        """Create and return the DataCollection object for training, validation and testing.
        This method constructs and organizes the datasets and data lodaers for the experiment.

        Returns:
            DataCollection: An object containing train, validation and test data loaders and related informations.
        """
        intents = list(Path(self.cfg.data_dir))
        intents.sort()
        
        _, full_dataset = self._create_data(inputs=intents, train=False)
        
        train_intents, val_intents = train_test_split(
            intents,
            test_size=consts.dl_model_training.VAL_DATASET_SIZE,
            random_state=consts.dl_model_training.RANDOM_STATE
        )
        
        train_loader, _ = self._create_data(inputs=train_intents, train=False)
        val_loader, _ = self._create_data(inputs=val_intents, train=False)
        
        data_collection = DataCollection(
            train_loader=train_loader,
            val_loader=val_loader,
            num_categories=full_dataset.get_categories,
            num_inputs=len(full_dataset)
        )
        
        return data_collection
    
    @property
    def model(self) -> nn.Module:
        """Creates and returns deep learning model using specified arguments.
        
        Returns:
            nn.Module: Deep learning model used for the training.
        """
        return ChatNet(
            input_size=self.data_collection.num_inputs,
            hidden_size=self.cfg.hidden_layers,
            output_size=self.data_collection.num_categories
            )
        
    @property
    def optimizer(self) -> torch.optim:
        """Creates and returns deep learning optimizer using specified arguments.
        
        Returns:
            torch.optim: Deep learning optimizer used for the training.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
    
    def run_training(self) -> None:
        """Run the training process.
        This method initiates the training process
        """
        for epoch in range(self.cfg.max_epochs):
            model = self.model.to(self.cfg.device)
            for (words, labels) in self.data_collection.train_loader:
                words = words.to(self.cfg.device)
                labels = labels.to(self.cfg.device).long()
                
                #forward
                outputs = model(words)
                loss = build_loss_fn(name=self.cfg.loss)(outputs, labels)
                
                #backward and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if (epoch+1) % 50 == 0:
                print(f"epoch {epoch+1}/{self.cfg.max_epochs}, loss = {loss.item():.4f}")
        
        torch.save(model, "data.pth")
        print("Training completed")     
