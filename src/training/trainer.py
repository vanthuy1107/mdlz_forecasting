"""OOP Trainer class for model training and evaluation."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from pathlib import Path


class Trainer:
    """Robust trainer class for training and evaluating models."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_interval: int = 5,
        save_dir: Optional[str] = None
    ):
        """
        Initialize Trainer.
        
        Args:
            model: PyTorch model to train.
            criterion: Loss function.
            optimizer: Optimizer for training.
            device: Device to run training on (cuda/cpu).
            scheduler: Optional learning rate scheduler.
            log_interval: Log metrics every N epochs.
            save_dir: Directory to save checkpoints.
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.current_epoch: int = 0
        
        # Best model tracking
        self.best_val_loss: float = float('inf')
        self.best_model_state: Optional[Dict[str, Any]] = None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
        
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        batch_losses = []
        
        for inputs, cat, labels in train_loader:
            # Move to device
            inputs = inputs.to(self.device)
            cat = cat.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs, cat)
            # Ensure labels match outputs shape
            if labels.ndim == 0:
                labels = labels.unsqueeze(-1)
            elif labels.ndim == 1:
                labels = labels.unsqueeze(-1) if outputs.ndim > 1 else labels
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            batch_losses.append(loss.item())
        
        avg_loss = sum(batch_losses) / len(batch_losses)
        return avg_loss
    
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation.
            return_predictions: Whether to return predictions and labels.
        
        Returns:
            Tuple of (avg_loss, predictions, labels) if return_predictions=True,
            otherwise (avg_loss, None, None).
        """
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, cat, labels in dataloader:
                # Move to device
                inputs = inputs.to(self.device)
                cat = cat.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, cat)
                # Ensure labels match outputs shape
                if labels.ndim == 0:
                    labels = labels.unsqueeze(-1)
                elif labels.ndim == 1:
                    labels = labels.unsqueeze(-1) if outputs.ndim > 1 else labels
                loss = self.criterion(outputs, labels)
                
                losses.append(loss.item())
                
                if return_predictions:
                    all_preds.append(outputs.cpu())
                    all_labels.append(labels.cpu())
        
        avg_loss = sum(losses) / len(losses)
        
        if return_predictions:
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            return avg_loss, all_labels, all_preds
        
        return avg_loss, None, None
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_best: bool = True,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            save_best: Whether to save best model based on validation loss.
            verbose: Whether to print training progress.
            config: Configuration dictionary to include in checkpoint (optional).
        
        Returns:
            Tuple of (train_losses, val_losses) lists.
        """
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, _, _ = self.evaluate(val_loader, return_predictions=False)
            self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save best model
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                
                if self.save_dir:
                    self.save_checkpoint(epoch, is_best=True, config=config)
            
            # Logging
            if verbose and self.current_epoch % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"EPOCH {self.current_epoch}/{epochs}: "
                    f"Train loss: {train_loss:.4f} | "
                    f"Val loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f}"
                )
        
        # Load best model state
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"\nBest validation loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def predict(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on a dataset.
        
        Args:
            dataloader: Data loader for prediction.
        
        Returns:
            Tuple of (predictions, labels) arrays.
        """
        _, labels, predictions = self.evaluate(dataloader, return_predictions=True)
        return labels, predictions
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model.
            filename: Custom filename (optional).
            config: Configuration dictionary to include in checkpoint (optional).
        """
        if self.save_dir is None:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = "checkpoint.pth" if not is_best else "best_model.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Include config if provided
        if config is not None:
            checkpoint['config'] = config
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file.
            load_optimizer: Whether to load optimizer state.
            load_scheduler: Whether to load scheduler state.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_scheduler and 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        self.current_epoch = checkpoint.get('epoch', 0)

