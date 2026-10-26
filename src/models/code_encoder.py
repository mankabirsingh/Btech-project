"""
CodeBERT-based Vulnerability Detection Model
=============================================
Fine-tuning CodeBERT for binary vulnerability classification.
Supports MPS (Apple Silicon), CUDA, and CPU backends.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class CodeDataset(Dataset):
    """PyTorch dataset for code vulnerability detection."""
    
    def __init__(self, codes: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        code = str(self.codes[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }


class CodeBERTClassifier(nn.Module):
    """
    CodeBERT-based binary classifier for vulnerability detection.
    
    Architecture:
    - CodeBERT encoder (frozen or fine-tuned)
    - Dropout
    - Linear classification head
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/codebert-base',
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class ProvenanceAwareClassifier(nn.Module):
    """
    Provenance-Aware Vulnerability Classifier.
    
    Combines CodeBERT code representation with provenance features:
    - AI involvement level
    - Provenance type (one-hot)
    - AI generator (embedding)
    - Edit count
    - Code length features
    
    Architecture:
    - Code encoder (CodeBERT [CLS] -> projection)
    - Provenance feature encoder (MLP)
    - Fusion (concatenation + MLP)
    - Classification head
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/codebert-base',
        n_provenance_features: int = 10,
        hidden_dim: int = 128,
        num_labels: int = 2,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel
        
        self.encoder = AutoModel.from_pretrained(model_name)
        code_hidden = self.encoder.config.hidden_size  # 768
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Code projection
        self.code_proj = nn.Sequential(
            nn.Linear(code_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Provenance feature encoder
        self.prov_encoder = nn.Sequential(
            nn.Linear(n_provenance_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Fusion + classification
        fusion_dim = hidden_dim + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
    
    def forward(self, input_ids, attention_mask, prov_features):
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            prov_features: Provenance features [batch, n_features]
        """
        # Encode code
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        code_repr = self.code_proj(cls_output)
        
        # Encode provenance
        prov_repr = self.prov_encoder(prov_features)
        
        # Fuse and classify
        fused = torch.cat([code_repr, prov_repr], dim=-1)
        logits = self.fusion(fused)
        
        return logits


def extract_provenance_features(df, feature_columns=None):
    """
    Extract numerical provenance features from a DataFrame.
    
    Features:
    - ai_involvement (float, 0-1)
    - provenance_type one-hot (human, ai, mixed, unknown)
    - code_length (normalized)
    - n_lines (normalized)
    - has_ai_generator (binary)
    - has_edit_history (binary)
    
    Returns:
        numpy array of shape [n_samples, n_features]
    """
    features = []
    
    # AI involvement
    ai_inv = df['ai_involvement'].fillna(0.5).values  # 0.5 = unknown
    features.append(ai_inv.reshape(-1, 1))
    
    # Provenance type one-hot
    prov_types = ['human', 'ai', 'mixed', 'unknown']
    for pt in prov_types:
        col = (df['provenance_type'] == pt).astype(float).values
        features.append(col.reshape(-1, 1))
    
    # Code length (log-normalized)
    code_lens = df['code'].str.len().values.astype(float)
    code_lens = np.log1p(code_lens)
    code_lens = (code_lens - code_lens.mean()) / (code_lens.std() + 1e-8)
    features.append(code_lens.reshape(-1, 1))
    
    # Number of lines (log-normalized)
    n_lines = df['code'].str.count('\n').values.astype(float) + 1
    n_lines = np.log1p(n_lines)
    n_lines = (n_lines - n_lines.mean()) / (n_lines.std() + 1e-8)
    features.append(n_lines.reshape(-1, 1))
    
    # Has AI generator info
    has_gen = df['ai_generator'].notna().astype(float).values
    features.append(has_gen.reshape(-1, 1))
    
    # Has edit history
    has_hist = df['human_edit_history'].notna().astype(float).values
    features.append(has_hist.reshape(-1, 1))
    
    # Comment density (proxy for code style / documentation habits)
    comment_chars = df['code'].str.count(r'(//|/\*|#)').values.astype(float)
    comment_density = comment_chars / (df['code'].str.len().values.astype(float) + 1.0)
    comment_density = (comment_density - comment_density.mean()) / (comment_density.std() + 1e-8)
    features.append(comment_density.reshape(-1, 1))
    
    result = np.hstack(features)  # [n_samples, 10]
    logger.info(f"Extracted {result.shape[1]} legitimate provenance features (zero label leakage) for {result.shape[0]} samples")
    
    return result.astype(np.float32)


def train_codebert(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs: int = 5,
    lr: float = 2e-5,
    checkpoint_dir: Optional[Path] = None,
    model_name: str = 'codebert',
) -> Dict:
    """
    Train CodeBERT or provenance-aware model.
    
    Returns:
        Dict with training history and best metrics.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            if 'prov_features' in batch:
                prov_features = batch['prov_features'].to(device)
                logits = model(input_ids, attention_mask, prov_features)
            else:
                logits = model(input_ids, attention_mask)
            
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                if 'prov_features' in batch:
                    prov_features = batch['prov_features'].to(device)
                    logits = model(input_ids, attention_mask, prov_features)
                else:
                    logits = model(input_ids, attention_mask)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{n_epochs}: "
                     f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                     f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc and checkpoint_dir:
            best_val_acc = val_acc
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss,
            }, checkpoint_path)
            logger.info(f"Best model saved: val_acc={val_acc:.4f}")
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'n_epochs': n_epochs,
        'lr': lr,
    }
