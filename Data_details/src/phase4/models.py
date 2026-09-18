"""
Phase 4 Model Architectures: Causal Motion Intelligence Benchmark Suite.

Includes:
1. Model A: Linear / Ridge Baseline
2. Model B: Causal 1D-CNN
3. Model C: Causal GRU
4. Model D: Causal LSTM
5. Model E: Causal Dilated Temporal Convolutional Network (TCN)
6. Model F: Heteroscedastic Uncertainty Model (predicting speed & log-variance)
7. Model G: Multitask Motion Model (predicting speed & 5-state motion classification)

All models are strictly causal: receptive fields never access future timesteps (t > t_k).
"""

import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D Convolution with strict causal left-padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, C, T)
        returns: shape (B, C_out, T) with strictly causal alignment
        """
        x_pad = F.pad(x, (self.pad, 0))  # Pad left only
        return self.conv(x_pad)


class LinearBaseline(nn.Module):
    """Model A: Linear / Ridge Regression Baseline."""

    def __init__(self, in_features: int = 12, window_length: int = 30):
        super().__init__()
        self.in_features = in_features
        self.window_length = window_length
        self.fc = nn.Linear(in_features * window_length, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C) -> flatten to (B, T*C)
        b = x.size(0)
        x_flat = x.reshape(b, -1)
        speed = self.fc(x_flat)
        return {"speed": F.relu(speed)}  # Forward speed is strictly non-negative


class Causal1DCNN(nn.Module):
    """Model B: Causal 1D Convolutional Neural Network."""

    def __init__(
        self,
        in_channels: int = 12,
        channels: Tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.extend([
                CausalConv1d(c_in, c_out, kernel_size=kernel_size),
                nn.BatchNorm1d(c_out),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            c_in = c_out
            
        self.feature_net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C) -> transpose to (B, C, T)
        x_trans = x.transpose(1, 2)
        feats = self.feature_net(x_trans)
        speed = self.head(feats)
        return {"speed": F.relu(speed)}


class CausalGRU(nn.Module):
    """Model C: Causal Unidirectional Gated Recurrent Unit."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_size: int = 48,
        num_layers: int = 2,
        dropout: float = 0.1,
        in_features: Optional[int] = None,
    ):
        super().__init__()
        num_in = in_features if in_features is not None else in_channels
        self.gru = nn.GRU(
            input_size=num_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # STRICTLY CAUSAL
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C)
        out, _ = self.gru(x)
        # Extract last causal timestep output (B, H)
        h_last = out[:, -1, :]
        speed = self.head(h_last)
        return {"speed": F.relu(speed)}


class CausalLSTM(nn.Module):
    """Model D: Causal Unidirectional Long Short-Term Memory."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_size: int = 48,
        num_layers: int = 2,
        dropout: float = 0.1,
        in_features: Optional[int] = None,
    ):
        super().__init__()
        num_in = in_features if in_features is not None else in_channels
        self.lstm = nn.LSTM(
            input_size=num_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # STRICTLY CAUSAL
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        speed = self.head(h_last)
        return {"speed": F.relu(speed)}


class TCNResidualBlock(nn.Module):
    """Dilated Causal Convolutional Residual Block for TCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return out + res


class CausalTCN(nn.Module):
    """Model E: Causal Dilated Temporal Convolutional Network."""

    def __init__(
        self,
        in_channels: int = 12,
        num_channels: Tuple[int, ...] = (32, 48, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(num_channels):
            dilation = 2 ** i  # Dilations: 1, 2, 4, 8 -> Receptive field = 31 samples >= 30
            layers.append(
                TCNResidualBlock(c_in, c_out, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            )
            c_in = c_out
            
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C) -> (B, C, T)
        x_trans = x.transpose(1, 2)
        y = self.network(x_trans)
        # Select last causal timestep: y[:, :, -1] (shape B, C_final)
        h_last = y[:, :, -1]
        speed = self.head(h_last)
        return {"speed": F.relu(speed)}


class HeteroscedasticSpeedModel(nn.Module):
    """Model F: Predicts both forward speed mean (mu) and log-variance (log sigma^2)."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: Tuple[int, ...] = (32, 48, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(hidden_channels):
            layers.append(
                TCNResidualBlock(c_in, c_out, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            )
            c_in = c_out
            
        self.backbone = nn.Sequential(*layers)
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_channels[-1], 32),
            nn.LeakyReLU(0.1),
        )
        self.mu_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_trans = x.transpose(1, 2)
        feats = self.backbone(x_trans)[:, :, -1]
        shared = self.fc_shared(feats)
        mu = F.relu(self.mu_head(shared))
        log_var = torch.clamp(self.logvar_head(shared), min=-6.0, max=4.0)  # Bound variance
        std = torch.exp(0.5 * log_var)
        return {
            "speed": mu,
            "log_var": log_var,
            "std": std,
        }


class MultitaskMotionModel(nn.Module):
    """Model G: Dual-Head Model predicting forward speed and 5-class motion state."""

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        hidden_channels: Tuple[int, ...] = (32, 48, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(hidden_channels):
            layers.append(
                TCNResidualBlock(c_in, c_out, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            )
            c_in = c_out
            
        self.backbone = nn.Sequential(*layers)
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_channels[-1], 32),
            nn.LeakyReLU(0.1),
        )
        self.speed_head = nn.Linear(32, 1)
        self.state_head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_trans = x.transpose(1, 2)
        feats = self.backbone(x_trans)[:, :, -1]
        shared = self.fc_shared(feats)
        speed = F.relu(self.speed_head(shared))
        state_logits = self.state_head(shared)
        return {
            "speed": speed,
            "state_logits": state_logits,
            "state_probs": F.softmax(state_logits, dim=-1),
        }


def get_model(model_name: str, in_channels: int = 12, window_length: int = 30) -> nn.Module:
    """Factory helper to instantiate any benchmark model by name."""
    name = model_name.lower()
    if name in ["linear", "ridge"]:
        return LinearBaseline(in_features=in_channels, window_length=window_length)
    elif name in ["cnn", "cnn1d"]:
        return Causal1DCNN(in_channels=in_channels)
    elif name == "gru":
        return CausalGRU(in_features=in_channels)
    elif name == "lstm":
        return CausalLSTM(in_features=in_channels)
    elif name == "tcn":
        return CausalTCN(in_channels=in_channels)
    elif name == "uncertainty":
        return HeteroscedasticSpeedModel(in_channels=in_channels)
    elif name == "multitask":
        return MultitaskMotionModel(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Supported: linear, cnn1d, gru, lstm, tcn, uncertainty, multitask")
