import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxDetDistiller3D(nn.Module):
    """
    Projects VoxDet's 3D classification features into the CLIP/LSeg embedding space (512-dim).

    Takes the cls_feat volume from VoxDet's SpatiallyDecoupledFPN and maps it to
    a 512-dim embedding per voxel, aligned with LSeg pixel features.

    This is analogous to OVO's Distiller3D (phi_3D) but adapted for:
    - VoxDet's channel dimensions (typically 128 from the FPN)
    - The decoupled cls branch output rather than MonoScene's BEV features
    """

    def __init__(self, in_channels=128, embedding_dim=512, mid_channels=256):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(
            mid_channels, embedding_dim, kernel_size=1, stride=1, padding=0
        )
        self.bn2 = nn.BatchNorm3d(embedding_dim)
        self.proj = nn.Conv3d(
            embedding_dim, embedding_dim, kernel_size=1, stride=1, padding=0
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cls_feat):
        """
        Args:
            cls_feat: [B, C, X, Y, Z] classification feature volume from VoxDet SVE
        Returns:
            aligned_feat: [B, 512, X, Y, Z] CLIP-aligned voxel embeddings
        """
        x = self.relu(self.bn1(self.conv1(cls_feat)))  # [B, 256, X, Y, Z]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 512, X, Y, Z]
        x = self.proj(x)  # [B, 512, X, Y, Z]
        return x


class VoxDetDistiller2D(nn.Module):
    """
    Projects VoxDet's 2D image features into the LSeg embedding space (512-dim).

    This is the 2D alignment regularizer (analogous to OVO's Distiller2D / phi_2D).
    Takes multi-scale 2D features from VoxDet's image encoder and maps them
    to 512-dim features aligned with LSeg.
    """

    def __init__(self, in_channels=128, embedding_dim=512):
        super().__init__()
        # VoxDet's image encoder outputs features after FPN, typically single-scale
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_feat):
        """
        Args:
            img_feat: [B*N, C, H, W] 2D image features from VoxDet backbone+FPN
        Returns:
            aligned_feat: [B*N, 512, H, W] LSeg-aligned 2D features
        """
        x = self.relu(self.bn1(self.conv1(img_feat)))
        x = self.conv2(x)
        return x


class TextEmbeddingClassifier(nn.Module):
    """
    Replaces VoxDet's learned classification head with CLIP text embeddings.

    Instead of a Conv3d that maps features -> num_classes logits,
    this module computes cosine similarity between voxel embeddings
    and CLIP text embeddings for each class.

    This is the key module that enables open-vocabulary inference:
    at test time, you can provide text embeddings for *any* set of classes.
    """

    def __init__(self, embedding_dim=512, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.text_embeddings = None  # Set during training/inference

    def set_text_embeddings(self, text_embeddings):
        """
        Args:
            text_embeddings: [num_classes, 512] CLIP text embeddings for each class
        """
        # Normalize text embeddings
        self.text_embeddings = F.normalize(text_embeddings, dim=-1)

    def forward(self, voxel_embeddings):
        """
        Args:
            voxel_embeddings: [B, 512, X, Y, Z] CLIP-aligned voxel features
        Returns:
            logits: [B, num_classes, X, Y, Z] classification logits
        """
        assert self.text_embeddings is not None, "Must call set_text_embeddings() first"

        B, C, X, Y, Z = voxel_embeddings.shape
        num_classes = self.text_embeddings.shape[0]

        # Normalize voxel features
        voxel_norm = F.normalize(voxel_embeddings, dim=1)  # [B, 512, X, Y, Z]

        # Reshape for matrix multiplication
        voxel_flat = voxel_norm.view(B, C, -1).permute(0, 2, 1)  # [B, X*Y*Z, 512]
        text_emb = self.text_embeddings.to(voxel_flat.device)  # [num_classes, 512]

        # Cosine similarity
        logits = (
            torch.matmul(voxel_flat, text_emb.T) / self.temperature
        )  # [B, X*Y*Z, num_classes]
        logits = logits.permute(0, 2, 1).view(
            B, num_classes, X, Y, Z
        )  # [B, num_classes, X, Y, Z]

        return logits
