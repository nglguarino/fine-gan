import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# GAN LOSS FUNCTIONS
# ============================================

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        from torchvision.models import VGG19_Weights

        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        # Extract specific layers
        self.layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18], # relu3_4
        ])

        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False

        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Normalize from [-1, 1] to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        # VGG normalization
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0
        x_pred, x_target = pred, target

        for layer in self.layers:
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            loss += F.l1_loss(x_pred, x_target)

        return loss

def hinge_loss_d(real_pred, fake_pred):
    """Hinge loss for discriminator"""
    loss = 0
    for real_p, fake_p in zip(real_pred, fake_pred):
        loss += torch.mean(F.relu(1 - real_p)) + torch.mean(F.relu(1 + fake_p))
    return loss / len(real_pred)

def hinge_loss_g(fake_pred):
    """Hinge loss for generator"""
    loss = 0
    for fake_p in fake_pred:
        loss += -torch.mean(fake_p)
    return loss / len(fake_pred)