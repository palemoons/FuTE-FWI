"""Loss functions for VelocityGAN"""
import torch
from torch import nn, autograd

class UnionLoss(nn.Module):
    """Union loss function for generator of VelocityGAN"""
    def __init__(self, lambda_g1v, lambda_g2v):
        super(UnionLoss, self).__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, output, label, discriminator=None):
        loss = self.l1_loss(output, label) * self.lambda_g1v + self.l2_loss(output, label) * self.lambda_g2v
        if discriminator is not None:
            return loss - torch.mean(discriminator(output))  # regularization
        return loss

class Wasserstein_GP(nn.Module):
    """Loss function for discriminator of VelocityGAN"""

    def __init__(self, device, lambda_gp):
        super(Wasserstein_GP, self).__init__()
        self.device = device
        self.lambda_gp = lambda_gp

    def forward(self, real, fake, model):
        gradient_penalty = self.compute_gradient_penalty(model, real, fake)
        loss_real = torch.mean(model(real))
        loss_fake = torch.mean(model(fake))
        loss = -loss_real + loss_fake + gradient_penalty * self.lambda_gp
        return loss, loss_real - loss_fake, gradient_penalty

    def compute_gradient_penalty(self, model, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = model(interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(real_samples.size(0), d_interpolates.size(1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty