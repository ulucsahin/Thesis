# Edited from Akanimax code https://github.com/akanimax/T2F
"""
This file also has some unused methods that I tried and may again use later.
"""

import torch as th
import numpy as np
from torch.nn.functional import upsample, softplus
from torch.nn import ModuleList, Upsample, Conv2d, AvgPool2d

class ConditionalGANLoss:
    """ Base class for all losses """

    def __init__(self, device, dis):
        self.device = device
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, latent_vector, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, latent_vector, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


class CondWGAN_GP(ConditionalGANLoss):

    def __init__(self, device, dis, drift=0.001, use_gp=False):
        super().__init__(device, dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, latent_vector,
                           height, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param latent_vector: used for conditional loss calculation
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        from torch.autograd import grad

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(self.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)

        # forward pass
        op, logits = self.dis.forward(merged, latent_vector, height, alpha)

        # obtain gradient of op wrt. merged
        gradient = grad(outputs=op, inputs=merged, create_graph=True, grad_outputs=th.ones_like(op), retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        #print("losses.py penalty:", penalty)
        return penalty

    def dis_loss(self, real_samps, fake_samps, latent_vector, height, alpha):
        # define the (Wasserstein) loss

        fake_out, _ = self.dis(fake_samps, latent_vector, height, alpha)
        real_out, _ = self.dis(real_samps, latent_vector, height, alpha)
        #print(fake_out)

        loss = (th.mean(fake_out) - th.mean(real_out) + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            fake_samps.requires_grad = True  # turn on gradients for penalty calculation
            gp = self.__gradient_penalty(real_samps, fake_samps, latent_vector, height, alpha)
            loss += gp

        return loss

    def dis_loss_uluc_no_mad(self, real_samps, fake_samps, latent_vector, height, alpha):
        real_out, _ = self.dis(real_samps, latent_vector, height, alpha)
        real_out = softplus(-real_out).mean()
        real_out.backward(retain_graph=True)

        grad_real = th.autograd.grad(outputs=real_out.sum(), inputs=real_samps, create_graph=True)[0]  # outputs takes loss, inputs takes images
        grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty_real = 10 / 2 * grad_penalty_real
        grad_penalty_real.backward()

        fake_out, _ = self.dis(fake_samps, latent_vector, height, alpha)
        fake_out = softplus(fake_out).mean()
        fake_out.backward()

        loss = real_out + fake_out

        return loss

    def dis_loss_uluc(self, real_samps, fake_samps, latent_vector, mis_match_embed, height, alpha):
        # define the (Wasserstein) loss

        real_out, _ = self.dis(real_samps, latent_vector, height, alpha)
        real_out = softplus(-real_out).mean()
        real_out.backward(retain_graph=True)

        grad_real = th.autograd.grad(outputs=real_out.sum(), inputs=real_samps, create_graph=True)[0]  # outputs takes loss, inputs takes images
        grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty_real = 10 / 2 * grad_penalty_real
        grad_penalty_real.backward()

        fake_out, _ = self.dis(fake_samps, latent_vector, height, alpha)
        fake_out = softplus(fake_out).mean()
        fake_out.backward()

        mis_match_out, _ = self.dis(real_samps, mis_match_embed, height, alpha)
        mis_match_out = softplus(mis_match_out).mean()
        mis_match_out.backward()

        #loss = (th.mean(mis_match_out) + th.mean(fake_out) - th.mean(real_out) + (self.drift * th.mean(real_out ** 2)))

        # if self.use_gp:
        #     # calculate the WGAN-GP (gradient penalty)
        #     fake_samps.requires_grad = True  # turn on gradients for penalty calculation
        #     gp = self.__gradient_penalty(real_samps, fake_samps, latent_vector, height, alpha)
        #     loss += gp

        loss = real_out + fake_out + mis_match_out

        return loss

    def dis_loss_uluc_2(self, real_samps, fake_samps, latent_vector, mis_match_embed, height, alpha):
        # define the (Wasserstein) loss

        real_out, _ = self.dis(real_samps, latent_vector, height, alpha)
        real_out = softplus(-real_out).mean()
        real_out.backward(retain_graph=True)

        grad_real = th.autograd.grad(outputs=real_out.sum(), inputs=real_samps, create_graph=True)[0]  # outputs takes loss, inputs takes images
        grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty_real = 10 / 2 * grad_penalty_real
        grad_penalty_real.backward()

        fake_out, _ = self.dis(fake_samps, latent_vector, height, alpha)
        fake_out = softplus(fake_out).mean()
        fake_out.backward()

        mis_match_out, _ = self.dis(real_samps, mis_match_embed, height, alpha)
        mis_match_out = softplus(mis_match_out).mean()
        mis_match_out.backward()

        loss = real_out + fake_out + mis_match_out + grad_real

        return loss

    def gen_loss_uluc(self, _, fake_samps, latent_vector, height, alpha, retain_graph=False):
        # calculate the WGAN loss for generator
        dis_output, _ = self.dis(fake_samps, latent_vector, height, alpha)
        dis_output = softplus(-dis_output).mean()
        dis_output.backward(retain_graph=retain_graph)
        # print("dis_output, loss", dis_output, loss)

        return dis_output

    def gen_loss_uluc_2(self, _, fake_samps, latent_vector, mismatched_latent_vector, height, alpha, retain_graph=False):
        # calculate the WGAN loss for generator

        dis_output, _ = self.dis(fake_samps, latent_vector, height, alpha)
        dis_output = softplus(-dis_output).mean()
        dis_output.backward(retain_graph=True)

        mismatch_dis_output, _ = self.dis(fake_samps, mismatched_latent_vector, height, alpha)
        mismatch_dis_output = softplus(-mismatch_dis_output).mean()
        mismatch_dis_output.backward(retain_graph=False)

        g_loss = (dis_output + mismatch_dis_output)
        # g_loss.backward(retain_graph=retain_graph)

        return g_loss

    def gen_loss(self, _, fake_samps, latent_vector, height, alpha):
        # calculate the WGAN loss for generator
        dis_output, _ = self.dis(fake_samps, latent_vector, height, alpha)

        loss = -th.mean(dis_output)
        #print("dis_output, loss", dis_output, loss)

        return loss
