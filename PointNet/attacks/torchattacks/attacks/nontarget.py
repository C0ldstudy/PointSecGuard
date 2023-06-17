import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from ..attack import Attack



class NB_attack(Attack):
    def __init__(self, model, eps=0.3, alpha=2/255, iters=40):
        super(NB_attack, self).__init__("NB_attack", model)
        self.model = model
        self.eps=eps
        self.alpha=alpha
        self.iters=iters

    def forward(self, images, labels):
        color = images[:, 3:6].clone().detach().to(self.device)
        ori_color = color.clone().detach().to(self.device)
        ori_image = images.clone().detach().to(self.device)

        adv_images = images.clone().detach().to(self.device)
        labels = torch.tensor(labels[0], dtype=torch.int64).to(self.device)
        loss = nn.CrossEntropyLoss()

        for i in range(self.iters):
            color.requires_grad = True
            adv_images[:, 3:6] = color
            outputs,_ = self.model(adv_images)
            outputs = outputs[0]#.max(1)[1].float()

            self.model.zero_grad()
            cost = loss(outputs, labels).to(self.device)
            cost.backward(retain_graph=True)

            adv_images[:, 3:6] = adv_images[:, 3:6] + self.alpha*color.grad.sign()
            eta = torch.clamp(adv_images[:, 3:6] - ori_color, min=-self.eps, max=self.eps)
            color = torch.clamp(ori_color + eta, min=0, max=1).detach_()

        dis = torch.dist(adv_images, ori_image, p=2) #/ batch_size
        return adv_images

class NU_attack(Attack):
    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01):
        super(NU_attack, self).__init__("NU_attack", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def forward(self, images, labels):
        neighour = 10
        color = images[:, 3:6].clone().detach().to(self.device)
        images = images.clone().detach().to(self.device)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
        w_color = self.inverse_tanh_space(color).detach()
        w_color.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device) # 1e10
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        prev_cost = torch.full([self.steps], 1e10)
        optimizer = optim.Adam([w_color], lr=self.lr)


        for step in range(self.steps):
            # Get Adversarial Images
            color = self.tanh_space(w_color)
            adv_images = best_adv_images.clone().detach()
            adv_images[:, 3:6] = color
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs,_ = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()
            Smooth_loss = self.smooth(adv_images, images, neighour).sum()
            cost = f_loss

            cost = f_loss + self.c * Smooth_loss + self.c * L2_loss
            # test acc
            pred = outputs.max(dim=2)[1]
            acc = pred.eq(labels.view_as(pred)).sum().item() / 4096

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            best_adv_images = adv_images.clone().detach()
            # Early Stop when loss does not converge.
            if acc < 1/13:
                return best_adv_images
                if (step > 0) & (step % 100 == 0):
                    lr = lr/2
                    optimizer = optim.Adam([w_pertub], lr=lr)
                    # print("lr oacc cost: ", lr, acc, cost)
                if (step > 10)&(step % 10 == 0):
                    if cost.item() > prev_cost[step-10]:
                        # print("======bingo======")
                        images[:, 3:6][:,:,mask] = images[:, 3:6][:,:,mask] + torch.empty_like(images[:, 3:6][:,:,mask]).uniform_(0, 1)
                        images[:, 3:6] = torch.clamp(images[:, 3:6], min=0, max=1)
        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)
        # return self.atanh(x)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        outputs = nn.functional.softmax(outputs, dim=2) # important
        outputs = outputs.transpose(1, 2)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
        one_hot_labels = one_hot_labels.transpose(1, 2)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j, _ = torch.max(one_hot_labels * outputs, dim=1)
        return torch.clamp(self._targeted*(j-i), min=-self.kappa)

    def smooth(self, adv_images, images, neighbour):
        adv_pos = adv_images[0,3:6].transpose(1,0) # [4096,3]
        pos = images[0,3:6].transpose(1, 0)
        dist = torch.cdist(adv_pos, pos)  # [4096, 4096]
        sorted_dist, ind_dist = torch.sort(dist, dim=1)
        return sorted_dist[:,:neighbour]
