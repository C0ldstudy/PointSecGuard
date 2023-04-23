import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from ..attack import Attack

class tar_NB_attack(Attack):
    def __init__(self, model, eps=0.3, alpha=2/255, iters=40,target=None, mask=None):
        super(tar_NB_attack, self).__init__("tar_NB_attack", model)
        self.model = model
        self.eps=eps
        self.alpha=alpha
        self.iters=iters
        self.target = target
        self.mask = mask

    def forward(self, images, labels):
        color = images[:, 3:6][:,:,self.mask].clone().detach().to(self.device)
        ori_color = color.clone().detach().to(self.device)
        ori_image = images.clone().detach().to(self.device)
        adv_images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        target_labels = torch.full(labels.shape, self.target).to(self.device)

        for i in range(self.iters):
            color.requires_grad = True
            adv_images[:,3:6][:,:,self.mask] = color
            outputs = self.model(adv_images)

            pred = outputs.max(dim=1)[1]
            acc = pred.eq(labels.view_as(pred)).sum().item() / 4096

            target_acc = pred[:,self.mask].eq(target_labels[:,self.mask].view_as(pred[:,self.mask])).sum().item() / self.mask.sum().item()
            other_acc = pred[0, ~self.mask].eq(labels[0, ~self.mask].view_as(pred[0, ~self.mask])).sum().item() / (~self.mask).sum().item()
            if target_acc> 0.9:
                return adv_images
            self.model.zero_grad()
            cost = loss(outputs, target_labels).to(self.device)
            cost.backward(retain_graph=True)
            adv_images[:, 3:6][:,:,self.mask] = adv_images[:, 3:6][:,:,self.mask] - self.alpha*color.grad.sign()
            eta = torch.clamp(adv_images[:, 3:6][:,:,self.mask] - ori_color, min=-self.eps, max=self.eps)
            color = torch.clamp(ori_color + eta, min=0, max=1).detach_()
        adv_images[:,3:6][:,:,self.mask] = color
        return adv_images




class tar_NU_attack(Attack):
    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01, target=None, mask=None):
        super(tar_NU_attack, self).__init__("tar_NU_attack", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.target = target
        self.mask = mask

    def forward(self, images, labels):
        neighour = 5
        color = images[:, 3:6][:,:,self.mask].clone().detach().to(self.device)
        images = images.clone().detach().to(self.device)
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
            torch.cuda.empty_cache()

            # Get Adversarial Images
            color = self.tanh_space(w_color)
            adv_images = best_adv_images.clone().detach()
            adv_images[:,3:6][:,:,self.mask] = color
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.model(adv_images)
            if self.target == None:
                f_loss = self.non_f(outputs, labels)
            else:
                f_loss = self.tar_f(outputs, labels)


            Smooth_loss = self.smooth(adv_images, images, neighour).sum()

            cost = f_loss + 0.0001 * Smooth_loss + self.c * L2_loss
            prev_cost[step] = cost
            # test acc
            pred = outputs.max(dim=1)[1]
            acc = pred.eq(labels.view_as(pred)).sum().item() / 4096
            target_labels = torch.full(labels.shape, self.target).to(self.device)

            if self.target == None:
                target_acc = pred[:,self.mask].eq(labels[:,self.mask].view_as(pred[:,self.mask])).sum().item() / self.mask.sum().item()
            else:
                target_acc = pred[:,self.mask].eq(target_labels[:,self.mask].view_as(pred[:,self.mask])).sum().item() / self.mask.sum().item()
            other_acc = pred[0, ~self.mask].eq(labels[0, ~self.mask].view_as(pred[0, ~self.mask])).sum().item() / (~self.mask).sum().item()
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            best_adv_images = adv_images.clone().detach()

            color_dis_mask = torch.dist(images[:, 3:6][:,:,self.mask], best_adv_images[:, 3:6][:,:,self.mask], p=2)
            color_dis_nmask = torch.dist(images[:, 3:6][:,:, ~self.mask], best_adv_images[:, 3:6][:,:, ~self.mask], p=2)
            if self.target == None:
                if target_acc < 1 / 13:
                    return best_adv_images
            else:
                if target_acc > 0.9:
                    return best_adv_images
            print(target_acc)
            if (step > 0) & (step % 50 == 0):
                self.lr = self.lr/2
                optimizer = optim.Adam([w_color], lr=self.lr)

            if (step > 10)&(step % 10 == 0):
                if cost.item() >= prev_cost[step-10]:
                    best_adv_images[:, 3:6][:,:,self.mask] = best_adv_images[:, 3:6][:,:,self.mask] + torch.empty_like(best_adv_images[:, 3:6][:,:,self.mask]).uniform_(0, 1)
                    best_adv_images = torch.clamp(best_adv_images, min=0, max=1)
        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def non_f(self, outputs, labels):
        criterion = torch.nn.CrossEntropyLoss()
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
        one_hot_labels = one_hot_labels.transpose(1, 2)
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        i = i[0][self.mask]
        j = torch.masked_select(outputs, one_hot_labels.bool())[self.mask]
        return torch.clamp(self._targeted*(j-i), min=-self.kappa).sum() #+ loss


    def tar_f(self, outputs, labels):
        criterion = torch.nn.CrossEntropyLoss()
        target_labels = torch.full(labels.shape, self.target).to(self.device).type(torch.long)
        one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(self.device)
        one_hot_labels = one_hot_labels.transpose(1, 2)
        j, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = j[0][self.mask]
        i = torch.masked_select(outputs, one_hot_labels.bool())[self.mask]
        return torch.clamp(self._targeted * (j - i), min= -self.kappa).sum()

    def smooth(self, adv_images, images, neighbour):
        adv_pos = adv_images[0,3:6,:,0].transpose(1,0) # [4096,3]
        pos = adv_images[0,3:6,:, 0].transpose(1, 0)
        dist = torch.cdist(adv_pos, pos)  # [4096, 4096]
        sorted_dist, ind_dist = torch.sort(dist, dim=1)
        return sorted_dist[:,:neighbour]
