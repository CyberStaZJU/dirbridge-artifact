import math

import torch
try:
    import torch_npu
except ImportError:
    pass
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from utils.state_dict_ops import load_param_dict_, model_param_dict, sd_lerp


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class Local(object):
    def __init__(
        self,
        args,
        dataset=None,
        idxs=None,
        iters=None,
        nums=None,
        rho=0,
    ):
        self.args = args
        self.iters = iters
        self.nums = nums
        self.rho = rho
        self.loss_func = nn.CrossEntropyLoss()

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=args.local_bs,
            shuffle=True,
            num_workers=args.num_workers,
        )
        local_epochs = getattr(args, 'local_epochs', None)
        if local_epochs is not None and float(local_epochs) > 0:
            self.iters = max(1, int(math.ceil(len(self.ldr_train) * float(local_epochs))))

    @property
    def device(self):
        return self.args.device

    def _named_trainable_params(self, net):
        for name, param in net.named_parameters():
            if param.requires_grad:
                yield name, param

    def _init_reference_params(self, net):
        if self.rho <= 0:
            return None
        init_params = {}
        for name, param in self._named_trainable_params(net):
            init_params[name] = param.data.clone().detach()
        return init_params

    def _process_batch(self, batch):
        images, labels = batch
        return {
            'images': images.to(self.device),
            'labels': labels.to(self.device),
        }

    def _forward_pass(self, net, inputs):
        logits = net(inputs['images'])
        loss = self.loss_func(logits, inputs['labels'])
        return type('Output', (object,), {'loss': loss, 'logits': logits})()

    def _compute_loss_and_backward(self, net, outputs, init_params=None):
        loss = outputs.loss
        if self.rho > 0 and init_params is not None:
            reg_loss = 0.0
            for name, param in self._named_trainable_params(net):
                diff = param - init_params[name]
                reg_loss += torch.sum(diff * diff)
            total_loss = loss + 0.5 * self.rho * reg_loss
            total_loss.backward()
        else:
            loss.backward()
        return loss

    def build_optimizer(self, net):
        weight_decay = getattr(self.args, 'weight_decay', 0.0) or 0.0
        params = [param for param in net.parameters() if param.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found for local training.")
        return torch.optim.SGD(params, lr=self.args.lr, weight_decay=weight_decay)

    def build_scheduler(self, optimizer):
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: 1.0])

    def before_train(self, net, optimizer):
        pass

    def modify_gradients(self, net):
        pass

    def after_step(self, net, step_idx):
        pass

    def after_train(self, net, steps_done):
        pass

    def train(self, net):
        net.train()
        num_batch = len(self.ldr_train)
        if num_batch == 0 or self.iters is None or self.iters <= 0:
            return model_param_dict(net, device=self.device)

        init_params = self._init_reference_params(net)
        optimizer = self.build_optimizer(net)
        scheduler = self.build_scheduler(optimizer)
        self.before_train(net, optimizer)

        steps_done = 0
        while steps_done < self.iters:
            for batch in self.ldr_train:
                if steps_done >= self.iters:
                    break
                inputs = self._process_batch(batch)
                net.zero_grad()
                outputs = self._forward_pass(net, inputs)
                self._compute_loss_and_backward(net, outputs, init_params)
                self.modify_gradients(net)
                optimizer.step()
                scheduler.step()
                steps_done += 1
                self.after_step(net, steps_done)

        self.after_train(net, steps_done)
        return model_param_dict(net, device=self.device)


class LocalSGD(Local):
    pass


class LocalFedASMU(Local):
    def __init__(
        self,
        *args,
        fresh_global=None,
        start_version=0,
        fresh_version=0,
        request_step=None,
        gamma_ctrl=1.0,
        nu_ctrl=0.0,
        mu_beta=1.0,
        eta_gamma=0.01,
        eta_nu=0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fresh_global = fresh_global
        self.start_version = int(start_version)
        self.fresh_version = int(fresh_version)
        self.request_step = request_step
        self.gamma_ctrl = float(gamma_ctrl)
        self.nu_ctrl = float(nu_ctrl)
        self.mu_beta = float(mu_beta)
        self.eta_gamma = float(eta_gamma)
        self.eta_nu = float(eta_nu)
        self.last_reward = 0.0
        self.last_beta = 0.0
        self.did_merge = False

    def _has_fresh_model(self):
        return (
            self.fresh_global is not None
            and self.request_step is not None
            and self.request_step > 0
            and self.fresh_version > self.start_version
        )

    def _compute_beta(self):
        if not self._has_fresh_model():
            return 0.0
        version_term = math.sqrt(max(1.0, float(self.fresh_version)))
        gap_term = math.sqrt(max(1.0, float(self.fresh_version - self.start_version + 1)))
        phi = (self.gamma_ctrl / version_term) * (1.0 - self.nu_ctrl / gap_term)
        scaled = self.mu_beta * phi
        if scaled <= 0:
            return 0.0
        beta = scaled / (1.0 + scaled)
        return float(max(0.0, min(1.0, beta)))

    def _batch_loss(self, net, batch):
        was_training = net.training
        net.eval()
        with torch.no_grad():
            inputs = self._process_batch(batch)
            outputs = self._forward_pass(net, inputs)
            loss = float(outputs.loss.item())
        if was_training:
            net.train()
        return loss

    def _merge_fresh_model(self, net, beta):
        current = model_param_dict(net, device=self.device)
        merged = sd_lerp(current, self.fresh_global, beta)
        load_param_dict_(net, merged)

    def _update_local_controls(self, reward):
        self.gamma_ctrl = max(0.0, self.gamma_ctrl + self.eta_gamma * reward)
        self.nu_ctrl = max(0.0, self.nu_ctrl - self.eta_nu * reward)

    def train(self, net):
        net.train()
        num_batch = len(self.ldr_train)
        if num_batch == 0 or self.iters is None or self.iters <= 0:
            return model_param_dict(net, device=self.device)

        init_params = self._init_reference_params(net)
        optimizer = self.build_optimizer(net)
        scheduler = self.build_scheduler(optimizer)
        self.before_train(net, optimizer)

        steps_done = 0
        data_iter = iter(self.ldr_train)
        while steps_done < self.iters:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.ldr_train)
                batch = next(data_iter)

            if self._has_fresh_model() and (steps_done + 1) == self.request_step:
                loss_before = self._batch_loss(net, batch)
                beta = self._compute_beta()
                if beta > 0:
                    self._merge_fresh_model(net, beta)
                loss_after = self._batch_loss(net, batch)
                reward = loss_before - loss_after
                self._update_local_controls(reward)
                self.last_reward = reward
                self.last_beta = beta
                self.did_merge = beta > 0

            inputs = self._process_batch(batch)
            net.zero_grad()
            outputs = self._forward_pass(net, inputs)
            self._compute_loss_and_backward(net, outputs, init_params)
            self.modify_gradients(net)
            optimizer.step()
            scheduler.step()
            steps_done += 1
            self.after_step(net, steps_done)

        self.after_train(net, steps_done)
        return model_param_dict(net, device=self.device)
