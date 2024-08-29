#!/user/bin/python
# author jeff
import sys

import torch
from torch.optim import SGD, Adam, Adagrad, RMSprop, Adadelta, AdamW

# cls表示你用的是哪个梯度下降的原函数，只有下面四种选择。
# *agrs,##kwargs为了解决不同梯度下降函数的参数问题，因为大家要的参数不同，统一用这两个进行入参填补，具体要传入的参数可以在创建这个DPOptimizer的时候输入
# 比如SGD就不需要动能参数。

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, encoder_s, encoder_e, *args, **kwargs):
            # args表示剩余参数的值，kwargs在args之后表示成对键值对。

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size  # batch_size
            self.encoder_s = encoder_s
            self.encoder_e = encoder_e

            for id, group in enumerate(self.param_groups):
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in
                                        group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()  # 这个清零应该是和下面的不一样，这个是模型参数清零

        # 'accum_grads':之前的梯度模型{} accum_grads到底是什么,可以看成一个中转变量

        # 单样本梯度裁剪（每轮固定范数C）
        def microbatch_step(self):  # 一个样本
            total_norm = 0.
            # 范数的计算,遍历一个样本梯度中每个元素
            for group in self.param_groups:  # 整个group里面包含params和accum_grads两个模块，params是每层的张量（正常单元和偏执单元分开）
                for param in group['params']:  # param是具体的那个张量（逐层），偏执单元和正常单元的张量分开
                    if param.requires_grad:
                        if param.grad is None:
                            continue
                        total_norm += param.grad.data.norm(2).item() ** 2.  # 对每层求范数然后把根号开出来，然后对它们求和

            total_norm = total_norm ** .5  # 最后求和的数再取平方根，完成了单样本梯度范数的计算
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)  # 范数比较，得到等下要裁剪的部分
            # 梯度的裁剪
            for group in self.param_groups:
                for iid, (param, accum_grad) in enumerate(zip(group['params'], group['accum_grads'])):
                    # print("iid: ", str(iid), " accum_grad:", accum_grad.shape)
                    if param.requires_grad:
                        if param.grad is None:
                            continue
                        # 裁剪是对param裁剪，#add_：参数更新。对accum_grad进行参数更新,将裁剪后的值加到accum_grad中去
                        accum_grad.add_(param.grad.data.mul(
                            clip_coef))  # 单层梯度裁剪，为什么要单层梯度裁剪呢，裁剪范数（范数的计算是所有层的）不变的情况下，其实单层裁剪和全部一起裁剪是一样的，只是单层更具备可调整性
            return total_norm

        # 这个是accum_grad清零
        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()  # 对accum_grad进行梯度清空，因为如果梯度不清零，pytorch中会将上次计算的梯度和本次计算的梯度累加。

        # 这里做的是全部样本相加、加噪然后平均
        def step_dp(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):  # 整个group里面包含params和accum_grads两个模块 逐层操作
                    if param.requires_grad:
                        # 将accum-grad的值克隆赋值给param.grad
                        param.grad.data = accum_grad.clone()

                        # 对求和的梯度进行加噪。randn_like：返回与输入相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充。torch.randn_like可以理解为标准正态分布
                        param.grad.data.add_(
                            self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        # 再除以batch数平均化
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            # 调用原函数的梯度下降,假设这个step只做梯度下降，包括学习率的那个梯度更新操作
            super(DPOptimizerClass, self).step(*args, **kwargs)

        # 这里做的是全部样本相加、加噪然后平均
        def step_dp_(self, *args, **kwargs):
            for group in self.param_groups:
                for iid, (param, accum_grad) in enumerate(zip(group['params'],
                                             group['accum_grads'])):  # 整个group里面包含params和accum_grads两个模块 逐层操作
                    if param.requires_grad:
                        if param.grad is None:
                            continue
                        # 将accum-grad的值克隆赋值给param.grad
                        param.grad.data = accum_grad.clone()
                        if self.encoder_s <= iid < self.encoder_e:
                        # 对求和的梯度进行加噪。randn_like：返回与输入相同大小的张量，该张量由区间[0,1)上均匀分布的随机数填充。torch.randn_like可以理解为标准正态分布
                            param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        # 再除以batch数平均化
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            # 调用原函数的梯度下降,假设这个step只做梯度下降，包括学习率的那个梯度更新操作
            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass


# 括号后面的是从Pytorch optimizer中调用的梯度下降函数，然后make_optimizer_class是自己对后面调用的原函数进行封装，如上
DPAdam = make_optimizer_class(Adam)
DPAdagrad = make_optimizer_class(Adagrad)
DPSGD = make_optimizer_class(SGD)
DPRMSprop = make_optimizer_class(RMSprop)

DPAdamW = make_optimizer_class(AdamW)
DPAdadelta = make_optimizer_class(Adadelta)
