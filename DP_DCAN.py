import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

from layers import ZINBLoss, MeanAct, DispAct
from single_cell_tools import cluster_acc

from dp.dp_optimizer import DPAdadelta, DPAdam
from dp.compute_rdp import compute_rdp
from dp.rdp_convert_dp import compute_eps

orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]


def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "tanh":
            net.append(nn.Tanh())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


def euclidean_dist(x, y):
    return torch.sum(torch.square(x - y), dim=1)


class DP_DCAN(nn.Module):
    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[],
                 activation="tanh", sigma=1., alpha=1., gamma=1., device="cuda"):
        super(DP_DCAN, self).__init__()
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.encoder = buildNetwork([input_dim] + encodeLayer, type="encode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self.steps = 0

        self.decoder = buildNetwork([z_dim] + decodeLayer, type="decode", activation=activation)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forwardAE(self, x):
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        return z0, _mean, _disp, _pi

    def forward(self, x):
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi

    def encodeBatch(self, X, batch_size=256):
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch).to(self.device)
            z, _, _, _ = self.forwardAE(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

        kldloss = kld(p, q)
        return kldloss

    def x_drop(self, x, p=0.2):
        mask_list = [torch.rand(x.shape[1]) < p for _ in range(x.shape[0])]
        mask = torch.vstack(mask_list)
        new_x = x.clone()
        new_x[mask] = 0.0
        return new_x

    def pretrain_autoencoder(self, X, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, alpha=0.5,
                             c1_dropout=0.2, c2_dropout=0.2, l2_norm_clip=0.1, noise_multiplier=2.0, wd=None):
        microbatch_size = 1

        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = DPAdam(
            l2_norm_clip=l2_norm_clip,  # 裁剪范数
            noise_multiplier=noise_multiplier,  # 噪声乘子
            minibatch_size=batch_size,  # 几个样本梯度进行一次梯度下降
            microbatch_size=microbatch_size,  # 几个样本梯度进行一次裁剪，这里选择逐样本裁剪
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            encoder_s=0,
            encoder_e=10
        )

        criterion = nn.CosineSimilarity(dim=0)
        self.train()
        for epoch in range(epochs):
            loss_val = 0
            zinb_loss_epoch = 0
            instance_loss_epoch = 0

            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                dataset_batch = TensorDataset(torch.Tensor(x_batch), torch.Tensor(x_raw_batch),
                                              torch.Tensor(sf_batch))
                dataloader_batch = DataLoader(dataset_batch, batch_size=microbatch_size,
                                              shuffle=True)  # 每批batch_size=256个，比如268个细胞就只有两批
                optimizer.zero_accum_grad()  # 梯度清空
                for iid, (X_microbatch, y_microbatch, sf_microbatch) in enumerate(dataloader_batch):
                    optimizer.zero_microbatch_grad()  # 每个样本的梯度清空

                    x_tensor = Variable(X_microbatch).to(self.device)  # 数据送给GPU
                    x_raw_tensor = Variable(y_microbatch).to(self.device)
                    sf_tensor = Variable(sf_microbatch).to(self.device)

                    _, mean_tensor, disp_tensor, pi_tensor = self.forwardAE(x_tensor)

                    x1 = self.x_drop(x_batch, p=c1_dropout).to(self.device)  # 数据增广
                    x2 = self.x_drop(x_batch, p=c2_dropout).to(self.device)  # 数据增广

                    x1_encode = self._enc_mu(self.encoder(x1))
                    x2_encode = self._enc_mu(self.encoder(x2))

                    instance_loss = -(
                            criterion(x1_encode, x2_encode).mean() + criterion(x2_encode, x1_encode).mean()) * 0.5
                    zinb_ = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                           scale_factor=sf_tensor)
                    loss = alpha * zinb_ + instance_loss * (1 - alpha)
                    loss.backward()
                    optimizer.microbatch_step()
                    zinb_loss_epoch += zinb_.item() * X_microbatch.shape[0]
                    instance_loss_epoch += instance_loss.item() * X_microbatch.shape[0]
                    loss_val += loss.item() * X_microbatch.shape[0]
                optimizer.step_dp_()
                self.steps += 1
            print('Pretrain epoch %3d, Total loss: %.8f, ZINB loss: %.8f, Instance loss: %.8f' % (
                epoch + 1, loss_val / X.shape[0],
                zinb_loss_epoch / X.shape[0],
                instance_loss_epoch / X.shape[0],
            ))
            wd.log({"Pretrain ZINB loss": zinb_loss_epoch / X.shape[0],
                    "Pretrain Instance loss": instance_loss_epoch / X.shape[0],
                    "Pretrain Total loss": loss_val / X.shape[0]})

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'checkpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, size_factor, n_clusters, exp_name, ae_save=True, init_centroid=None, y=None, lr=1.,
            y_pred_init=None, batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir="", beta1=0.4,
            beta2=0.4, beta3=0.2, l2_norm_clip=0.1, noise_multiplier=2.0, steps_dp=900, wd=None):
        """X: tensor data"""

        print("Clustering stage")
        X = torch.tensor(X, dtype=torch.float32)
        X_raw = torch.tensor(X_raw, dtype=torch.float32)
        size_factor = torch.tensor(size_factor, dtype=torch.float32)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
        optimizer = DPAdadelta(
            l2_norm_clip=l2_norm_clip,  # 裁剪范数
            noise_multiplier=noise_multiplier,  # 噪声乘数
            minibatch_size=batch_size,  # 几个样本梯度进行一次梯度下降
            microbatch_size=1,  # 几个样本梯度进行一次裁剪，这里选择逐样本裁剪
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            rho=.95,
            encoder_s=0,
            encoder_e=11
        )
        print("Initializing cluster centers with kmeans.")
        if init_centroid is None:
            kmeans = KMeans(n_clusters, n_init=20)
            data = self.encodeBatch(X)
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float32))
            self.y_pred = y_pred_init
            self.y_pred_last = self.y_pred
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        is_stop = False

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        lst = []  # 创建一个空列表，用于存放指标
        pred = []  # 创建一个空列表，用于存放预测标签
        latent_encode = []
        last_ari = 0.0

        self.train()
        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                kmeans = KMeans(n_clusters, n_init=20)
                data = self.encodeBatch(X)
                self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
                self.y_pred_last = self.y_pred
                self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
            if epoch % update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                pred.append(self.y_pred)  # 在列表中增加元素，只不过这个元素是每一次预测的标签
                latent_encode.append(latent)  # 在列表中增加元素，只不过这个元素是每一次预测的标签

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    lst.append([final_acc, final_ari, final_nmi])
                    print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch + 1, acc, nmi, ari))
                    wd.log({"ACC": acc, "NMI": nmi, "ARI": ari})

                kmeans = KMeans(n_clusters, n_init=20)
                data_ = self.encodeBatch(X)
                y_pred_ = kmeans.fit_predict(data_.data.cpu().numpy())
                ari_ = np.round(metrics.adjusted_rand_score(y, y_pred_), 5)

                if ae_save and ari_ >= last_ari:
                    torch.save(self.state_dict(), f'model_weights/{exp_name}_params.pth')
                    last_ari = ari_

                # save current model
                if (epoch > 0 and delta_label < tol) or epoch % 10 == 0:
                    self.save_checkpoint({'epoch': epoch + 1,
                                          'state_dict': self.state_dict(),
                                          'mu': self.mu,
                                          'y_pred': self.y_pred,
                                          'y_pred_last': self.y_pred_last,
                                          'y': y
                                          }, epoch + 1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred

            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            c_loss_val = 0.0

            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor), torch.Tensor(p))
            dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 每批batch_size=256个，比如268个细胞就只有两批

            for batch_idx, (xbatch, xrawbatch, sfbatch, pbatch) in enumerate(dataloader1):
                dataset_batch = TensorDataset(torch.Tensor(xbatch), torch.Tensor(xrawbatch),
                                              torch.Tensor(sfbatch), torch.Tensor(pbatch))
                dataloader_batch = DataLoader(dataset_batch, batch_size=1, shuffle=True)
                optimizer.zero_accum_grad()  # 梯度清空
                for iid, (X_microbatch, y_microbatch, sf_microbatch, p_microbatch) in enumerate(dataloader_batch):
                    optimizer.zero_microbatch_grad()

                    inputs = Variable(X_microbatch).to(self.device)
                    rawinputs = Variable(y_microbatch).to(self.device)
                    sfinputs = Variable(sf_microbatch).to(self.device)
                    target = Variable(p_microbatch).to(self.device)

                    zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                    cluster_loss = self.cluster_loss(target, qbatch)
                    recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)

                    c_loss = self.mu / torch.norm(self.mu, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
                    c_loss = torch.mm(c_loss, c_loss.T)  # 矩阵乘法
                    c_loss = torch.sum(torch.sum(c_loss)) / (self.mu.shape[0] * self.mu.shape[0])

                    loss = beta1 * recon_loss + beta2 * cluster_loss + beta3 * c_loss
                    loss.backward()
                    optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作

                    cluster_loss_val += cluster_loss.item() * X_microbatch.shape[0]
                    recon_loss_val += recon_loss.item() * X_microbatch.shape[0]
                    c_loss_val += c_loss.item() * X_microbatch.shape[0]
                    train_loss += loss.item() * X_microbatch.shape[0]
                optimizer.step_dp_()  # 这个做的是梯度加噪和梯度平均更新下降的操作
                self.steps += 1
                if self.steps == steps_dp:
                    is_stop = True
                    break
            print("Epoch %3d: Total: %.8f Clustering Loss: %.8f ZINB Loss: %.8f C Loss: %.8f" %
                  (epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num, c_loss_val / num))
            wd.log({"Train ZINB loss": recon_loss_val / num,
                    "Train Clustering loss": cluster_loss_val / num,
                    "Train C loss": c_loss_val / num,
                    "Train loss": train_loss / num})
            if is_stop:
                break

        rdp = compute_rdp(batch_size / X.shape[0], noise_multiplier, steps_dp, orders)
        eps, opt_order = compute_eps(orders, rdp, 10 ** (-5))  # 再根据RDP转换为对应的最佳eps和lamda
        print("eps:", format(eps) + "| order:", format(opt_order))
        cunari = []  # 初始化
        for j in range(len(lst)):  # j从0到num_epochs-1
            aris = lst[j][1]
            cunari.append(aris)
        max_ari = max(cunari)  # 找到最大的ari
        maxid = cunari.index(max_ari)  # 找到最大的ari的指标
        optimal_pred = pred[maxid]
        optimal_latent = latent_encode[maxid]

        np.savetxt('latents/%s.txt' % exp_name, optimal_latent, delimiter=",")
        print('Best Clustering : ACC= %.4f, NMI= %.4f, ARI= %.4f' % (lst[maxid][0], lst[maxid][2], lst[maxid][1]))
        return optimal_pred, final_acc, final_nmi, final_ari, final_epoch
