import math
import os

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

from layers import ZINBLoss, MeanAct, DispAct
from single_cell_tools import cluster_acc

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


def euclidean_dist(x, y):
    return torch.sum(torch.square(x - y), dim=1)


class DCAN(nn.Module):
    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[],
                 activation="relu", sigma=1., alpha=1., gamma=1., device="cuda"):
        super(DCAN, self).__init__()
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.encoder = buildNetwork([input_dim] + encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim] + decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
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
        return self.gamma * kldloss

    def x_drop(self, x, p=0.2):
        mask_list = [torch.rand(x.shape[1]) < p for _ in range(x.shape[0])]
        mask = torch.vstack(mask_list)
        new_x = x.clone()
        new_x[mask] = 0.0
        return new_x

    def pretrain_autoencoder(self, X, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, c1_dropout=0.2,
                             c2_dropout=0.2, alpha=0.5, wd=None):
        self.train()
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        criterion = nn.CosineSimilarity(dim=0)
        for epoch in range(epochs):
            loss_val = 0
            zinb_loss_epoch = 0
            instance_loss_epoch = 0

            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                _, mean_tensor, disp_tensor, pi_tensor = self.forwardAE(x_tensor)

                x1 = self.x_drop(x_batch, p=c1_dropout).to(self.device)  # 数据增广
                x2 = self.x_drop(x_batch, p=c2_dropout).to(self.device)  # 数据增广

                x1_encode = self._enc_mu(self.encoder(x1))
                x2_encode = self._enc_mu(self.encoder(x2))

                instance_loss = -(
                        criterion(x1_encode, x2_encode).mean() + criterion(x2_encode, x1_encode).mean()) * 0.5
                zinb_ = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                       scale_factor=sf_tensor)
                loss = alpha * zinb_ + (1 - alpha) * instance_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                zinb_loss_epoch += zinb_.item() * len(x_batch)
                instance_loss_epoch += instance_loss.item() * len(x_batch)
                loss_val += loss.item() * len(x_batch)
            print('Pretrain epoch %3d, Total loss: %.8f, ZINB loss: %.8f, Instance loss: %.8f' % (
                epoch + 1, loss_val / X.shape[0],
                zinb_loss_epoch / X.shape[0],
                instance_loss_epoch / X.shape[0],
            ))
            wd.log({"Pretrain ZINB loss": zinb_loss_epoch / X.shape[0],
                    "Pretrain Instance loss": instance_loss_epoch / X.shape[0],
                    "Pretrain Total loss": loss_val / X.shape[0]})

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, size_factor, n_clusters, exp_name, ae_save=True, init_centroid=None, y=None, batch_size=256,
            beta1=0.4, beta2=0.4, beta3=0.2, lr=1., y_pred_init=None, num_epochs=10, update_interval=1, tol=1e-3,
            save_dir="", beta=0.5, wd=None):
        '''X: tensor data'''
        self.train()
        print("Clustering stage")
        X = torch.tensor(X, dtype=torch.float32)
        X_raw = torch.tensor(X_raw, dtype=torch.float32)
        size_factor = torch.tensor(size_factor, dtype=torch.float32)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        update_center_patience = 2 + 1
        history_update_center = deque(maxlen=update_center_patience)
        for i in range(update_center_patience):
            history_update_center.append(-100.0)

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

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        lst = []  # 创建一个空列表，用于存放指标
        pred = []  # 创建一个空列表，用于存放预测标签
        latent_encode = []
        last_ari = 0.0

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
                latent_encode.append(latent)

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    lst.append([final_acc, final_ari, final_nmi])
                    history_update_center.append(ari)
                    print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch + 1, acc, nmi, ari))
                    wd.log({"ACC": acc, "NMI": nmi, "ARI": ari})

                kmeans = KMeans(n_clusters, n_init=20)
                data_ = self.encodeBatch(X)
                y_pred_ = kmeans.fit_predict(data_.data.cpu().numpy())
                ari_ = np.round(metrics.adjusted_rand_score(y, y_pred_), 5)

                if ae_save and ari_ >= last_ari:
                    # print('ari', ari)
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

            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                xrawbatch = X_raw[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                sfbatch = size_factor[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                rawinputs = Variable(xrawbatch).to(self.device)
                sfinputs = Variable(sfbatch).to(self.device)
                target = Variable(pbatch).to(self.device)

                zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                c_loss = self.mu / torch.norm(self.mu, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
                c_loss = torch.mm(c_loss, c_loss.T)  # 矩阵乘法
                c_loss = torch.sum(torch.sum(c_loss)) / (self.mu.shape[0] * self.mu.shape[0])

                cluster_loss = self.cluster_loss(target, qbatch)

                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)

                loss = beta1 * recon_loss + beta2 * cluster_loss + beta3 * c_loss

                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.item() * len(inputs)
                recon_loss_val += recon_loss.item() * len(inputs)
                c_loss_val += c_loss.item() * len(inputs)
                train_loss += loss.item() * len(inputs)

            print("Epoch %3d: Total: %.8f Clustering Loss: %.8f ZINB Loss: %.8f C Loss: %.8f" %
                  (epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num, c_loss_val / num))
            wd.log({"Train ZINB loss": recon_loss_val / num,
                    "Train Clustering loss": cluster_loss_val / num,
                    "Train C loss": c_loss_val / num,
                    "Train loss": train_loss / num})

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
