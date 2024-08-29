import os
from time import time

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import metrics

from preprocess import read_dataset, normalize
from DCAN import DCAN
from single_cell_tools import geneSelection, cluster_acc
import wandb


# for repeatability
torch.manual_seed(42)

if __name__ == "__main__":
    # setting the hyperparameters
    import argparse

    parser = argparse.ArgumentParser(
        description='DCAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=0, type=int,
                        help='number of clusters, 0 means estimating by the Louvain algorithm')
    parser.add_argument('--knn', default=20, type=int,
                        help='number of nearest neighbors, used by the Louvain algorithm')
    parser.add_argument('--resolution', default=.8, type=float,
                        help='resolution parameter, used by the Louvain algorithm, larger value for more number of '
                             'clusters')
    parser.add_argument('--select_genes', default=0, type=int,
                        help='number of selected genes, 0 means using all genes')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--experiment_name', default='')
    parser.add_argument('--fit_epochs', default=2000, type=int)
    parser.add_argument('--z_dim', default=32, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float, help='')
    parser.add_argument('--c1_dropout', default=0.2, type=float, help='')
    parser.add_argument('--c2_dropout', default=0.2, type=float, help='')
    parser.add_argument('--sigma', default=2.5, type=float, help='')
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.5, type=float)
    parser.add_argument('--beta3', default=0.5, type=float)
    parser.add_argument('--l2_norm_clip', default=0.1, type=float)
    parser.add_argument('--noise_multiplier', default=2.0, type=float)
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float,
                        help='tolerance for delta clustering labels to terminate training stage')
    parser.add_argument('--ae_weights', default=None,
                        help='file to pretrained weights, None for a new pretraining')
    parser.add_argument('--save_dir', default='checkpoints/SCC/',
                        help='directory to save model weights during the training stage')
    parser.add_argument('--predict_label_file', default='pred_labels.txt',
                        help='file name to save final clustering labels')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="DP-DCAN",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{args.data_file[:-3]}_{args.experiment_name}",
        # Track hyperparameters and run metadata
        config={
            "dataset": args.data_file,
            "sigma": args.sigma,
            "alpha": args.alpha,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "beta3": args.beta3,
            "pretrain_epochs": args.pretrain_epochs,
            "fit_epochs": args.fit_epochs,
            "batch_size": args.batch_size,
            "z_dim": args.z_dim,
            "select_genes": args.select_genes,
            "c1_dropout": args.c1_dropout,
            "c2_dropout": args.c2_dropout,
        })

    data_mat = h5py.File(f"datasets/train/{args.data_file}", 'r')
    x = np.array(data_mat['X'])
    # y is the ground truth labels for evaluating clustering performance
    # If not existing, we skip calculating the clustering performance metrics (e.g. NMI ARI)
    if 'Y' in data_mat:
        y = np.array(data_mat['Y'])
    else:
        y = None
    data_mat.close()

    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]

    # print(x.shape, '--------------------------------------')
    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    if y is not None:
        adata.obs['Group'] = y

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(args)

    print(adata.X.shape)
    if y is not None:
        print(y.shape)

    #    x_sd = adata.X.std(0)
    #    x_sd_median = np.median(x_sd)
    #    print("median of gene sd: %.5f" % x_sd_median)

    model = DCAN(input_dim=adata.n_vars, z_dim=args.z_dim,
                          encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma,
                          device=args.device)

    print(str(model))

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                   batch_size=args.batch_size, epochs=args.pretrain_epochs, alpha=args.alpha,
                                   c1_dropout=args.c1_dropout, c2_dropout=args.c2_dropout, wd=wandb)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError

    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.n_clusters > 0:
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                       n_clusters=args.n_clusters, init_centroid=None, ae_save=True,
                                       y_pred_init=None, y=y, batch_size=args.batch_size, num_epochs=args.fit_epochs,
                                       update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir,
                                       beta1=args.beta1, beta2=args.beta2, beta3=args.beta3, wd=wandb,
                                       exp_name=f"{args.data_file[:-3]}_{args.experiment_name}")
    else:
        # estimate number of clusters by Louvain algorithm on the autoencoder latent representations
        pretrain_latent = model.encodeBatch(torch.tensor(adata.X, dtype=torch.float32)).cpu().numpy()
        adata_latent = sc.AnnData(pretrain_latent)
        sc.pp.neighbors(adata_latent, n_neighbors=args.knn, use_rep="X")
        sc.tl.louvain(adata_latent, resolution=args.resolution)
        y_pred_init = np.asarray(adata_latent.obs['louvain'], dtype=int)
        features = pd.DataFrame(adata_latent.X, index=np.arange(0, adata_latent.n_obs))
        Group = pd.Series(y_pred_init, index=np.arange(0, adata_latent.n_obs), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        n_clusters = cluster_centers.shape[0]
        print('Estimated number of clusters: ', n_clusters)
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                       n_clusters=n_clusters, init_centroid=cluster_centers,
                                       y_pred_init=y_pred_init, y=y, batch_size=args.batch_size,
                                       num_epochs=args.fit_epochs, update_interval=args.update_interval, tol=args.tol,
                                       save_dir=args.save_dir)

    print('Total time: %d seconds.' % int(time() - t0))

    if y is not None:
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    final_latent = model.encodeBatch(torch.tensor(adata.X, dtype=torch.float32)).cpu().numpy()
    np.savetxt('predicted_y/%s_%s.txt' % (args.data_file[:-3], args.experiment_name), y_pred, delimiter=",", fmt="%i")
    wandb.finish()

