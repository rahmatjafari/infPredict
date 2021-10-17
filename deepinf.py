# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:35:53 2021

@author: rjafa
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import shutil
import os
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from torch.utils.data import DataLoader
import sklearn
import itertools
import logging
import igraph
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from tensorboard_logger import tensorboard_logger

logger = logging.getLogger(__name__)

def load_w2v_feature(file, max_idx=0):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[0.] * d for i in range(max(n, max_idx + 1))]
                continue
            index = int(content[0])
            while len(feature) <= index:
                feature.append([0.] * d)
            for i, x in enumerate(content[1:]):
                feature[index][i] = float(x)
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)

class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class InfluenceDataSet(Dataset):
    def __init__(self, file_dir, embedding_dim, seed, shuffle, model):
        self.graphs = np.load(os.path.join(file_dir, "adjacency_matrix.npy")).astype(np.float32)

        # self-loop trick, the input graphs should have no self-loop
        identity = np.identity(self.graphs.shape[1])
        self.graphs += identity
        self.graphs[self.graphs != 0] = 1.0
        if model == "gat" or model == "pscn":
            self.graphs = self.graphs.astype(np.dtype('B'))
        elif model == "gcn":
            # normalized graph laplacian for GCN: D^{-1/2}AD^{-1/2}
            for i in range(len(self.graphs)):
                graph = self.graphs[i]
                d_root_inv = 1. / np.sqrt(np.sum(graph, axis=1))
                graph = (graph.T * d_root_inv).T * d_root_inv
                self.graphs[i] = graph
        else:
            raise NotImplementedError
        logger.info("graphs loaded!")

        # wheather a user has been influenced
        # wheather he/she is the ego user
        self.influence_features = np.load(
                os.path.join(file_dir, "influence_feature.npy")).astype(np.float32)
        logger.info("influence features loaded!")

        self.labels = np.load(os.path.join(file_dir, "label.npy"))
        logger.info("labels loaded!")

        self.vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
        logger.info("vertex ids loaded!")

        if shuffle:
            self.graphs, self.influence_features, self.labels, self.vertices = \
                    sklearn.utils.shuffle(
                        self.graphs, self.influence_features,
                        self.labels, self.vertices,
                        random_state=seed
                    )

        vertex_features = np.load(os.path.join(file_dir, "vertex_feature.npy"))
        vertex_features = preprocessing.scale(vertex_features)
        self.vertex_features = torch.FloatTensor(vertex_features)
        logger.info("global vertex features loaded!")

        embedding_path = os.path.join(file_dir, "deepwalk.emb_%d" % embedding_dim)
        max_vertex_idx = np.max(self.vertices)
        embedding = load_w2v_feature(embedding_path, max_vertex_idx)
        self.embedding = torch.FloatTensor(embedding)
        logger.info("%d-dim embedding loaded!", embedding_dim)

        self.N = self.graphs.shape[0]
        logger.info("%d ego networks loaded, each with size %d" % (self.N, self.graphs.shape[1]))

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.labels))
        self.class_weight = torch.FloatTensor(class_weight)

    def get_embedding(self):
        return self.embedding

    def get_vertex_features(self):
        return self.vertex_features

    def get_feature_dimension(self):
        return self.influence_features.shape[-1]

    def get_num_class(self):
        return np.unique(self.labels).shape[0]

    def get_class_weight(self):
        return self.class_weight

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx], self.influence_features[idx], self.labels[idx], self.vertices[idx]

                
class BatchGraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(BatchGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        init.xavier_uniform_(self.weight)

    def forward(self, x, lap):
        expand_weight = self.weight.expand(x.shape[0], -1, -1)
        support = torch.bmm(x, expand_weight)
        output = torch.bmm(lap, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
             
                
class BatchGCN(nn.Module):
    def __init__(self, n_units, dropout, pretrained_emb, vertex_feature,
            use_vertex_feature, fine_tune=False, instance_normalization=False):
        super(BatchGCN, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)

        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += pretrained_emb.size(1)

        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
            self.vertex_feature.weight = nn.Parameter(vertex_feature)
            self.vertex_feature.weight.requires_grad = False
            n_units[0] += vertex_feature.size(1)

        self.layer_stack = nn.ModuleList()

        for i in range(self.num_layer):
            self.layer_stack.append(
                    BatchGraphConvolution(n_units[i], n_units[i + 1])
                    )

    def forward(self, x, vertices, lap):
        emb = self.embedding(vertices)
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, lap)
            if i + 1 < self.num_layer:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
    
    
# Training settings
class args:
  tensorboard_log='x'              #name of this run
  model = 'gcn'                   #models used
  no_cuda=False                   #Disables CUDA training.
  seed=42                         #Random seed.
  epochs=2                        #Number of epochs to train.
  lr=0.05                         #Initial learning rate.
  weight_decay=1e-3               #Weight decay (L2 loss on parameters).
  dropout=0.2                     #Dropout rate (1 - keep probability).
  hidden_units="16,8"             #Hidden units in each hidden layer, splitted with comma
  heads="1,1,1"                   #Heads in each layer, splitted with comma
  batch=128                      #Batch size
  dim=64                          #Embedding dimension
  check_point=1                  #Check point
  instance_normalization=True    #Enable instance normalization
  shuffle=False                   #Shuffle dataset
  file_dir='D:/payan name/dataset/DeepInf/digg/digg'    #Input file directory
  train_ratio=75                  #Training ratio (0, 100)
  valid_ratio=12.5                  #Validation ratio (0, 100)
  class_weight_balanced=False     #Adjust weights inversely proportional to class frequencies in the input data
  use_vertex_feature=True         #Whether to use vertices' structural features
  sequence_size=16                #Sequence size (only useful for pscn)
  neighbor_size=5                 #Neighborhood size (only useful for pscn)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp



args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

tensorboard_log_dir = 'tensorboard/%s_%s' % (args.model, args.tensorboard_log)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
tensorboard_logger.Logger(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

# adj N*n*n
# feature N*n*f
# labels N*n*c
# Load data
# vertex: vertex id in global network N*n


influence_dataset = InfluenceDataSet(
    args.file_dir, args.dim, args.seed, args.shuffle, args.model)

N = len(influence_dataset)
n_classes = 2
class_weight = influence_dataset.get_class_weight() \
        if args.class_weight_balanced else torch.ones(n_classes)
logger.info("class_weight=%.2f:%.2f", class_weight[0], class_weight[1])

feature_dim = influence_dataset.get_feature_dimension()
n_units = [feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")] + [n_classes]      #2*16*8*2
logger.info("feature dimension=%d", feature_dim)
logger.info("number of classes=%d", n_classes)

train_start,  valid_start, test_start = \
        0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
train_loader = DataLoader(influence_dataset, batch_size=args.batch,
                        sampler=ChunkSampler(valid_start - train_start, 0))
valid_loader = DataLoader(influence_dataset, batch_size=args.batch,
                        sampler=ChunkSampler(test_start - valid_start, valid_start))
test_loader = DataLoader(influence_dataset, batch_size=args.batch,
                        sampler=ChunkSampler(N - test_start, test_start))

# Model and optimizer
if args.model == "gcn":
    model = BatchGCN(pretrained_emb=influence_dataset.get_embedding(),
                vertex_feature=influence_dataset.get_vertex_features(),
                use_vertex_feature=args.use_vertex_feature,
                n_units=n_units,
                dropout=args.dropout,
                instance_normalization=args.instance_normalization)
else:
    raise NotImplementedError
'''
elif args.model == "gat":
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = BatchGAT(pretrained_emb=influence_dataset.get_embedding(),
            vertex_feature=influence_dataset.get_vertex_features(),
            use_vertex_feature=args.use_vertex_feature,
            n_units=n_units, n_heads=n_heads,
            dropout=args.dropout, instance_normalization=args.instance_normalization)
'''


if args.cuda:
    model.cuda()
    class_weight = class_weight.cuda()

params = [{'params': filter(lambda p: p.requires_grad, model.parameters())
    if args.model == "pscn" else model.layer_stack.parameters()}]

optimizer = optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)


def evaluate(epoch, loader, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    for i_batch, batch in enumerate(loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            vertices = vertices.cuda()

        output = model(features, vertices, graph)
        if args.model == "gcn" or args.model == "gat":
            output = output[:, -1, :]
        loss_batch = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
            log_desc, loss / total, auc, prec, rec, f1)

    tensorboard_logger.Logger(log_desc + 'loss', loss / total, epoch + 1)
    tensorboard_logger.Logger(log_desc + 'auc', auc, epoch + 1)
    tensorboard_logger.Logger(log_desc + 'prec', prec, epoch + 1)
    tensorboard_logger.Logger(log_desc + 'rec', rec, epoch + 1)
    tensorboard_logger.Logger(log_desc + 'f1', f1, epoch + 1)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr
    else:
        return None


def train(epoch, train_loader, valid_loader, test_loader, log_desc='train_'):
    model.train()

    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(train_loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            vertices = vertices.cuda()

        optimizer.zero_grad()
        output = model(features, vertices, graph)
        if args.model == "gcn" or args.model == "gat":
            output = output[:, -1, :]
        loss_train = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss in this epoch %f", loss / total)
    tensorboard_logger.Logger('train_loss', loss / total, epoch + 1)
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr = evaluate(epoch, valid_loader, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, test_loader, thr=best_thr, log_desc='test_')


# Train model
t_total = time.time()
logger.info("training...")
for epoch in range(args.epochs):
    train(epoch, train_loader, valid_loader, test_loader)
logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

logger.info("retrieve best threshold...")
best_thr = evaluate(args.epochs, valid_loader, return_best_thr=True, log_desc='valid_')

# Testing
logger.info("testing...")
evaluate(args.epochs, test_loader, thr=best_thr, log_desc='test_')   

influence_dataset.get_class_weight() 
    
