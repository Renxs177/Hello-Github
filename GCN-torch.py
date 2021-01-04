import time
import math
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


'''
定义一个显示超参数的函数，将代码中所有的超参数打印
'''
def show_Hyperparameter(args):
    argsDict = args.__dict__
    print(argsDict)
    print('the settings are as following')
    for key in argsDict:
        print(key,':',argsDict[key])
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
# show_Hyperparameter(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed) # 设置种子
if args.cuda:
    torch.cuda.manual_seed(args.seed)
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        #  ==== 请教师兄 ====
        super(GraphConvolution, self).__init__() # 子类 GraphConvolution 调用父类的__init__()方法

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

idx_features_labels = np.genfromtxt("./data/cora/cora.content", dtype=np.dtype(str)) # 按照空格分割
print(idx_features_labels)
edges_unordered = np.genfromtxt("./data/cora/cora.cites", dtype=np.int32) # 按照空格分割
print(edges_unordered)

features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
all_labels = idx_features_labels[:, -1]
classes = set(all_labels)
one_hot_coding = np.identity(len(classes))
classes_dict = {c: one_hot_coding[i, :] for i, c in enumerate(classes)}
map_labels = map(classes_dict.get, all_labels)
labels = np.array(list(map_labels), dtype=np.int32)
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered_one_row = edges_unordered.flatten()
map_edge = map(idx_map.get, edges_unordered_one_row)
edges = np.array(list(map_edge), dtype=np.int32).reshape(edges_unordered.shape)
num_edges = edges.shape[0]
data_edge = np.ones(num_edges)
row_edge = edges[:, 0]
col_edge = edges[:, 1]
num_nodes = labels.shape[0]
adj = sp.coo_matrix((data_edge, (row_edge, col_edge)), shape=(num_nodes, num_nodes), dtype=np.float32)
adj = adj + adj.T
rowsum = np.array(features.sum(1))
r_inv = np.power(rowsum, -1).flatten()
r_inv[np.isinf(r_inv)] = 0.
r_mat_inv = sp.diags(r_inv)
features_normalize = r_mat_inv.dot(features)
adj_ = adj + sp.eye(adj.shape[0])
rowsum = np.array(adj_.sum(1))
r_inv = np.power(rowsum, -1).flatten()
r_inv[np.isinf(r_inv)] = 0.
r_mat_inv = sp.diags(r_inv)
adj_normalize = r_mat_inv.dot(adj_)
idx_train = range(140)  # train
idx_val = range(200, 500)  # val
idx_test = range(500, 1500)  # test
features_tensor = torch.FloatTensor(np.array(features_normalize.todense()))
labels_ = np.where(labels)[1]
labels_tensor = torch.LongTensor(labels_)
sparse_mx = adj_normalize.tocoo().astype(np.float32)
indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
values = torch.from_numpy(sparse_mx.data)
shape = torch.Size(sparse_mx.shape)
adj_tensor = torch.sparse.FloatTensor(indices, values, shape)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

nfeat = features_tensor.shape[1]
nhid = args.hidden
nclass = labels_tensor.max().item() + 1
model = GCN(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
t_total = time.time()

for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output_tensor = model(features_tensor, adj_tensor)
    loss_train = F.nll_loss(output_tensor[idx_train], labels_tensor[idx_train])
    preds = output_tensor[idx_train].max(1)[1].type_as(labels_tensor[idx_train])
    correct = preds.eq(labels_tensor[idx_train]).double()
    correct = correct.sum()
    acc_train = correct / len(labels_tensor[idx_train])
    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        model.eval()
        output_tensor = model(features_tensor, adj_tensor)
    loss_val = F.nll_loss(output_tensor[idx_val], labels_tensor[idx_val])
    preds = output_tensor[idx_val].max(1)[1].type_as(labels_tensor[idx_val])
    correct = preds.eq(labels_tensor[idx_val]).double()
    correct = correct.sum()
    acc_val = correct / len(labels_tensor[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
model.eval()
output_tensor = model(features_tensor, adj_tensor)
loss_test = F.nll_loss(output_tensor[idx_test], labels_tensor[idx_test])
preds = output_tensor[idx_test].max(1)[1].type_as(labels_tensor[idx_test])
correct = preds.eq(labels_tensor[idx_test]).double()
correct = correct.sum()
acc_test = correct / len(labels_tensor[idx_test])
print("Test set results:",
      "loss= {:.4f}".format(loss_test.item()),
      "accuracy= {:.4f}".format(acc_test.item()))