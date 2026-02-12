import copy
import torch
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
from torch import nn
from scipy.special import comb
from models.attention import GAT
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter,Linear,ModuleList
from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops
import numpy as np

class NF_prop(MessagePassing):
    def __init__(self, K, alpha, Init, num_classes, rank=1, Gamma=None, bias=True, **kwargs):
        super(NF_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        if Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
            TEMP = torch.tensor(np.array([TEMP for i in range(rank)]))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
            TEMP = torch.Tensor(np.array([TEMP for i in range(rank)]))
        elif Init == 'Fix':
            TEMP = np.ones(K + 1)
            TEMP = torch.Tensor(np.array([TEMP for i in range(rank)]))
        elif Init == 'Mine':
            TEMP = []
            para = torch.ones([rank, K + 1])
            TEMP = torch.nn.init.xavier_normal_(para)
        elif Init == 'Mine_PPR':
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
            TEMP = torch.tensor(np.array([TEMP] * rank))
        self.gamma = Parameter(TEMP.float())
        proj_list = []
        for _ in range(K + 1):
            proj_list.append(Linear(num_classes, rank))
        self.proj_list = ModuleList(proj_list)
        self.rank = rank
    def reset_parameters(self):
        torch.nn.init.zeros_(self.gamma)
        for k in range(self.K + 1):
            self.gamma.data[k] = self.alpha * (1 - self.alpha) ** k
        self.gamma.data[-1] = (1 - self.alpha) ** self.K
    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str] = "sym",
                 lambda_max: OptTensor = None, dtype: Optional[int] = None,
                 batch: OptTensor = None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization, dtype, num_nodes)
        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype, device=edge_index.device)
        assert lambda_max is not None
        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]
        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=-1., num_nodes=num_nodes)
        assert edge_weight is not None
        return edge_index, edge_weight
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    def __repr__(self):
        return '{}(K={}, gamma={})'.format(self.__class__.__name__, self.K, self.gamma)
    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = self.__norm__(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        x_list, eta_list = [], []
        Tx_0 = x
        Tx_1 = self.propagate(edge_index, x=Tx_0, norm=norm)
        h_0 = torch.tanh(self.proj_list[0](Tx_0))
        h_1 = torch.tanh(self.proj_list[1](Tx_1))
        gamma_0 = self.gamma[:, 0].unsqueeze(dim=-1)
        gamma_1 = self.gamma[:, 1].unsqueeze(dim=-1)
        eta_0 = torch.matmul(h_0, gamma_0) / self.rank
        eta_1 = torch.matmul(h_1, gamma_1) / self.rank
        hidden = torch.matmul(Tx_0.unsqueeze(dim=-1), eta_0.unsqueeze(dim=-1)).squeeze(dim=-1)
        hidden = hidden + torch.matmul(Tx_1.unsqueeze(dim=-1), eta_1.unsqueeze(dim=-1)).squeeze(dim=-1)
        x_list.append(Tx_0)
        x_list.append(Tx_1)
        eta_list.append(eta_0)
        eta_list.append(eta_1)
        for k in range(1, self.K):
            Tx_2 = 2. * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            Tx_0, Tx_1 = Tx_1, Tx_2
            x_list.append(Tx_1)
            h_k = torch.tanh(self.proj_list[k + 1](Tx_1))
            gamma_k = self.gamma[:, k + 1].unsqueeze(dim=-1)
            eta_k = torch.matmul(h_k, gamma_k) / self.rank
            hidden = hidden + torch.matmul(Tx_1.unsqueeze(dim=-1), eta_k.unsqueeze(dim=-1)).squeeze(dim=-1)
            eta_list.append(eta_k)
        return hidden, torch.stack(eta_list, dim=1).squeeze(dim=-1)

class NFGNN(torch.nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, K=10, alpha=0.5, Init="Mine_PPR", rank=5, Gamma=None,
                 dprate=0.5, dropout=0.4, **kwargs):
        super(NFGNN, self).__init__()
        self.lin1 = Linear(in_feats, h_feats)
        self.lin2 = Linear(h_feats, num_classes)
        self.prop1 = NF_prop(K, alpha, Init, num_classes, rank, Gamma)
        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout
    def reset_parameters(self):
        self.prop1.reset_parameters()
    def forward(self, is_conf, graph):
        if is_conf == 0:
            in_feat = graph.ndata['feature']
        else:
            in_feat = graph.ndata['conf_feature']

        u, v = graph.edges()
        edge_index = torch.stack([u, v], dim=0)
        x = in_feat

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        x, eta = self.prop1(x, edge_index)
        self.eta = eta
        return x#F.log_softmax(x, dim=1)

class DynamicRouting(nn.Module):
    def __init__(self, K, d_model, routing_iter=3):
        super().__init__()
        self.K = K
        self.prototypes = nn.Parameter(torch.Tensor(K, d_model))
        nn.init.xavier_normal_(self.prototypes)
        self.routing_iter = routing_iter

    def forward(self, H):
        # H: [N, d], prototypes: [K, d]
        logits = torch.einsum('nd,kd->nk', H, self.prototypes)  # [N, K]
        for _ in range(self.routing_iter):
            probs = F.softmax(logits, dim=1)
            if not self.training:
                cluster_means = torch.einsum('nk,nd->kd', probs, H) / (probs.sum(0, keepdim=True) + 1e-8)
                self.prototypes.data = 0.9 * self.prototypes + 0.1 * cluster_means
            logits = torch.einsum('nd,kd->nk', H, self.prototypes)
        return probs

class ChebyshevConv(nn.Module):
    def __init__(self, K, M, d_model):
        super().__init__()
        self.K = K
        self.M = M
        self.weights = nn.Parameter(torch.randn(K, M + 1, d_model, d_model))
        nn.init.xavier_normal_(self.weights)

    def _apply_L(self, graph, X, D_invsqrt):
        """ 使用DGL消息传递实现L*X操作 """
        with graph.local_scope():
            graph.ndata['h'] = X * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return X - graph.ndata['h'] * D_invsqrt

    def _chebyshev_basis(self, graph, X):
        """ 生成Chebyshev多项式基 """
        D_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(X.device)
        Tx_0 = X
        if self.M == 0:
            return Tx_0.unsqueeze(0)
        Tx_1 = self._apply_L(graph, X, D_invsqrt)
        basis = [Tx_0, Tx_1]
        for m in range(2, self.M + 1):
            Tx_m = 2 * self._apply_L(graph, Tx_1, D_invsqrt) - Tx_0
            basis.append(Tx_m)
            Tx_0, Tx_1 = Tx_1, Tx_m
        return torch.stack(basis, dim=0)  # [M+1, N, d]

    def forward(self, graph, X, subgraph_probs):
        basis = self._chebyshev_basis(graph, X)  # [M+1, N, d]
        weighted_output = 0
        for k in range(self.K):
            theta = self.weights[k]  # [M+1, d, d]
            probs = subgraph_probs[:, k].unsqueeze(-1)  # [N, 1]
            filtered = torch.einsum('mnd,mde->ne', basis, theta)
            weighted_output += probs * filtered
        return weighted_output

class SSGNN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, K=4, M=3, num_layers=2, mlp_layers=2,
                 dropout_rate=0.0, activation='ReLU', **kwargs):
        super().__init__()
        self.K = K
        self.M = M
        # 特征转换层
        self.linear_in = nn.Linear(in_feats, h_feats)
        self.linear_hid = nn.Linear(h_feats, h_feats)
        # 动态路由和Chebyshev卷积层
        self.routers = nn.ModuleList([DynamicRouting(K, h_feats) for _ in range(num_layers)])
        self.convs = nn.ModuleList([ChebyshevConv(K, M, h_feats) for _ in range(num_layers)])
        # 输出层
        self.mlp = MLP(h_feats * num_layers, h_feats, num_classes, mlp_layers, dropout_rate)
        # 公用组件
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, is_conf, graph):
        if is_conf == 0:
            in_feat = graph.ndata['feature']
        else:
            in_feat = graph.ndata['conf_feature']
        # in_feat = graph.ndata['feature']
        h = self.act(self.linear_in(in_feat))
        h = self.act(self.linear_hid(h))
        all_outputs = []
        for router, conv in zip(self.routers, self.convs):
            subgraph_probs = router(h)
            h = conv(graph, h, subgraph_probs)
            h = self.act(h)
            all_outputs.append(self.dropout(h))
        h_final = torch.cat(all_outputs, dim=-1)
        return self.mlp(h_final, False)

class PolyConv(nn.Module):
    def __init__(self, theta):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k] * feat
        return h

class BernConv(nn.Module):
    def __init__(self, orders=2):
        super().__init__()
        self.K = orders
        self.weight = nn.Parameter(torch.ones(orders + 1))

    def forward(self, graph, feat):
        def unnLaplacian1(feat, D_invsqrt, graph):
            """ \hat{L} X """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        def unnLaplacian2(feat, D_invsqrt, graph):
            """ (2I - \hat{L}) X """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat + graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            tmp = [feat]
            weight = nn.functional.relu(self.weight)
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            for i in range(self.K):
                feat = unnLaplacian2(feat, D_invsqrt, graph)
                tmp.append(feat)

            out_feat = (comb(self.K, 0) / (2 ** self.K)) * weight[0] * tmp[self.K]
            for i in range(self.K):
                x = tmp[self.K - i - 1]
                for j in range(i + 1):
                    x = unnLaplacian1(feat, D_invsqrt, graph)
                out_feat = out_feat + (comb(self.K, i + 1) / (2 ** self.K)) * weight[i + 1] * x
        return out_feat

class AMNet(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, orders=2, num_layers=3, mlp_layers=1, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp_layers = nn.ModuleList()
        if mlp_layers > 0:
            for i in range(mlp_layers - 1):
                self.mlp_layers.append(nn.Linear(h_feats, h_feats))
            self.mlp_layers.append(nn.Linear(h_feats, num_classes))
        self.act = getattr(nn, activation)()
        self.attn = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(BernConv(orders))
        self.linear_transform_in = nn.Sequential(self.linear1, self.act, self.linear2)
        self.W_f = nn.Sequential(self.linear2, self.attn)
        self.W_x = nn.Sequential(self.linear2, self.attn)
        self.linear_cls_out = nn.Sequential(self.mlp_layers)

    def forward(self, is_conf, graph):
        if is_conf == 0:
            in_feat = graph.ndata['feature']
        else:
            in_feat = graph.ndata['conf_feature']
        # in_feat = graph.ndata['feature']
        h = self.linear_transform_in(in_feat)
        h_list = []
        for i, layer in enumerate(self.layers):
            h_ = layer(graph, h)
            h_list.append(h_)
        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(h).unsqueeze(-1)
        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score
        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.num_layers):
            res += h_filters[:, i, :] * score[:, i]
        y_hat = self.linear_cls_out(res)
        return y_hat

def calculate_theta(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)
    return thetas

class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=2, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta(d=num_layers)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(self.thetas[i]))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(h_feats * len(self.conv), h_feats, num_classes, mlp_layers, dropout_rate)
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, is_conf, graph):
        if is_conf == 0:
            in_feat = graph.ndata['feature']
        else:
            in_feat = graph.ndata['conf_feature']
        # in_feat = graph.ndata['feature']
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0], device=h.device)

        for conv in self.conv:
            h0 = conv(graph, h)
            h_final = torch.cat([h_final, h0], -1)
        h_final = self.dropout(h_final)
        h = self.mlp(h_final, False)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act))
        for i in range(num_layers - 1):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act))
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, is_conf, graph):
        if is_conf == 0:
            h = graph.ndata['feature']
        else:
            h = graph.ndata['conf_feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        h = self.mlp(h, False)
        return h

class ConfGNN(torch.nn.Module):
    def __init__(self, model, backbone, in_feats=2, h_feats=32, num_classes=2, num_heads=4, agg='pool', num_layers=2, mlp_layers=1, dropout_rate=0, activation='ReLU'):
        super().__init__()
        self.model = model
        if backbone == 'GCN':
            self.confgnn = GCN(in_feats=in_feats, h_feats=h_feats, num_classes=num_classes, num_layers=num_layers, mlp_layers=mlp_layers, dropout_rate=dropout_rate, activation=activation)
        elif backbone == 'GAT':
            self.confgnn = GAT(in_feats=in_feats, h_feats=h_feats, num_layers=num_layers, hidden_dim_multiplier=1, num_heads=num_heads, num_classes=num_classes, dropout_rate=dropout_rate)
        elif backbone == 'GraphSAGE':
            self.confgnn = GraphSAGE(in_feats=in_feats, h_feats=h_feats, num_classes=num_classes, num_layers=num_layers, agg=agg, dropout_rate=dropout_rate, activation=activation)
        elif backbone == 'PCGNN':
            self.confgnn = ChebNet(in_feats=in_feats, h_feats=h_feats, num_classes=num_classes, num_layers=num_layers, dropout_rate=dropout_rate, activation=activation)
        elif backbone == 'GHRN':
            self.confgnn = BWGNN(in_feats=in_feats, h_feats=h_feats, num_classes=num_classes, num_layers=num_layers, mlp_layers=mlp_layers, dropout_rate=dropout_rate, activation=activation)
        elif backbone == 'BWGNN':
            self.confgnn = BWGNN(in_feats=in_feats, h_feats=h_feats, num_classes=num_classes, num_layers=num_layers, mlp_layers=mlp_layers, dropout_rate=dropout_rate, activation=activation)
        elif backbone == "SSGNN":
            ### amazon bwgnn
            self.confgnn = SSGNN(in_feats=in_feats, h_feats=h_feats, num_classes=num_classes, K=4, M=3, num_layers=num_layers, mlp_layers=mlp_layers, dropout_rate=dropout_rate)
            # self.confgnn = NFGNN(in_feats=in_feats, h_feats=h_feats, num_classes=num_classes, K=3, alpha=0.4, Init="Fix", rank=5, Gamma=None, dprate=0.5, dropout=0.4)
            ### weibo gcn
            # self.confgnn = NFGNN(in_feats=in_feats, h_feats=h_feats, num_classes=num_classes, K=10, alpha=0.5, Init="Mine_PPR", rank=5, Gamma=None, dprate=0.5, dropout=0.4)

    def forward(self, graph):
        with torch.no_grad():
            scores = self.model(0,graph)
        probs = F.softmax(scores, dim=1)
        # new_graph = copy.deepcopy(graph)
        # new_graph.ndata['feature'] = probs
        graph.ndata['conf_feature'] = probs
        adjust_scores = self.confgnn(1,graph)
        return adjust_scores, scores

class ChebNet(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super().__init__()
        self.input_linear = nn.Linear(in_feats, h_feats)
        self.act = getattr(nn, activation)()
        self.chebconv = dglnn.ChebConv(h_feats, h_feats, num_layers, activation=self.act)
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, is_conf, graph):
        if is_conf == 0:
            h = graph.ndata['feature']
        else:
            h = graph.ndata['conf_feature']
        # h = graph.ndata['feature']
        h = self.input_linear(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.chebconv(graph, h, lambda_max=[2])
        h = self.dropout(h)
        h = self.output_linear(h)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, agg='pool', dropout_rate=0,
                 activation='ReLU', **kwargs):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.SAGEConv(in_feats, h_feats, agg, activation=self.act))
        for i in range(num_layers - 1):
            self.layers.append(dglnn.SAGEConv(h_feats, h_feats, agg, activation=self.act))
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, is_conf, graph):
        if is_conf == 0:
            h = graph.ndata['feature']
        else:
            h = graph.ndata['conf_feature']
        # h = graph.ndata['feature']
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h)
        h = self.output_linear(h)
        return h

class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers - 1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata['feature']

        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers) - 1:
                h = self.act(h)
        return h

def hinge_loss(labels, scores):
    margin = 1
    ls = labels * scores

    loss = F.relu(margin - ls)
    loss = loss.mean()
    return loss