import numpy as np
from deeprobust.graph.global_attack import BaseAttack
import scipy.sparse as sp
import random
import dgl
from scipy.sparse import csr_matrix

class Random(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(Random, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        assert not self.attack_features, 'RND does NOT support attacking features'
    def attack(self, ori_adj, n_perturbations, type='add', **kwargs):
        if self.attack_structure:
            modified_adj = self.perturb_adj(ori_adj, n_perturbations, type)
            self.modified_adj = modified_adj
    def perturb_adj(self, adj, n_perturbations, type='add'):
        modified_adj = adj.tolil()
        type = type.lower()
        assert type in ['add', 'remove', 'flip']
        if type == 'flip':
            # sample edges to flip
            edges = self.random_sample_edges(adj, n_perturbations, exclude=set())
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1 - modified_adj[n1, n2]
                modified_adj[n2, n1] = 1 - modified_adj[n2, n1]
        if type == 'add':
            # sample edges to add
            nonzero = set(zip(*adj.nonzero()))
            edges = self.random_sample_edges(adj, n_perturbations, exclude=nonzero)
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1
                modified_adj[n2, n1] = 1
        if type == 'remove':
            # sample edges to remove
            nonzero = np.array(sp.triu(adj, k=1).nonzero()).T
            indices = np.random.permutation(nonzero)[: n_perturbations].T
            modified_adj[indices[0], indices[1]] = 0
            modified_adj[indices[1], indices[0]] = 0
        # self.check_adj(modified_adj)
        return modified_adj
    def perturb_features(self, features, n_perturbations):
        raise NotImplementedError
        print('number of pertubations: %s' % n_perturbations)
        return modified_features
    def inject_nodes(self, adj, n_add, n_perturbations):
        print('number of pertubations: %s' % n_perturbations)
        raise NotImplementedError
        modified_adj = adj.tolil()
        return modified_adj
    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]
    def sample_forever(self, adj, exclude):
        while True:
            t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))

class DICE(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(DICE, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        assert not self.attack_features, 'DICE does NOT support attacking features'

    def attack(self, ori_adj, labels, n_perturbations, **kwargs):
        print('number of pertubations: %s' % n_perturbations)
        modified_adj = ori_adj.tolil()
        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)

        degree_array = np.array(sp.triu(modified_adj).sum(axis=1)).flatten()
        indices = sp.triu(modified_adj).nonzero()
        indices_list = [x for x in zip(indices[0], indices[1])]
        indices_list = np.random.permutation(indices_list)
        remove_indices = []
        for x in indices_list:
            if labels[x[0]] == labels[x[1]] and len(remove_indices) < n_remove and degree_array[x[0]] > 1 and degree_array[x[1]] > 1:
                remove_indices.append(x)
                degree_array[x[0]] -= 1
                degree_array[x[1]] -= 1
            if len(remove_indices) >= n_remove: break
        # possible_indices = [x for x in zip(indices[0], indices[1]) if labels[x[0]] == labels[x[1]] and x[0]]
        # remove_indices = np.random.permutation(possible_indices)[: n_remove]
        remove_indices = np.array(remove_indices)
        if len(remove_indices) != 0:
            modified_adj[remove_indices[:, 0], remove_indices[:, 1]] = 0
            modified_adj[remove_indices[:, 1], remove_indices[:, 0]] = 0
        n_insert = n_perturbations - n_remove
        added_edges = 0
        while added_edges < n_insert:
            n_remaining = n_insert - added_edges
            candidate_edges = np.array([np.random.choice(ori_adj.shape[0], n_remaining),
                                        np.random.choice(ori_adj.shape[0], n_remaining)]).T
            candidate_edges = set([(u, v) for u, v in candidate_edges if labels[u] != labels[v]
                                        and modified_adj[u, v] == 0 and modified_adj[v, u] == 0])
            candidate_edges = np.array(list(candidate_edges))
            if len(candidate_edges) == 0:
                continue
            modified_adj[candidate_edges[:, 0], candidate_edges[:, 1]] = 1
            modified_adj[candidate_edges[:, 1], candidate_edges[:, 0]] = 1
            added_edges += candidate_edges.shape[0]
        # self.check_adj(modified_adj)
        self.modified_adj = modified_adj
    def sample_forever(self, adj, exclude):
        while True:
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))
    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

def add_perturbation(data, attack_type, structral_attack_ratio):
    ####################################
    # data: DGL format  ————
    # attack_type: random or dice  ————
    ####################################
    clean_feature, label, train_masks, val_masks, test_masks = data.graph.ndata["feature"], data.graph.ndata["label"]\
        , data.graph.ndata["train_masks"], data.graph.ndata["val_masks"], data.graph.ndata["test_masks"]
    src, dst = data.graph.edges()
    num_nodes = data.graph.number_of_nodes()
    adj_data = np.ones(len(src))
    clean_csr_mat = csr_matrix((adj_data, (src.numpy(), dst.numpy())), shape=(num_nodes, num_nodes))
    n_perturbations = int(structral_attack_ratio * (clean_csr_mat.sum()//2))
    if attack_type == "random":
        attack_model = Random()
        attack_model.attack(clean_csr_mat, n_perturbations=n_perturbations)
    elif attack_type == "dice":
        attack_model = DICE()
        attack_model.attack(clean_csr_mat, label, n_perturbations=n_perturbations)
    else:
        attack_model = None
    modified_adj = attack_model.modified_adj
    row_indices, col_indices = modified_adj.nonzero()
    new_graph = dgl.graph((row_indices, col_indices))
    new_graph.ndata["feature"] = clean_feature
    new_graph.ndata["label"] = label
    new_graph.ndata["train_masks"] = train_masks
    new_graph.ndata["val_masks"] = val_masks
    new_graph.ndata["test_masks"] = test_masks
    return new_graph