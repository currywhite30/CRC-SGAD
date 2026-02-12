from dgl.data.utils import load_graphs

class Dataset:
    def __init__(self, name='tfinance', prefix='/root/bsr/CRC-SGAD/datasets/'):
        graph = load_graphs(prefix + name)[0][0]
        self.name = name
        self.graph = graph

    def split(self, semi_supervised=True, trial_id=0):
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id].bool()
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id].bool()
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id].bool()

        self.num_calib = int(sum(self.graph.ndata['test_mask']) * 0.5)
        # self.graph.ndata['calib_mask'] = self.graph.ndata['test_masks'][:,trial_id].bool()[:self.num_calib]
        self.graph.ndata['calib_test_mask'] = self.graph.ndata['test_masks'][:,trial_id].bool()
        
        print(self.graph.ndata['train_mask'].sum(), self.graph.ndata['val_mask'].sum(), self.graph.ndata['test_mask'].sum())