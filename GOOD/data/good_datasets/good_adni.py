"""
The GOOD-HIV dataset adapted from `MoleculeNet
<https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a>`_.
"""
import itertools
import os
import os.path as osp
import random
from copy import deepcopy
from dgl.data.utils import load_graphs
import gdown
import numpy as np
import torch
from munch import Munch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.datasets import MoleculeNet
from tqdm import tqdm
import csv
import json


class DomainGetter():
    r"""
    A class containing methods for data domain extraction.
    """

    def __init__(self):
        pass

    def get_scaffold(self, smile: str) -> str:
        """
        Args:
            smile (str): A smile string for a molecule.
        Returns:
            The scaffold string of the smile.
        """
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smile), includeChirality=False)
            return scaffold
        except ValueError as e:
            print('Get scaffold error.')
            raise e

    def get_nodesize(self, smile: str) -> int:
        """
        Args:
            smile (str): A smile string for a molecule.
        Returns:
            The number of node in the molecule.
        """
        mol = Chem.MolFromSmiles(smile)
        if (mol is None):
            print('GetNumAtoms error, smiles:{}'.format(smile))
            return len(smile)
        number_atom = mol.GetNumAtoms()
        return number_atom


from GOOD import register


@register.dataset_register
class GOODADNI(InMemoryDataset):
    r"""
    The GOOD-HIV dataset. Adapted from `MoleculeNet
    <https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, data_list: list = None):

        self.name = self.__class__.__name__
        self.mol_name = 'ABIDE'
        self.domain = domain
        self.metric = 'Accuracy'
        self.task = 'Multi-label classification'
        self.url = ''

        self.generate = generate

        super().__init__(root, transform, pre_transform)
        shift_mode = {'no_shift': 0, 'covariate': 3, 'concept': 8}
        mode = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        subset_pt = shift_mode[shift] + mode[subset]
        # self.data, self.slices = torch.load(self.processed_paths[subset_pt])
        # print(data_list[0].edge_index)
        # print(data_list[1].edge_index)
        self.data, self.slices = self.collate(data_list)
        # self.data = data_list

    @property
    def raw_dir(self):
        return osp.join(self.root)

    # def _download(self):
    #     if os.path.exists(osp.join(self.raw_dir, self.name)) or self.generate:
    #         return
    #     if not os.path.exists(self.raw_dir):
    #         os.makedirs(self.raw_dir)
    #     self.download()
    #
    # def download(self):
    #     path = gdown.download(self.url, output=osp.join(self.raw_dir, self.name + '.zip'), fuzzy=True)
    #     extract_zip(path, self.raw_dir)
    #     os.unlink(path)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'ood_val.pt', 'ood_test.pt', 'id_val.pt', 'id_test.pt',]


    def process(self):
        print('#IN#Using default OOD splits')
        # dataset = MoleculeNet(root=self.root, name=self.mol_name)
        # print('Load data done!')

        # data_list = []
        # for i, data in enumerate(dataset):
        #     data.idx = i
        #     data_list.append(data)
        # self.num_data = data_list.__len__()
        # print('Extract data done!')
        # print('#IN#Loading default OOD splits...')
        # all_datasets = self.load()
        # print('#IN#Loading default OOD splits done!')

        # no_shift_list = self.get_no_shift_list(deepcopy(data_list))
        # print('#IN#No shift dataset done!')
        # sorted_data_list, sorted_domain_split_data_list = self.get_domain_sorted_list(data_list, domain=self.domain)
        # covariate_shift_list = self.get_covariate_shift_list(deepcopy(sorted_data_list))
        # print()
        # print('#IN#Covariate shift dataset done!')
        # concept_shift_list = self.get_concept_shift_list(deepcopy(sorted_domain_split_data_list))
        # print()
        # print('#IN#Concept shift dataset done!')

        # all_data_list = no_shift_list + covariate_shift_list + concept_shift_list
        # for i, final_data_list in enumerate(all_data_list):
        #     data, slices = self.collate(final_data_list)
        #     torch.save((data, slices), self.processed_paths[i])

    @staticmethod
    def load(dataset_root: str, domain: str="site", shift: str = 'no_shift', generate: bool = False, fold: int = 0):
        r"""
        A staticmethod for dataset loading. This method instantiates dataset class, constructing train, id_val, id_test,
        ood_val (val), and ood_test (test) splits. Besides, it collects several dataset meta information for further
        utilization.

        Args:
            dataset_root (str): The dataset saving root.
            domain (str): The domain selection. Allowed: 'degree' and 'time'.
            shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
            generate (bool): The flag for regenerating dataset. True: regenerate. False: download.

        Returns:
            dataset or dataset splits.
            dataset meta info.
        """
        meta_info = Munch()
        meta_info.dataset_type = 'brain'
        meta_info.model_level = 'graph'
        meta_info.num_node_features = 100
        meta_info.name = 'adni_ood_schaefer100'
        meta_info.edge_ratio = 0.2
        meta_info.node_feat_transform = 'pearson'
        with open('./GOOD/data/good_datasets/adni_ood_schaefer100/meta.json', 'r') as f:
            meta_json = json.load(f)
            # for kk in meta_json.keys():
            #     meta_json[kk]=np.array(meta_json[kk])

        G_dataset, Labels = load_graphs('/path/to/binfile/adni_schaefer100.bin')

        error_case = []
        # min_feat_dim = G_dataset[0].ndata['N_features'].shape[-1]
        for i in range(len(G_dataset)):
            if len(((G_dataset[i].ndata['N_features'] != 0).sum(dim=-1) == 0).nonzero()) > 0:
                error_case.append(i)
            # if G_dataset[i].ndata['N_features'].shape[-1] < min_feat_dim:
            #     min_feat_dim = G_dataset[i].ndata['N_features'].shape[-1]
        print(error_case)
        # G_dataset = [n for i, n in enumerate(G_dataset) if i not in error_case]

        for i in tqdm(range(len(G_dataset))):
            # if edge_ratio:
            threshold_idx = int(len(G_dataset[i].edata['E_features']) * (1 - meta_info.edge_ratio))
            threshold = sorted(G_dataset[i].edata['E_features'].tolist())[threshold_idx]

            G_dataset[i].remove_edges(torch.squeeze((torch.abs(G_dataset[i].edata['E_features']) < float(threshold)).nonzero()))
            # G_dataset[i].edata['E_features'][G_dataset[i].edata['E_features'] < 0] = 0
            G_dataset[i].edata['feat'] = G_dataset[i].edata['E_features'].unsqueeze(-1).clone()

            if meta_info.node_feat_transform == 'pearson':
                G_dataset[i].ndata['feat'] = torch.from_numpy(np.corrcoef(G_dataset[i].ndata['N_features'].numpy())).clone()
            else:
                raise NotImplementedError

        all_idx = get_all_split_idx(meta_info.name)
        for split in range(10):
            all_idx['train'][split] = [i for i in all_idx['train'][split] if i not in error_case]
            all_idx['id_val'][split] = [i for i in all_idx['id_val'][split] if i not in error_case]
            all_idx['id_test'][split] = [i for i in all_idx['id_test'][split] if i not in error_case]
            all_idx['ood_val'][split] = [i for i in all_idx['ood_val'][split] if i not in error_case]
            all_idx['ood_test'][split] = [i for i in all_idx['ood_test'][split] if i not in error_case]

        train_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['train'][fold]]
        id_val_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['id_val'][fold]]
        id_test_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['id_test'][fold]]
        val_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['ood_val'][fold]]
        test_data = [dgl_to_pyg(G_dataset[idx], Labels['glabel'][idx],meta_json[f'idx2{domain}'][idx]) for idx in all_idx['ood_test'][fold]]

        train_dataset = GOODADNI(root=dataset_root,
                                domain=domain, shift=shift, subset='train', generate=generate, data_list=train_data)
        id_val_dataset = GOODADNI(root=dataset_root,
                                 domain=domain, shift=shift, subset='id_val',
                                 generate=generate, data_list=id_val_data) if shift != 'no_shift' else None
        id_test_dataset = GOODADNI(root=dataset_root,
                                  domain=domain, shift=shift, subset='id_test',
                                  generate=generate, data_list=id_test_data) if shift != 'no_shift' else None
        val_dataset = GOODADNI(root=dataset_root,
                              domain=domain, shift=shift, subset='val', generate=generate, data_list=val_data)
        test_dataset = GOODADNI(root=dataset_root,
                               domain=domain, shift=shift, subset='test', generate=generate, data_list=test_data)

        meta_info.dim_node = 100 #train_dataset.num_node_features
        meta_info.dim_edge = 0 #train_dataset.num_edge_features

        # meta_info.num_envs = torch.unique(train_dataset.data.env_id).shape[0]
        meta_info.num_envs = torch.unique(torch.tensor(meta_json["idx2site"]).long()).shape[0]

        # Define networks' output shape.
        if train_dataset.task == 'Binary classification':
            meta_info.num_classes = 2  # train_dataset.data.y.shape[1]
        elif train_dataset.task == 'Regression':
            meta_info.num_classes = 1
        elif train_dataset.task == 'Multi-label classification':
            meta_info.num_classes = torch.unique(train_dataset.data.y).shape[0]
            # meta_info.num_classes = train_dataset.data.y.shape[-1]

        # --- clear buffer dataset._data_list ---
        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
        val_dataset._data_list = None
        test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'val': val_dataset, 'test': test_dataset, 'task': train_dataset.task,
                'metric': train_dataset.metric}, meta_info


def get_all_split_idx(name):
    """
        - Split total number of graphs into 3 (train, val and test) in 80:10:10
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 10 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 10 fold have unique test set.
    """
    root_idx_dir = './GOOD/data/good_datasets/{}/'.format(name)
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}

    # If there are no idx files, do the split and store the files
    if not (os.path.exists(root_idx_dir + 'train.index')):
        print("[!] no split at {}".format(root_idx_dir))
        raise NotImplementedError

    # reading idx from the files
    for section in ['train', 'ood_val', 'ood_test', 'id_val', 'id_test']:
        with open(root_idx_dir + section + '.index', 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx


def dgl_to_pyg(graph, y, domain):
    # print(graph,y.size(),len(domain))
    # Extract information from the DGLHeteroGraph
    x = graph.ndata['feat']
    edge_index = torch.stack(graph.edges()).contiguous()
    yy = torch.zeros(1)
    yy[0] = y
    # Create a PyG Data object
    data = Data(x=x.float(), edge_index=edge_index, y=yy, domain=domain)
    data.env_id = domain
    # print(data.edge_index.size())
    return data
