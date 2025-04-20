import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
import os, random

def generate_vrp_data(dataset_size, problem_size, random_capacity=True):
    # problem_size.shape: (dataset_size)
    if random_capacity == False:
        # From VRP with RL paper https://arxiv.org/abs/1802.04240
        CAPACITIES = {
                    10: 20.,
                    20: 30.,
                    50: 40.,
                    100: 50.,
                    200: 80.,
                    500: 100.,
                    1000: 250.
                }
        cap=50.0
        data = {
            'loc': torch.FloatTensor(dataset_size, problem_size, 2).uniform_(0, 1),
            # Uniform 1 - 9, scaled by capacities
            'demand': (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 9).int() + 1).float() / cap,
            'depot': torch.FloatTensor(dataset_size, 1, 2).uniform_(0, 1)
        }
    else:
        # Following the set-X of VRPLib ("New benchmark instances for the capacitated vehicle routing problem") to generate capacity
        route_length = torch.tensor(np.random.triangular(3, 6, 25, size=dataset_size))
        demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 9).int() + 1).float()
        capacities = torch.ceil(route_length * demand.sum(1) / problem_size)
        data = {
            'loc': torch.FloatTensor(dataset_size, problem_size, 2).uniform_(0, 1),
            'demand': (demand / capacities[:, None]).float(),
            'depot': torch.FloatTensor(dataset_size, 1, 2).uniform_(0, 1)
        } 

    return data

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }

class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, test=False):
        super(VRPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            if test == True:
                self.data = [make_instance(args) for args in data[offset:offset+num_samples]]
            else:
                self.data = data[offset:offset+num_samples]

        else:
            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.,
                200: 80.,
                500: 100.,
                1000: 250.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(1, 2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]