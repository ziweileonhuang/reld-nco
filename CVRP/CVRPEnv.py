from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F

from utils import augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    dist: torch.Tensor = None
    # shape: (batch, problem+1, problem+1)


@dataclass
class Step_State:
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, multi)
    current_node: torch.Tensor = None
    # shape: (batch, multi)
    ninf_mask: torch.Tensor = None
    # shape: (batch, multi, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, multi)


class CVRPEnv:
    def __init__(self, multi_width, device, problem_size=None):

        # Const @INIT
        ####################################
        self.device = device
        self.vrplib = False
        self.problem_size = problem_size
        self.multi_width = multi_width

        self.depot_xy = None
        self.unscaled_depot_xy = None
        self.node_xy = None
        self.node_demand = None
        self.input_mask = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, multi)
        self.selected_node_list = None
        # shape: (batch, multi, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, multi)
        self.load = None
        # shape: (batch, multi)
        self.visited_ninf_flag = None
        # shape: (batch, multi, problem+1)
        self.ninf_mask = None
        # shape: (batch, multi, problem+1)
        self.finished = None
        # shape: (batch, multi)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

        

    def load_vrplib_problem(self, instance, aug_factor=1, aug_idx=-1):
        
        self.vrplib = True
        self.batch_size = 1
        node_coord = torch.FloatTensor(instance['node_coord']).unsqueeze(0).to(self.device)
        demand = torch.FloatTensor(instance['demand']).unsqueeze(0).to(self.device)
        demand = demand / instance['capacity']
        self.unscaled_depot_node_xy = node_coord
        # shape: (batch, problem+1, 2)
        
        min_x = torch.min(node_coord[:, :, 0], 1)[0]
        min_y = torch.min(node_coord[:, :, 1], 1)[0]
        max_x = torch.max(node_coord[:, :, 0], 1)[0]
        max_y = torch.max(node_coord[:, :, 1], 1)[0]
        scaled_depot_node_x = (node_coord[:, :, 0] - min_x) / (max_x - min_x)
        scaled_depot_node_y = (node_coord[:, :, 1] - min_y) / (max_y - min_y)
        
        # self.depot_node_xy = self.unscaled_depot_node_xy / 1000
        self.depot_node_xy = torch.cat((scaled_depot_node_x[:, :, None]
                                        , scaled_depot_node_y[:, :, None]), dim=2)
        depot = self.depot_node_xy[:, instance['depot'], :]
        # shape: (batch, problem+1)
        if aug_factor > 1:
            if aug_idx == -1:
                if aug_factor == 8:
                    print("Using AUG*8: ")
                    self.batch_size = self.batch_size * 8
                    depot = augment_xy_data_by_8_fold(depot)
                    self.depot_node_xy = augment_xy_data_by_8_fold(self.depot_node_xy)
                    self.unscaled_depot_node_xy = augment_xy_data_by_8_fold(self.unscaled_depot_node_xy)
                    demand = demand.repeat(8, 1)
                else:
                    raise NotImplementedError
            else:
                # take the augmentation with index "aug_idx"
                self.batch_size = self.batch_size
                depot = augment_xy_data_by_8_fold(depot, aug_idx)
                self.depot_node_xy = augment_xy_data_by_8_fold(self.depot_node_xy, aug_idx)
                self.unscaled_depot_node_xy = augment_xy_data_by_8_fold(self.unscaled_depot_node_xy, aug_idx)
                demand = demand.repeat(1, 1)
        
        self.depot_node_demand = demand
        self.reset_state.depot_xy = depot
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand[:, 1:]
        self.reset_state.mask = self.visited_ninf_flag
        self.problem_size = self.reset_state.node_xy.shape[1]

        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist
    
    def load_vrplib_problem_batch(self, instance_list, max_size=200):
        # instance_list = instance_list[:1]
        self.batch_size = len(instance_list)
        # n * 2 -> max_size * 2 padding by 0
        node_coord = torch.stack(
            [F.pad(torch.tensor(instance[0]['node_coord']), 
            (0, 0, 0, max_size - instance[0]['node_coord'].shape[0]), 'constant', 0
            ) for instance in instance_list]).to(self.device)
        self.input_mask = torch.stack([torch.cat(
            (torch.zeros(instance[0]['node_coord'].shape[0]),
            torch.ones(max_size - instance[0]['node_coord'].shape[0]) * 1e-6)) 
            for instance in instance_list]).to(self.device)
        demand = torch.stack(
            [F.pad(torch.tensor(instance[0]['demand']), 
            (0, max_size - instance[0]['demand'].shape[0]), 'constant', 0
            ) for instance in instance_list]).to(self.device)
        capacities = torch.tensor([instance[0]['capacity'] for instance in instance_list]).to(self.device)
        demand = demand / capacities[:, None]
        self.unscaled_depot_node_xy = node_coord
        # shape: (batch, problem+1, 2)
        
        min_x = torch.min(node_coord[:, :, 0], 1)[0][:, None]
        min_y = torch.min(node_coord[:, :, 1], 1)[0][:, None]
        max_x = torch.max(node_coord[:, :, 0], 1)[0][:, None]
        max_y = torch.max(node_coord[:, :, 1], 1)[0][:, None]
        scaled_depot_node_x = (node_coord[:, :, 0] - min_x) / (max_x - min_x)
        scaled_depot_node_y = (node_coord[:, :, 1] - min_y) / (max_y - min_y)
        
        # self.depot_node_xy = self.unscaled_depot_node_xy / 1000
        self.depot_node_xy = torch.cat((scaled_depot_node_x[:, :, None]
                                        , scaled_depot_node_y[:, :, None]), dim=2)
        # depot_idx = torch.tensor(np.array([instance[0]['depot'] for instance in instance_list])).to(self.device)
        # depot = torch.take_along_dim(self.depot_node_xy, depot_idx[:, :, None].expand(self.batch_size, 1, 2), dim=1)
        self.depot_node_demand = demand
        # shape: (batch, problem+1)

        self.reset_state.depot_xy = self.depot_node_xy[:, 0:1, :]
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand[:, 1:]
        self.reset_state.mask = self.visited_ninf_flag
        self.problem_size = self.reset_state.node_xy.shape[1]
        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1) + 1e6 * torch.eye(self.problem_size + 1, device=self.device)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist

    def load_random_problems(self, batch, aug_factor=1, varying_size=None):
        self.batch_size = batch['loc'].shape[0]
        node_coord = batch['loc'].to(self.device)
        demand = batch['demand'].to(self.device)
        depot = batch['depot'].to(self.device)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot = augment_xy_data_by_8_fold(depot)
                node_coord = augment_xy_data_by_8_fold(node_coord)
                demand = demand.repeat(8, 1)
            else:
                raise NotImplementedError
            
        self.depot_node_xy = torch.cat((depot, node_coord), dim=1)
        self.depot_node_demand = torch.cat((torch.zeros(self.batch_size, 1).to(self.device), demand), dim=1)    
            
        self.reset_state.depot_xy = depot
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand
        
        if varying_size is None:
            self.problem_size = self.reset_state.node_xy.shape[1]

        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist
        
    def use_saved_problems(self, filename, device, episodes=None):
        self.FLAG__use_saved_problems = True
        if filename[-3:] != 'txt':

            self.raw_data_node_flag = None
            
            if filename[-3:] == 'pkl':
                self.saved_depot_xy = None
                self.saved_node_xy = None
                self.saved_node_demand = None
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

                for inst_id, inst in enumerate(data[:episodes]):
                    depot_coor, coors, demand, capacity = inst
                    depot_coor = torch.FloatTensor(depot_coor)[None, None, :].to(device)
                    coors = torch.FloatTensor(coors)[None, :, :].to(device)
                    demand = torch.FloatTensor(demand)[None, :].to(device)

                    if self.saved_depot_xy is not None:               
                        self.saved_depot_xy = torch.concat((self.saved_depot_xy, depot_coor), dim=0)
                        self.saved_node_xy = torch.concat((self.saved_node_xy, coors), dim=0)
                        self.saved_node_demand = torch.concat((self.saved_node_demand, demand / capacity), dim=0)
                    else:
                        # print(depot_coor.shape, coors.shape, demand.shape)
                        self.saved_depot_xy = depot_coor
                        self.saved_node_xy = coors
                        self.saved_node_demand = demand / capacity
            else:
                
                loaded_dict = torch.load(filename, map_location=device)
                self.saved_depot_xy = loaded_dict['depot_xy']
                self.saved_node_xy = loaded_dict['node_xy']
                self.saved_node_demand = loaded_dict['node_demand']

        else:
            self.load_dataset(filename, episodes)

        self.saved_index = 0

    def load_dataset(self, filename, episodes=1000000):
        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
            return tow_col_node_flag

        self.raw_data_nodes = []
        self.raw_data_capacity = []
        self.raw_data_demand = []
        self.raw_data_cost = []
        self.raw_data_node_flag = []
        for line in open(filename, "r").readlines()[0:episodes]:
            line = line.split(",")

            depot_index = int(line.index('depot'))
            customer_index = int(line.index('customer'))
            capacity_index = int(line.index('capacity'))
            demand_index = int(line.index('demand'))
            cost_index = int(line.index('cost'))
            node_flag_index = int(line.index('node_flag'))

            depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
            customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]

            loc = depot + customer
            capacity = int(float(line[capacity_index + 1]))
            if int(line[demand_index + 1]) ==0:
                demand = [int(line[idx]) * 1.0 / capacity for idx in range(demand_index + 1, cost_index)]
            else:
                demand = [0] + [int(line[idx]) * 1.0 / capacity for idx in range(demand_index + 1, cost_index)]

            cost = float(line[cost_index + 1])
            node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]

            node_flag = tow_col_nodeflag(node_flag)

            self.raw_data_nodes.append(loc)
            self.raw_data_capacity.append(capacity)
            self.raw_data_demand.append(demand)
            self.raw_data_cost.append(cost)
            self.raw_data_node_flag.append(node_flag)

        
        depot_node_xy = torch.tensor(self.raw_data_nodes, requires_grad=False)

        # shape (B )
        self.raw_data_node_flag = torch.tensor(self.raw_data_node_flag, requires_grad=False)
        # shape (B,V,2)

        self.saved_depot_xy = depot_node_xy[:, [0], :].to(self.device)
        self.saved_node_xy = depot_node_xy[:, 1:, :].to(self.device)
        self.saved_node_demand = torch.tensor(self.raw_data_demand, requires_grad=False)[:, 1:].to(self.device)

        print(f'load raw dataset done!')

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size
        def get_random_problems(batch_size, problem_size, demand_scaler=50, TSP=False):

            depot_xy = torch.rand(size=(batch_size, 1, 2))
            # shape: (batch, 1, 2)


            node_xy = torch.rand(size=(batch_size, problem_size, 2))
            # shape: (batch, problem, 2)

            if demand_scaler == None:
                if problem_size <= 20:
                    demand_scaler = 30
                elif problem_size <= 50:
                    demand_scaler = 40
                elif problem_size <= 100:
                    demand_scaler = 50
                elif problem_size <= 200:
                    demand_scaler = 80
                elif problem_size <= 500:
                    demand_scaler = 100
                else:
                    # raise NotImplementedError
                    demand_scaler = 250
            # demand_scaler = 250
            if TSP == True:
                node_demand = torch.zeros((batch_size, problem_size))
            else:
                node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)

            # shape: (batch, problem)

            return depot_xy, node_xy, node_demand

        if not self.FLAG__use_saved_problems:
            # generate random problems
            # shape: (batch, problem, ...)
            
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size][:, :self.problem_size, :]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index + batch_size][:, :self.problem_size]
            if self.raw_data_node_flag is not None:
                self.solution = self.raw_data_node_flag[self.saved_index:self.saved_index + batch_size]
                # shape (B,V,2)
            else:
                self.solution = None
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        
        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.multi_width)
        self.POMO_IDX = torch.arange(self.multi_width)[None, :].expand(self.batch_size, self.multi_width)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

        
        self.reset_state.problem_size = self.problem_size
    
    def reset(self):
        self.selected_count = 0
        self.current_node = None

        self.multi_width = min(100, self.problem_size)
        # shape: (batch, multi)
        self.selected_node_list = torch.zeros(size=(self.batch_size, self.multi_width, 0), dtype=torch.long, device=self.device)
        # shape: (batch, multi, 0~)
        self.depot_flag = torch.zeros(size=(self.batch_size, self.multi_width, 0), dtype=torch.long, device=self.device)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.multi_width), dtype=torch.bool, device=self.device)
        # shape: (batch, multi)
        self.load = torch.ones(size=(self.batch_size, self.multi_width), device=self.device)
        # shape: (batch, multi)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.multi_width, self.problem_size+1), device=self.device)
        # shape: (batch, multi, problem+1)
        
        self.reset_state.mask = torch.zeros(size=(self.batch_size, self.problem_size + 1), device=self.device)

        if self.input_mask is not None:
            self.visited_ninf_flag = self.input_mask[:, None, :].expand(self.batch_size, self.multi_width, self.problem_size+1).clone()

        self.ninf_mask = torch.zeros(size=(self.batch_size, self.multi_width, self.problem_size+1), device=self.device)
        # shape: (batch, multi, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.multi_width), dtype=torch.bool, device=self.device)
        # shape: (batch, multi)

        reward = None
        done = False
        return self.reset_state, reward, done

    def reset_width(self, new_width):
        self.multi_width = new_width

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.visited_ninf_flag = self.visited_ninf_flag

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, multi)
        # Dynamic-1
        ####################################
        
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, multi)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, multi, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.multi_width, -1)
        # shape: (batch, multi, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, multi, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, multi)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag.scatter_(2, self.selected_node_list, float('-inf'))
        # shape: (batch, multi, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 1e-6
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, multi, problem+1)

        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, multi, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, multi)
        self.finished = self.finished + newly_finished
        # shape: (batch, multi)

        self.depot_flag = torch.cat((self.depot_flag, (self.at_the_depot.long() - self.finished.long())[:, :, None]), dim=2)


        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0
        self.visited_ninf_flag[:, :, 0][self.finished] = 0


        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        # returning values
        done = self.finished.all()
        if done:
            if self.vrplib == True:
                reward = self.compute_unscaled_reward()
            else:
                reward = self._get_reward()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_reward(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, multi)
        return -travel_distances

    def compute_unscaled_reward(self, solutions=None, rounding=True):
        if solutions is None:
            solutions = self.selected_node_list
        gathering_index = solutions[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.unscaled_depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        if rounding == True:
            segment_lengths = torch.round(segment_lengths)
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, multi)
        return -travel_distances

    def get_cur_feature(self):
        if self.current_node is None:
            return None, None
        
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.multi_width, 1, self.problem_size + 1)

        # Compute the relative distance
        cur_dist = torch.take_along_dim(self.dist[:, None, :, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, self.problem_size + 1), 
                                        current_node, dim=2).squeeze(2)
        # shape: (batch, multi, problem)
        return cur_dist


    def _get_travel_distance_sol(self):
        problems = self.depot_node_xy
        order_node = self.solution[:,:,0]
        order_flag = self.solution[:,:,1]
        travel_distances = self.cal_length( problems, order_node, order_flag)
        return travel_distances

    def cal_length(self, problems, order_node, order_flag):
        # problems:   [B,V+1,2]
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()

        order_flag_ = order_flag.clone()
        
        # index_small: position in the solution that is non-depot, 0-indexing
        # index_bigger: position in the solution that return to depot, i.e., index 0
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]

        order_flag_[index_bigger] = 0

        roll_node = order_node_.roll(dims=1, shifts=1)

        problem_size = problems.shape[1] - 1

        order_gathering_index = order_node_.unsqueeze(2).expand(-1, problem_size, 2)
        order_loc = problems.gather(dim=1, index=order_gathering_index)

        roll_gathering_index = roll_node.unsqueeze(2).expand(-1, problem_size, 2)
        roll_loc = problems.gather(dim=1, index=roll_gathering_index)

        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        order_lengths = ((order_loc - flag_loc) ** 2)

        order_flag_[:,0]=0
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        roll_lengths = ((roll_loc - flag_loc) ** 2)

        length = (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)

        return length
