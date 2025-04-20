import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes

class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.forcing_first_step = model_params['forcing_first_step']

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        # shape: (batch, problem+1, embedding)

        self.decoder.set_kv(self.encoded_nodes)

    def one_step_rollout(self, state, cur_dist, eval_type='greedy'):
        device = state.ninf_mask.device
        batch_size = state.ninf_mask.shape[0]
        multi_width = state.ninf_mask.shape[1]
        problem_size = state.ninf_mask.shape[2] - 1
        
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        elif state.selected_count == 1 and self.forcing_first_step:  # Second Move, POMO
            selected = torch.tensor(random.sample(range(0, problem_size), multi_width), device=device)[
                           None, :] \
                    .expand(batch_size, multi_width)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        else:

            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)

            probs = self.decoder(encoded_last_node, state.load, cur_dist, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo+1, problem+1)

            if eval_type == 'sample':
                with torch.no_grad():
                    selected = probs.reshape(batch_size * multi_width, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                prob = torch.take_along_dim(probs, selected[:, :, None], dim=2).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                if not (prob != 0).all():   # avoid sampling prob 0
                    prob += 1e-6

            else:
                with torch.no_grad():
                    # Perform the argmax operation
                    
                    if state.selected_count == 1:
                        # select the best multi_width candidates according to prob 
                        selected = torch.topk(probs[:, [0], :], multi_width, 2)[1].reshape(probs.shape[0], multi_width)
                    else:
                        selected = probs.argmax(dim=2).detach()
                        # shape: (batch, pomo+1)
                
                prob = None  # value not needed. Can be anything.

        return selected, prob


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)

        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+1, embedding)

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feed_forward = FeedForward(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = input1 + multi_head_out
        out2 = self.feed_forward(out1)
        out3 = out1 + out2

        return out3
        # shape: (batch, problem, embedding)

########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)

        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # IDT & FF
        self.capacity_mapping = nn.Linear(1, embedding_dim, bias=False)
        self.feed_forward = FeedForward(**model_params)
        
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        
    def set_kv(self, encoded_nodes):

        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)

        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, encoded_last_node, load, cur_dist, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']
        problem_size = ninf_mask.shape[2]
        batch_size = ninf_mask.shape[0]
        pomo_size = ninf_mask.shape[1]

        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)

        q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        # IDT
        mh_atten_out = mh_atten_out + encoded_last_node + self.capacity_mapping(load[:,:,None].clone())
        # FF
        q_refined = self.feed_forward(mh_atten_out) + mh_atten_out

        #  Single-Head Attention, for probability calculation
        score = torch.matmul(q_refined, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['embedding_dim'] ** 0.5
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim

        # -log(dist) heuristic
        score = score_scaled - torch.log(cur_dist)

        score = logit_clipping * torch.tanh(score)

        score_masked = score + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)
        
        return probs

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    if len(qkv.shape) == 4:
        qkv = qkv.reshape(qkv.size(0) * qkv.size(1), qkv.size(2), qkv.size(3))
        # shape: (batch * multi, n, head_num*key_dim)
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))