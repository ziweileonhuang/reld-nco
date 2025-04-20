import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.utils.data import DataLoader
import numpy as np
import yaml
import wandb
import datetime
import math
import os
from tqdm import trange
from generate_data import generate_vrp_data, VRPDataset
from CVRPModel import CVRPModel
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible, Logger


def test_rollout(loader, env, model):
    avg_cost = 0.
    num_batch = 0.
    for batch in loader:
        env.load_random_problems(batch)
        reset_state, _, _ = env.reset()
        model.eval()
        # greedy rollout
        with torch.no_grad():
            model.pre_forward(reset_state)
            solutions, probs, rewards = rollout(model=model, env=env, eval_type='greedy')
        # check feasible
        check_feasible(solutions[0:1], reset_state.node_demand[0:1])
        batch_cost = -rewards.max(1)[0].mean()
        avg_cost += batch_cost
        num_batch += 1.
    avg_cost /= num_batch

    return avg_cost

def validate(model, multiple_width, device):
    # Initialize env
    env = CVRPEnv(multi_width=multiple_width, device=device)
    # validation dataset
    val_100 = VRPDataset('data/vrp100_val.pkl', num_samples=1000)
    val_100_loader = DataLoader(val_100, batch_size=1000)
    val_200 = VRPDataset('data/vrp200_val.pkl', num_samples=1000)
    val_200_loader = DataLoader(val_200, batch_size=1000)
    val_500 = VRPDataset('data/vrp500_val.pkl', num_samples=100)
    val_500_loader = DataLoader(val_500, batch_size=10)
    # validate
    val_100_cost = test_rollout(val_100_loader, env, model)
    val_200_cost = test_rollout(val_200_loader, env, model)
    val_500_cost = test_rollout(val_500_loader, env, model)
    avg_cost_list = [val_100_cost.cpu().numpy().tolist(), 
                     val_200_cost.cpu().numpy().tolist(), 
                     val_500_cost.cpu().numpy().tolist()]
    return avg_cost_list
    

def train(model, load_checkpoint, start_steps, train_steps, train_batch_size, problem_size, multiple_width, 
 lr, device, logger, fileLogger, dir_path, log_step, scheduler_param):
    # Initialize env
    env = CVRPEnv(multi_width=multiple_width, device=device)
    optimizer = Optimizer(model.parameters(), lr=lr)
    
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        start_steps = checkpoint['step'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()  # Clear gradients if needed
    
    # Set the initial learning rate in each parameter group before using the scheduler
    for param_group in optimizer.param_groups:
        param_group.setdefault('initial_lr', param_group['lr'])

    scheduler = Scheduler(optimizer, milestones=scheduler_param['milestones'], gamma=scheduler_param['gamma'], last_epoch=start_steps - 1)

    rewards_history = []
    best_cost = None
    # REINFORCE training
    for i in trange(start_steps, train_steps + 1):
        model.train()
        
        # lr decay
        scheduler.step()

        problem_size = np.random.randint(40, 101)
        
        batch = generate_vrp_data(dataset_size=train_batch_size, problem_size=problem_size)
        env.load_random_problems(batch, varying_size=None)
        reset_state, _, _ = env.reset()

        model.pre_forward(reset_state)
        solutions, probs, rewards = rollout(model=model, env=env, eval_type='sample')

        losses = []  # List to store loss values
        optimizer.zero_grad()

        bl_val = rewards.mean(dim=1)[:, None]

        log_prob = probs.log().sum(dim=1)
        advantage = rewards - bl_val
        J = - advantage * log_prob
        
        J = J.mean()

        J.backward()
        optimizer.step()
        
        # Collect loss value
        current_reward = -rewards.max(1)[0].mean().item()
        
        rewards_history.append(current_reward)
        print(f"Step {i}, Reward: {current_reward}")

        # validation and log
        if i % log_step == 0:

            val_info = validate(model, multiple_width, device)
            cost = np.array(val_info)
            if best_cost is None:
                best_cost = cost
            else:
                best_cost = np.min(np.concatenate([cost[:, None], best_cost[:, None]], axis=1), axis=1)

            fileLogger.log(val_info)
            if logger is not None:
                logger.log({'val_100_cost': val_info[0],
                            'val_300_cost': val_info[1],
                            'val_500_cost': val_info[2]},
                        step=i)   

            checkpoint_dict = {
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
            torch.save(checkpoint_dict, dir_path + '/model_epoch_{}.pt'.format(int(i / log_step)))
        

if __name__ == "__main__":
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    # params
    name = config['name']
    device = "cuda:{}".format(config['cuda_device_num']) if config['use_cuda'] else 'cpu'
    logger_name = config['logger']
    load_checkpoint = config['load_checkpoint']
    problem_size = config['train_params']['problem_size']
    multiple_width = config['train_params']['multiple_width']
    start_steps = config['train_params']['start_steps']
    train_steps = config['train_params']['train_steps']
    train_batch_size = config['train_params']['train_batch_size']
    lr = config['train_params']['learning_rate']
    log_step = config['train_params']['log_step']
    model_params = config['model_params']

    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-ts{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}'
    dir_path = 'weights/{}_{}'.format(name, ts_name)
    os.mkdir(dir_path)

    log_config = config.copy()
    param_config = log_config['train_params'].copy()
    log_config.pop('train_params')
    model_params_config = log_config['model_params'].copy()
    log_config.pop('model_params')
    log_config.update(param_config)
    log_config.update(model_params_config)

    # Initialize logger
    if(logger_name == 'wandb'):
        log_config = config.copy()
        param_config = log_config['train_params'].copy()
        log_config.pop('train_params')
        model_params_config = log_config['model_params'].copy()
        log_config.pop('model_params')
        log_config.update(param_config)
        log_config.update(model_params_config)
        logger = wandb.init(project="RELD-CVRP",
                         name=name + ts_name,
                         config=log_config)
    else:
        logger = None
    
    # Initialize fileLogger
    filename = 'log/{}_{}'.format(name, ts_name)
    fileLogger = Logger(filename, config)

    model = CVRPModel(**model_params)
    model.to(device)
    
    # Training
    train(model=model,
          load_checkpoint=load_checkpoint,
          start_steps=start_steps,
          train_steps=train_steps,
          train_batch_size=train_batch_size, 
          problem_size=problem_size, 
          multiple_width=multiple_width,
          lr=lr,
          device=device,
          logger=logger,
          fileLogger=fileLogger,
          dir_path=dir_path,
          log_step=log_step,
          scheduler_param=config['train_params']['scheduler_param'])