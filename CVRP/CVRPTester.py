
import torch

import os
from logging import getLogger

from CVRPEnv import CVRPEnv
from CVRPModel import CVRPModel

from utils import *


class CVRPTester:
    def __init__(self,
                 config):

        self.config = config
        self.tester_params = self.config['test_params']
        self.model_params = self.config['model_params']
        self.env_params = self.config['env_params']

        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = config['use_cuda']
        if USE_CUDA:
            cuda_device_num = config['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # load trained model
        self.model = CVRPModel(**model_params)
        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {'Total': total_num, 'Trainable': trainable_num}
        print(get_parameter_number(self.model))

        self.env = CVRPEnv(config['env_params']['pomo_size'], self.device, config['env_params']['problem_size'])

        self.aug_factor = config['test_params']['aug_factor']

        # utility
        self.time_estimator = TimeEstimator()


    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        avg_route_size = AverageMeter()
        avg_num_route = AverageMeter()
        sol_score = AverageMeter()

        self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device, self.tester_params['test_episodes'])
        self.from_saved_data = True
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problem_size = self.env_params['problem_size']


        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(episode, batch_size)

            sol_s = -1
            if self.from_saved_data:
                # score, aug_score, sol_s = self._test_one_batch(batch_size)
                if self.env.solution is not None:
                    sol_s = self.env._get_travel_distance_sol().mean().item()
            
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            sol_score.update(sol_s, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                if self.from_saved_data:
                    self.logger.info(" SOLUTION SCORE: {:.4f} ".format(sol_score.avg))
                    self.logger.info("            GAP: {:.4f}% ".format((aug_score_AM.avg - sol_score.avg) / sol_score.avg * 100))

        return score_AM.avg, aug_score_AM.avg, (aug_score_AM.avg - sol_score.avg) / sol_score.avg

    def _test_one_batch(self, episode, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # load problems and pre-forward
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()

        # POMO Rollout
        ###############################################
        
        self.model.eval()
        self.model.requires_grad_(False)
        with torch.no_grad():
            self.model.pre_forward(reset_state)
        state, reward, done = self.env.pre_step()

        with torch.no_grad():
            while not done:
                cur_dist = self.env.get_cur_feature()
                selected, prob = self.model.one_step_rollout(state, cur_dist)
                state, reward, done = self.env.step(selected)

                # shape: (batch, pomo)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.multi_width)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()

        

