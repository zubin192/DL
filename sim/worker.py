import numpy as np
import torch
from vmas.simulator.core import Agent
from vmas.simulator.utils import Color


class Worker(Agent):
    
    def __init__(self, name, shape = None, movable = True, rotatable = True, collide = True, density = 25, mass = 1, f_range = None, max_f = None, t_range = None, max_t = None, v_range = None, max_speed = None, color=Color.BLUE, alpha = 0.5, obs_range = None, obs_noise = None, u_noise = 0, u_range = 1, u_multiplier = 1, action_script = None, sensors = None, c_noise = 0, silent = True, adversary = False, drag = None, linear_friction = None, angular_friction = None, gravity = None, collision_filter = ..., render_action = False, dynamics = None, action_size = None, discrete_action_nvec = None, velocity=0.1, modality_funcs=[]):
        
        self.agent_type = 'worker'
        self.obs = []
        self.modality_funcs = modality_funcs
        self.velocity = velocity
        self.action_dict = {0: [0.0,0.0],
                        1: [0.0,-self.velocity],
                        2: [0.0,self.velocity],
                        3: [-self.velocity,0.0],
                        4: [-self.velocity,-self.velocity],
                        5: [-self.velocity,self.velocity],
                        6: [self.velocity,0.0],
                        7: [self.velocity,-self.velocity],
                        8: [self.velocity,self.velocity],                   
                        }
        self.specialization = None
        
        super().__init__(name, shape, movable, rotatable, collide, density, mass, f_range, max_f, t_range, max_t, v_range, max_speed, color, alpha, obs_range, obs_noise, u_noise, u_range, u_multiplier, action_script, sensors, c_noise, silent, adversary, drag, linear_friction, angular_friction, gravity, collision_filter, render_action, dynamics, action_size, discrete_action_nvec)
        
    def process_obs(self, obs):
        """
        Process agent observations, potentially record in internal map
        
        obs dim is obs_dim x batch_size
        """
        
        self.obs = obs
        
        
    def get_action(self,
                   env,
                   selection_strategy='one_step',
                   ):
        """
        Use coordinator-assigned agent specialization to select action
        
        Discrete action indices: 0-8, with:
        - 0 remain stationary
        - 1 down
        - 2 up
        - 3 left
        - 4 down left
        - 5 up left
        - 6 right
        - 7 down right
        - 8 up right
        """
        
        
        
        if selection_strategy == 'one_step': # Use one-step lookahead selection
                
            # Evaluate action per-environment
            best_act_ids = (torch.tensor([0], device=env.device, dtype=torch.long)
                                .unsqueeze(-1)
                                .expand(env.batch_dim, 1)
                                )
            best_act_vals = (torch.tensor([torch.inf], device=env.device)
                                .unsqueeze(-1)
                                .expand(env.batch_dim, 1)
                                )
            
            for act in self.action_dict.keys():
                # TODO: Simulate impacts of action
                # Evaluate each action impact given obs
                act_tensor = (torch.tensor([act], device=env.device, dtype=torch.long)
                                .unsqueeze(-1)
                                .expand(env.batch_dim, 1)
                                )
                
                modality_vals = []
                for mode_func in self.modality_funcs:
                    modality_vals.append(mode_func(self.obs))
                    
                # print("Act", act, "modality list:", modality_vals)
                modality_vals = torch.stack(modality_vals, dim=1)

                # Process modality values, scale according to specialization
                # print("Act", act, "modality vals:", modality_vals, modality_vals.shape)
                # print("Worker specs:", self.specialization, self.specialization.shape)
                act_vals = modality_vals @ self.specialization.float().unsqueeze(1) # TODO check dims
                # print("Act", act, "scaled vals:", act_vals)
                
                # NOTE Finding action that minimizes vals
                best_act_ids = torch.where(act_vals < best_act_vals, act_tensor, best_act_ids)
                best_act_vals = torch.where(act_vals < best_act_vals, act_vals, best_act_vals)

            print("Best act vals:\n", best_act_vals)
            print("Best acts:\n", best_act_ids)

            action = best_act_ids
        
        return action.clone()