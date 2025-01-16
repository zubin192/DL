import torch
from vmas.simulator.core import Agent
from vmas.simulator.utils import Color


class Coordinator(Agent):
    
    def __init__(self, name, shape = None, movable = True, rotatable = True, collide = True, density = 25, mass = 1, f_range = None, max_f = None, t_range = None, max_t = None, v_range = None, max_speed = None, color=Color.BLUE, alpha = 0.5, obs_range = None, obs_noise = None, u_noise = 0, u_range = 1, u_multiplier = 1, action_script = None, sensors = None, c_noise = 0, silent = True, adversary = False, drag = None, linear_friction = None, angular_friction = None, gravity = None, collision_filter = ..., render_action = False, dynamics = None, action_size = None, discrete_action_nvec = None):
        
        self.agent_type = 'coordinator'
        self.obs = []
        
        super().__init__(name, shape, movable, rotatable, collide, density, mass, f_range, max_f, t_range, max_t, v_range, max_speed, color, alpha, obs_range, obs_noise, u_noise, u_range, u_multiplier, action_script, sensors, c_noise, silent, adversary, drag, linear_friction, angular_friction, gravity, collision_filter, render_action, dynamics, action_size, discrete_action_nvec)
        
    def process_obs(self, obs):
        """Process communicated observations, record in internal map"""
        # TODO
        
        self.obs = obs
        
    def update_specializations(self, env, specs_policy):
        """
        Use specs_policy to updated agent specializations
        
        return torch.Tensor
        """
        # TODO Update when we have a policy
        
        agent_specs = torch.tensor([1.0, 0.5], device=env.device, dtype=torch.float64).repeat(env.batch_dim, 1)
        
        return agent_specs
        
    def get_action(self, env):
        """
        Coordinator does not move.
        """
        action = (torch.tensor([0], device=env.device, dtype=torch.long)
                    .unsqueeze(-1)
                    .expand(env.batch_dim, 1)
                )
        return action.clone()