from itertools import combinations

import torch

"""
OBSERVATION format:

obs_dict x batch_size

obs_dict = pos, vel (self); agentpos, agentvel (others); taskpos, taskstatus
        

"""

def get_mode_do_tasks(agent_obs: dict):
    """Return relative position mag to nearest incomplete task"""

    pos_tensors = [] # rel pos to tasks
    status_tensors = [] # tasks status
    
    for key in agent_obs.keys():
        if "task" in key and "pos" in key:
            # TODO check norm math here
            pos_tensors.append(torch.norm(agent_obs[key], dim=1).unsqueeze(-1))
        if "task" in key and "status" in key:
            status_tensors.append(agent_obs[key])
            
    # print("Position tensors:", pos_tensors,
    #       "\nStatus tensors:", status_tensors)

    # Stack distance tensors and boolean masks
    stacked_distances = torch.stack(pos_tensors, dim=0)  # Shape: (num_tensors, ...)
    stacked_masks = torch.stack(status_tensors, dim=0)  # Shape: (num_tensors, ...)

    # Replace values where the corresponding boolean mask is True with infinity
    masked_distances = stacked_distances.clone()
    masked_distances[stacked_masks] = torch.inf  # Set masked positions to infinity

    # Compute the minimum across the first dimension (num_tensors)
    min_distances = torch.min(masked_distances, dim=0).values.squeeze()  # Shape: same as input tensors
    
    # print("Min task dists", min_distances)

    return min_distances # mag x batch_size


def get_mode_do_comms(agent_obs: dict):
    """
    Return relative position mag to nearest comms waypoint.
    
    Needs to compute comms waypoint
    """

    # First, find optimal comms waypoint
    # TODO This is currently nearest midpoint between any 2 agents. Maybe update?

    pos_tensors = [] # rel pos to other agents

    for key in agent_obs.keys():
        if "worker" in key or "coordinator" in key and "pos" in key:
            pos_tensors.append(agent_obs[key])
            # print(key,":", agent_obs[key])
    # print("Comms pos tensors", pos_tensors)

    midpt_pos_tensors = []
    for pair in combinations(pos_tensors, 2):
        # print("Evaluating pair", pair)
        midpt_pos_tensors.append(torch.norm((pair[0]-pair[1])/2, dim=1))#.unsqueeze(-1))
    # print("Midpt comms dists", midpt_pos_tensors)
        
    # Returns distance to nearest midpoint
    stacked_distances = torch.stack(midpt_pos_tensors, dim=0)
    # print("Stacked comms dists", stacked_distances)
    min_distances = torch.min(stacked_distances, dim=0).values
    # print("Min comms distances", min_distances)

    return min_distances # mag x batch_size