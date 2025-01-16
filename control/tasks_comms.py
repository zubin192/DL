

def one_step_lookahead(observation, motion):
    
    new_obs = {}
    
    for key in observation:
        if key == 'pos':
            new_obs[key] = observation[key] + motion
        elif 'pos' in key:
            new_obs[key] = observation[key] - motion
        else:
            new_obs[key] = observation[key]
            
    return new_obs