import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

def value_iteration(threshold, world, rewards, discount = 0.01):
    V = torch.zeros(world.n_states,dtype=torch.float32).to(device)
    delta = np.inf 

    while delta > threshold:
        delta = 0
        for s in world.states:
            max_v = torch.tensor([-float('inf')]).to(device)
            for a in world.actions:
                probs = torch.from_numpy(world.dynamics[:,a,s]).float().to(device)
                max_v = torch.maximum(max_v,torch.dot(probs,rewards+discount*V))
            delta = max(delta,torch.abs(max_v-V[s]).detach().cpu().numpy())
            V[s] = max_v

    policy = torch.zeros((world.n_states,world.n_actions),dtype=torch.float32).to(device)
    for s in world.states:
        for a in world.actions:
            probs = torch.from_numpy(world.dynamics[:,a,s]).float().to(device)
            policy[s,a] = torch.dot(probs,rewards+discount*V)

    policy = policy - policy.max(dim=1,keepdim=True)[0]
    exps = torch.exp(policy)
    policy = exps / exps.sum(dim=1,keepdim=True)
    return policy

