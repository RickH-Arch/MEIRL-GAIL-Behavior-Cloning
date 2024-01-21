import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
from tqdm import tqdm

def value_iteration(threshold, world, rewards, discount = 0.01,showInfo = False,demo = False):
    V = torch.zeros(world.n_states_active,dtype=torch.float32).to(device)
    delta = np.inf 

    with torch.no_grad():
        if demo:
            while delta > threshold:
                    delta = 0
                    for s in world.states_active:
                        index = world.state_fid[s]
                        max_v = torch.tensor([-float('inf')],dtype=torch.float32).to(device)
                        for a in world.actions:
                            probs = torch.from_numpy(world.dynamics_fid[index,a,:]).float().to(device)
                            max_v = torch.maximum(max_v,torch.dot(probs,rewards+discount*V))
                        index = world.state_fid[s]
                        delta = max(delta,torch.abs(V[index] - max_v).detach().cpu().numpy())
                        V[index] = max_v
                    if showInfo:
                        print(f"delta: {delta}")
        else:
            delta = 0
            for s in world.states_active:
                index = world.state_fid[s]
                max_v = torch.tensor([-float('inf')],dtype=torch.float32).to(device)
                for a in world.actions:
                    probs = torch.from_numpy(world.dynamics_fid[index,a,:]).float().to(device)
                    max_v = torch.maximum(max_v,torch.dot(probs,rewards+discount*V))
                index = world.state_fid[s]
                delta = max(delta,torch.abs(V[index] - max_v).detach().cpu().numpy())
                V[index] = max_v
            delta_last = delta
            total = int((delta - threshold)*1000000)
            with tqdm(total = total) as pbar:
                pbar.set_description("Value Iteration:")
                while delta > threshold:
                    delta = 0
                    for s in world.states_active:
                        index = world.state_fid[s]
                        max_v = torch.tensor([-float('inf')],dtype=torch.float32).to(device)
                        for a in world.actions:
                            probs = torch.from_numpy(world.dynamics_fid[index,a,:]).float().to(device)
                            max_v = torch.maximum(max_v,torch.dot(probs,rewards+discount*V))
                        index = world.state_fid[s]
                        delta = max(delta,torch.abs(V[index] - max_v).detach().cpu().numpy())
                        V[index] = max_v
                    cut_d = threshold if delta < threshold else delta
                    step = delta_last-cut_d
                    pbar.update(int(step*1000000))
                    delta_last = delta
                    if showInfo:
                        print(f"delta: {delta}")
        #print("value_iteration done, computing policy...")

        policy = torch.zeros((world.n_states_active,world.n_actions),dtype=torch.float32).to(device)
        for s in world.states_active:
            index = world.state_fid[s]
            for a in world.actions:
                probs = torch.from_numpy(world.dynamics_fid[index,a,:]).float().to(device)
                policy[index,a] = torch.dot(probs,rewards+discount*V)
        #print("policy done")

    policy = policy - policy.max(dim=1,keepdim=True)[0]
    exps = torch.exp(policy)
    policy = exps / exps.sum(dim=1,keepdim=True)
    return policy

def value_iteration_fullGrid(threshold,world,rewards,discount = 0.01):
    V = torch.zeros(world.n_states,dtype=torch.float32).to(device)
    delta = np.inf 

    with torch.no_grad():
        while delta > threshold:
            delta = 0
            for s in world.states:
                
                max_v = torch.tensor([-float('inf')],dtype=torch.float32).to(device)
                for a in world.actions:
                    probs = torch.from_numpy(world.dynamics[s,a,:]).float().to(device)
                    max_v = torch.maximum(max_v,torch.dot(probs,rewards+discount*V))
                
                delta = max(delta,torch.abs(V[s] - max_v).detach().cpu().numpy())
                V[s] = max_v
            #print(f"delta: {delta}")
        #print("value_iteration done, computing policy...")

        policy = torch.zeros((world.n_states,world.n_actions),dtype=torch.float32).to(device)
        for s in world.states:
            for a in world.actions:
                probs = torch.from_numpy(world.dynamics[s,a,:]).float().to(device)
                policy[s,a] = torch.dot(probs,rewards+discount*V)
        #print("policy done")

    policy = policy - policy.max(dim=1,keepdim=True)[0]
    exps = torch.exp(policy)
    policy = exps / exps.sum(dim=1,keepdim=True)
    return policy
