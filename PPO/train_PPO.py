from env import RegionSensor
from PPO import PPO
import os
import sys
sys.path.append(os.getcwd())

from itertools import count
from collections import namedtuple
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])



config = {
    'width':10,
    'height':10,
    'envs_img_folder_path':os.getcwd()+'/../demo_dmeirl/demo_label/train',
    'target_svf_delta':{50:0.5,40:1,30:1,20:1,10:1,0:1},
    'model_path':os.getcwd()+'/../demo_dmeirl/demo_result/1_model.pth',
    'max_step_count':20,
    'experts_traj_path':os.getcwd()+'/../demo_dmeirl/demo_expert_trajs_0205.csv',
}

env = RegionSensor(custom_config=config,config=None)
shape_state = env.observation_space.shape
num_action = env.action_space.n
print(f'num_state:{shape_state}')
print(f'num_action:{num_action}')

agent = PPO(shape_state,num_action,aPool_num = 1,layers = [100,100])

for i_epoch in range(5):
    state = env.reset()[0]
    for t in count():
        action,action_prob = agent.select_action(state)
        next_state,reward,done,*_ = env.step(action)
        trans = Transition(state,action,action_prob,reward,next_state)
        agent.store_transition(trans)
        state = next_state

        if done:
            if len(agent.buffer) >= agent.batch_size:agent.update(i_epoch)
            break