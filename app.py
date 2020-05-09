import streamlit as st
import pandas as pd
import numpy as np
import gym
from agent import DQNAgent,A2CAgent,DDPGAgent,TD3Agent,PPOAgent
from utils import get_agent_params,episode_update_train,single_step_update_train

agent_dict = {
				'DQN':DQNAgent,
				'A2C':A2CAgent,
				'DDPG':DDPGAgent,
				'TD3':TD3Agent,
				'PPO':PPOAgent,
				}

single_step_update_alg = ['DQN','DDPG','TD3','PPO']
episode_update_alg = ['A2C']

# main UI
st.title('reinforcement learning platform')
game_title = st.empty()
render_area = st.empty()
progress_bar = st.progress(0)
score_area = st.line_chart(pd.DataFrame([[np.nan,np.nan]],columns=['reward','rolling_reward']).astype("float"))

# left sidebar select algo env and common params
st.sidebar.subheader('algorithm')
alg_name = st.sidebar.selectbox('',tuple(agent_dict.keys()))

st.sidebar.subheader('environment')
env_name = st.sidebar.selectbox('',('CartPole-v0','Pendulum-v0','LunarLander-v2','LunarLanderContinuous-v2','BipedalWalker-v3'))

st.sidebar.subheader('Hyperparamter')
max_episodes = st.sidebar.number_input('max_episodes',value=1000)
max_steps = st.sidebar.number_input('max_steps',value=1000)

# if the algorithm need batch_size set batch_size
if alg_name in single_step_update_alg:
	batch_size = st.sidebar.number_input('batch_size',value=32)

# get_agent_params and user set
alg_param = {}
for key,value in get_agent_params(agent_dict[alg_name]).items():
	alg_param[key] = st.sidebar.number_input(key,value=value,format="%.4f")

# start button
start = st.sidebar.button('start training')

if start:
	env = gym.make(env_name)
	agent = agent_dict[alg_name](env,**alg_param)

	if alg_name == 'PPO':
		print('PPO selected!')
		agent.train(max_episodes=max_episodes,max_steps=max_steps,batch_size=batch_size,
			render_area=render_area,score_area=score_area,progress_bar=progress_bar)
	
	if alg_name in episode_update_alg:
		episode_update_train(env, agent, max_episodes, max_steps,render_area,score_area,progress_bar)
	
	if alg_name in single_step_update_alg:
		single_step_update_train(env, agent, max_episodes, max_steps, batch_size,render_area,score_area,progress_bar)




