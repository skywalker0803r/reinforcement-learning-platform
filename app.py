import streamlit as st
import pandas as pd
import numpy as np
import gym
from agent import DQNAgent,A2CAgent,DDPGAgent
from utils import on_policy_train,off_policy_train,get_agent_params

agent_dict = {
				'DQN':DQNAgent,
				'A2C':A2CAgent,
				'DDPG':DDPGAgent,
				}

off_policy_alg = ['DQN','DDPG']
on_policy_alg = ['A2C']

# main UI
st.title('reinforcement learning platform')
game_title = st.empty()
render_area = st.empty()
progress_bar = st.progress(0)
score_area = st.line_chart(pd.DataFrame([[np.nan,np.nan]],columns=['reward','rolling_reward']).astype("float"))

# left sidebar select algo env and common params
st.sidebar.subheader('algorithm')
alg_name = st.sidebar.selectbox('',('DQN','A2C','DDPG'))

st.sidebar.subheader('environment')
env_name = st.sidebar.selectbox('',('CartPole-v0','Pendulum-v0','LunarLander-v2','LunarLanderContinuous-v2','BipedalWalker-v3'))

st.sidebar.subheader('Hyperparamter')
max_episodes = st.sidebar.number_input('max_episodes',value=1000)
max_steps = st.sidebar.number_input('max_steps',value=1000)

# if the algorithm need batch_size set batch_size
if alg_name in off_policy_alg:
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
	
	if alg_name in on_policy_alg:
		on_policy_train(env, agent, max_episodes, max_steps,render_area,score_area,progress_bar)
	
	if alg_name in off_policy_alg:
		off_policy_train(env, agent, max_episodes, max_steps, batch_size,render_area,score_area,progress_bar)




