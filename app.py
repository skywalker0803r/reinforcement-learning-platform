import streamlit as st
import pandas as pd
import numpy as np
import time
from PIL import Image
import gym
import pyglet
from utils import DDPGAgent,mini_batch_train

# pyglet setting=======================================
pyglet.options['shadow_window']=True

# Hyperparam setting====================================
param_dict = {'DDPG':{'max_episodes':{'min':100,'max':200},
						'max_steps':{'min':500,'max':1000},
						'batch_size':{'min':32,'max':128},
						'gamma':{'min':0.9,'max':0.999},
						'tau':{'min':1e-2,'max':1e-1},
						'buffer_maxlen':{'min':100000,'max':1000000},
						'actor_lr':{'min':1e-3,'max':1e-2},
						'critic_lr':{'min':1e-3,'max':1e-2}}
						}

# main loop=============================================
def main(alg_slot,env_slot,param_slot,render_area,score_area,progress_bar):
	# create env
	env = gym.make(env_slot)
	env.reset()
	
	# create agent
	agent = DDPGAgent(env,param_slot['gamma'], 
		param_slot['tau'],param_slot['buffer_maxlen'],param_slot['critic_lr'], param_slot['actor_lr'])
	
	# training agent use env
	episode_rewards,agent = mini_batch_train(env, agent, 
		param_slot['max_episodes'], param_slot['max_steps'], param_slot['batch_size'],render_area=render_area,score_area=score_area,progress_bar=progress_bar)

	# after traininged plot loss history
	plt.plot(np.array(episode_rewards))
	st.pyplot()

# show main page=========================================
st.title("reinforcement learning platform")
game_title = st.empty()
render_area = st.empty()
progress_bar = st.progress(0)
score_area = st.line_chart(pd.DataFrame([[np.nan,np.nan]],columns=['reward','rolling_reward']).astype("float"))

# show left sidebar======================================
st.sidebar.subheader('algorithm')
alg_slot = st.sidebar.selectbox('algorithm',('DDPG','A2C'))

st.sidebar.subheader('environment')
env_slot = st.sidebar.selectbox('environment',('Pendulum-v0','LunarLander-v2'))
game_title.text(env_slot)

st.sidebar.subheader('Hyperparamter')
param_slot = {}
for param_name,param_values in param_dict[alg_slot].items():
	param_slot[param_name] = st.sidebar.number_input(param_name,param_values['min'],param_values['max'])

start = st.sidebar.button('start training')

# press start button to run=================================
if start:
	main(alg_slot,env_slot,param_slot,render_area,score_area,progress_bar)
		