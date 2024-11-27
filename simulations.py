import helpers
import environment as simul_env
import rules
import pandas as pd
import numpy as np



def simulate_plain_hebb(setup):
	'''
	instantiaciate neurons and synapses, runs simulations and plot the results
	'''
	heb_pre_syn_neuron = simul_env.Neuron(
		current=setup['pre_syn_current'],
		rate=setup['pre_syn_rate']
		)

	heb_post_syn_neuron = simul_env.Neuron(
		current=setup['post_syn_current'],
		rate=setup['post_syn_rate']
		)

	plain_hebb_01 = rules.Plain_Hebb(
		weight=setup['init_weights'],
		lr=setup['lr'],
		connection=[heb_pre_syn_neuron,heb_post_syn_neuron])

	plain_hebb_weights_df= helpers.run_simulation(
		time=setup['time'],
		step_size=setup['step_size'],
		synapse=[plain_hebb_01],
		verbose=setup['verbose']
			)

	helpers.plot_weights(
		time=setup['time'],
		step_size=setup['step_size'],
		plt_title=setup['plot_title'],
		weights_df=plain_hebb_weights_df,
		labels=setup['plot_labels'],
		limit_range=setup['plot_limit_range']
		)


def simulate_bcm(setup):

	bcm_pre_syn_neuron = simul_env.Neuron(
		current=setup['pre_syn_current'],
		rate=setup['pre_syn_rate']
		)

	bcm_post_syn1_neuron = simul_env.Neuron(
		current=setup['post_syn1_current'],
		rate=setup['post_syn1_rate']
		)

	bcm_post_syn2_neuron = simul_env.Neuron(
		current=setup['post_syn2_current'],
		rate=setup['post_syn2_rate']
		)

	bcm_01 = rules.BCM(
		weight=setup['init_weights'],
		lr=setup['lr'],
		connection=[bcm_pre_syn_neuron,bcm_post_syn1_neuron]
		)
	bcm_02 = rules.BCM(
		weight=setup['init_weights'],
		lr=setup['lr'],
		connection=[bcm_pre_syn_neuron,bcm_post_syn2_neuron]
		)
	bcm_weights_df= helpers.run_simulation(
		time=setup['time'],
		step_size=setup['step_size'],
		synapse=[bcm_01,bcm_02],
		verbose=setup['verbose'],
		theta=setup['theta']
		)
	helpers.plot_weights(
		time=setup['time'],
		step_size=setup['step_size'],
		plt_title=setup['plot_title'],
		weights_df=bcm_weights_df,
		labels=setup['plot_labels'],
		legends=setup['plot_legends'],
		limit_range=setup['plot_limit_range'],
		)


def simulate_oja(setup):

	oja_pre_syn1_neuron = simul_env.Neuron(
		current=setup['pre_syn1_current'],
		rate=setup['pre_syn1_rate']
		)	
	oja_pre_syn2_neuron = simul_env.Neuron(
		current=setup['pre_syn2_current'],
		rate=setup['pre_syn2_rate']
		)
	oja_post_syn_neuron = simul_env.Neuron(
		current=setup['post_syn_current'],
		rate=setup['post_syn_rate']
		)

	oja_01 = rules.Oja(
		weight=setup['init_weights'],
		lr=setup['lr'],
		connection=[oja_pre_syn1_neuron,oja_post_syn_neuron]
		)	
	oja_02 = rules.Oja(
		weight=setup['init_weights'],
		lr=setup['lr'],
		connection=[oja_pre_syn2_neuron,oja_post_syn_neuron]
		)

	oja_weights_df= helpers.run_simulation(
		time=setup['time'],
		step_size=setup['step_size'],
		synapse=[oja_01,oja_02],
		verbose=setup['verbose'],
		alpha=setup['alpha'],
		)

	helpers.plot_weights(
		time=setup['time'],
		step_size=setup['step_size'],
		plt_title=setup['plot_title'],
		weights_df=oja_weights_df,
		labels=setup['plot_labels'],
		limit_range=setup['plot_limit_range'],
		legends=setup['plot_legends']
		)

def simulate_xp_6_1(setup):
	'''
	instantiaciate neurons and synapses, runs simulations and plot the results
	'''
	xp_6_1_pre_syn_neuron_1 = simul_env.Neuron(
		current=setup['pre_syn_current_1'],
		rate=setup['pre_syn_rate_1']
		)

	xp_6_1_pre_syn_neuron_2 = simul_env.Neuron(
		current=setup['pre_syn_current_2'],
		rate=setup['pre_syn_rate_2']
		)

	xp_6_1_post_syn_neuron = simul_env.Neuron(
		current=setup['post_syn_current'],
		rate=setup['post_syn_rate']
		)

	xp_6_1_synapse_01 = rules.Plain_Hebb(
		weight=setup['init_weights'],
		lr=setup['lr'],
		connection=[xp_6_1_pre_syn_neuron_1,xp_6_1_post_syn_neuron]
		)

	xp_6_1_synapse_02 = rules.Plain_Hebb(
		weight=setup['init_weights'],
		lr=setup['lr'],
		connection=[xp_6_1_pre_syn_neuron_2,xp_6_1_post_syn_neuron]
		)

	#running the simulation
	dataset = helpers.generate_ds(
		s_x=setup['s_x'],
		s_y=setup['s_y'],
		theta=setup['theta'],
		o=setup['o'],
		exp_num = setup['exp_num']
		)
	
	weights_df_per_datapoint = []
	for x,y in dataset:
	# 	# updating currents and rate of presynaptics
		xp_6_1_synapse_01.connection[0].current = x
		xp_6_1_synapse_01.connection[0].update_rate()
		xp_6_1_synapse_02.connection[0].current = y
		xp_6_1_synapse_02.connection[0].update_rate()
		weights_df_per_datapoint.append(helpers.run_simulation(
			time=setup['time'],
			step_size=setup['step_size'],
			synapse=[xp_6_1_synapse_01,xp_6_1_synapse_02],
			verbose=setup['verbose']
			))
	weights_df = pd.concat(weights_df_per_datapoint)

	helpers.plot_weights(
		time_axis=np.arange(0,setup['time']*len(dataset),setup['step_size']),
		plt_title=setup['plot_title'],
		weights_df=weights_df,
		labels=setup['plot_labels'],
		limit_range=setup['plot_limit_range'],
		w_trajectory=setup['plot_w_trajectory'],
		scatter_currents=setup['plot_scatter_currents'],
		current_ds = dataset
		)
	#TODO, RUN THE EXP1. GENERATE DS SHOULD WORK, WEIGHT UPDATE,CONCATENATE, PLOT, CALC ALIGHNMENT