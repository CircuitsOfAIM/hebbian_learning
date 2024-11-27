import simulations as simul

if __name__ == '__main__':


	# #Plain Hebb Rule-----------------------------------------------------
	plain_hebb_setup={
		'time':100,
		'step_size':0.1,
		'pre_syn_current':0.5,
		'pre_syn_rate':0,
		'post_syn_current':0.05,
		'post_syn_rate':0,
		'init_weights':0,
		'lr':0.1,
		'verbose':True,
		'plot_title':'Plain Hebb rule weight update',
		'plot_labels':['time','weight update'],
		'plot_limit_range':False
	}

	# simul.simulate_plain_hebb(plain_hebb_setup)
	

	# #BCM Rule-------------------------------------------------------------
	bcm_setup={
		'time':100,
		'step_size':0.1,
		'pre_syn_current':0.5,
		'pre_syn_rate':0,
		'post_syn1_current':0.2,
		'post_syn1_rate':0,		
		'post_syn2_current':0.4,
		'post_syn2_rate':0,
		'init_weights':0,
		'lr':0.1,
		'theta':0.3,
		'verbose':True,
		'plot_title':'BCN rule weight update with pre-synaptic current 0.2 and 0.4',
		'plot_labels':['time','weight update'],
		'plot_legends':['post_syn current 0.2','post_syn current 0.4'],
		'plot_limit_range':False
	}

	# simul.simulate_bcm(bcm_setup)

	# Oja Rule-----------------------------------------------------
	oja_setup={
		'time':100,
		'step_size':0.1,
		'pre_syn1_current':0.5,
		'pre_syn1_rate':0,
		'pre_syn2_current':0.7,
		'pre_syn2_rate':0,
		'post_syn_current':0.1,
		'post_syn_rate':0,
		'init_weights':0,
		'lr':0.1,
		'alpha':1,
		'verbose':True,
		'plot_title':'Oja rule weight update with pre-synaptic current 0.5 and 0.7',
		'plot_labels':['time','weight update'],
		'plot_legends':['pre_syn current 0.5','pre_syn current 0.7'],
		'plot_limit_range':False
	}

	# simul.simulate_oja(oja_setup)


	# EXPERIMENT01--------------------------------------------------
	xp1_setup={
		'time':2,
		'step_size':0.1,
		'pre_syn_current_1':0,
		'pre_syn_rate_1':0,
		'pre_syn_current_2':0,
		'pre_syn_rate_2':0,
		'post_syn_current':0.1,
		'post_syn_rate':0,
		'init_weights':0,
		's_x':1,
		's_y':0.3,
		'theta':45,
		'o':0,
		'lr':0.005,
		'exp_num':1,
		'verbose':True,
		'plot_title':'Weights trajectory for experiment 1 with theta 45',
		'plot_labels':['pre_syn 1 weight','pre_syn 2 weight'],
		'plot_w_trajectory':True,
		'plot_limit_range':3,
		'plot_scatter_currents':True,
	}
	# simul.simulate_xp_6_1(xp1_setup)

	# EXPERIMENT02--------------------------------------------------

	#reconfig for xp2
	xp2_setup = xp1_setup.copy()
	xp2_setup['exp_num']=2
	xp2_setup['theta']=20
	xp2_setup['plot_title']='Weights trajectory for experiment 2 with theta 20'

	# simul.simulate_xp_6_1(xp2_setup)

	# EXPERIMENT03---------------------------------------------------
	xp3_setup = xp1_setup.copy()
	xp3_setup['exp_num']=3
	xp3_setup['theta']=-45
	xp3_setup['o']=2
	xp3_setup['plot_limit_range']=5
	xp3_setup['plot_title']='Weights trajectory for experiment 3 with theta -45 and o=2'

	simul.simulate_xp_6_1(xp3_setup)

	# EXPERIMENT04---------------------------------------------
	
	# xp_6_4_post_syn_neuron = Neuron(current=0.1,rate=0)
	# xp_6_4_pre_syn_neuron_1 = Neuron(current=0,rate=0)
	# xp_6_4_pre_syn_neuron_2 = Neuron(current=0,rate=0)

	# xp_6_4_synapse_01 = Covrule(weight=0,lr=0.005,q=0.1,connection=[xp_6_4_pre_syn_neuron_1,xp_6_4_post_syn_neuron])
	# xp_6_4_synapse_02 = Covrule(weight=0,lr=0.005,q=0.1,connection=[xp_6_4_pre_syn_neuron_2,xp_6_4_post_syn_neuron])
	
	# time,step_size,s_x,s_y,theta,o,exp_num = 2,0.1,1,0.3,-45,2,4



	# # Generating or loading dataset
	# try:
	# 	ds=np.loadtxt(f"./ds{exp_num}.txt", delimiter="\t")
	# 	print(f'dataset for experiment {exp_num} loaded')
	
	# except FileNotFoundError:
	# 	ds = generate_ds(s_x,s_y,theta,o,exp_num)
	# 	print(f'dataset for experiment {exp_num} generated')

	# ds_principal_components = extract_principal_components(ds)



#running the simulation
	# syn_01_weights = []
	# syn_02_weights = []
	
	# for x,y in ds:
	# 	# updating currents and rate of presynaptics
	# 	synapses[0].connection[0].current = x
	# 	synapses[0].connection[0].update_rate()
	# 	synapses[1].connection[0].current = y
	# 	synapses[1].connection[0].update_rate()
	# 	for i in np.arange(0,2,0.1):
	# 		syn_01_weights.append(learn_rule(0.1,0.1,synapses[0]))
	# 		syn_02_weights.append(learn_rule(0.1,0.1,synapses[1]))
	# syn_01_weights=np.hstack(syn_01_weights)
	# syn_02_weights=np.hstack(syn_02_weights)



	# xp4weights_x,xp4weights_y,ds_principal_components=run_simulation(synapses=[xp_6_4_synapse_01,xp_6_4_synapse_02],time=time,step_size=step_size,s_x=s_x,s_y=s_y,theta=theta,o=o,exp_num=exp_num)

	# x_mean_nan = np.nanmean(np.where(np.isinf(xp4weights_x),np.nan,xp4weights_x))
	# y_mean_nan = np.nanmean(np.where(np.isinf(xp4weights_y),np.nan,xp4weights_y))
	# mean_vec =np.array((x_mean_nan,y_mean_nan))

	# stats=calculate_alignment(mean_vec, ds_principal_components[:,0], ds_principal_components[:,1])
	# print(stats)
