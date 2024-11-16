import numpy as np
import matplotlib.pyplot as plt


class Neuron:
	def __init__(self,rate,current,synapses=None):
		if synapses is None:
			synapses = []
		self.rate = rate
		self.current = current
		#[synapse1 , ..., synapsen]
		self.synapses = synapses
		self.update_rate()

	def update_rate(self):
		total_sum =0
		for synapse in self.synapses:
			if id(synapse.connection[0])==id(self):
				other_neuron = synapse.connection[1]
			else:
				other_neuron = synapse.connection[0]
			total_sum=total_sum+ (synapse.weight * other_neuron.rate)
		self.rate=total_sum+self.current


class Synapse:
	def __init__(self,weight,lr,connection=None):
		if connection is None:
			connection = [None,None]
		self.weight = weight
		self.lr = lr
		#connection [neruon1,...,neuronn]
		self.connection = connection
		self.set_connection(self.connection)

	def update_weight(self):
		pass
	
	def set_connection(self,connection):
		for neuron in connection:
			neuron.synapses.append(self)
	

class Plain_Hebb(Synapse):
	#synapse with plain Hebb rule
	def __init__(self,*args, **kwargs):
		super().__init__(*args,**kwargs)

	def update_weight(self,dt=0):
		self.weight = self.weight+ dt*(self.lr * self.connection[1].rate *self.connection[0].rate)


class BCM(Synapse):
	#synapse with BCMS rule
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def update_weight(self,dt,theta):
		self.weight = self.weight + dt*(self.lr * self.connection[1].rate*self.connection[0].rate*(self.connection[1].rate-theta))


class Oja(Synapse):
	#synapse with Oja rule
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def update_weight(self,dt,alpha):
		self.weight = self.weight + dt*(self.lr *((self.connection[1].rate*self.connection[0].rate)-(alpha*self.weight* self.connection[1].rate*self.connection[1].rate)))


class Covrule(Synapse):
	#synapse with covariance rule 
	def __init__(self,q,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.q = q
		self.mean_r_j = 0
		self.mean_r_i = 0

	def update_weight(self,dt):
		# calculating means
		self.mean_r_j = (self.q * self.connection[0].rate)+(1-self.q)*self.mean_r_j
		self.mean_r_i = (self.q * self.connection[1].rate)+(1-self.q)*self.mean_r_i

		self.weight = self.weight + dt*(self.lr *(self.connection[1].rate-self.mean_r_i)*(self.connection[0].rate-self.mean_r_j))


def learn_rule(time,step_size,synapse,**kwargs):
	#perform learning rule based on euler method
	
	# weights=[synapse.weight]
	weights=[]
	
	for step in np.arange(0, time, step_size):
		print(f'time>{step_size*step:8f}\nrate_j>{synapse.connection[0].rate}\nrate_i>{synapse.connection[1].rate}\nweight_ij>{synapse.weight}')
		print('--------------')
		synapse.update_weight(step_size,**kwargs)
		synapse.connection[1].update_rate()
		#storing weights for plotting
		weights.append(synapse.weight)
	
	# return synapse.weight		
	return np.array(weights)


def plot_weights(time,step_size,plt_title,weights,labels,current_ds=None,pos_vec_line=False,scatter_currents=False,annotate_ds=np.zeros((1)),limit_range=False):

	plt.figure()
	if type(weights).__name__!='list':
		# plotting a single graph
		print(np.arange(0,time,step_size).shape)
		# print(weights.shape)	
		plt.plot(np.arange(0,time,step_size), weights, label=labels)
		
	else:

		#plotting trajectory of weights
		plt.plot(weights[0],weights[1])
		
		# if pos_vec_line:
		# 	xfirst_inf_index = np.where(np.isinf(weights[0]))[0][0]
		# 	yfirst_inf_index = np.where(np.isinf(weights[1]))[0][0]
		# 	print()
		# 	plt.plot([0,0],[weights[0][-],weights[1][yfirst_inf_index-1]],linestyle='--',color='lightgray')
		
		color_sct = ['#883e03','#034f8c']
		if scatter_currents:
			plt.scatter(current_ds[:,0],current_ds[:,1], s=5,marker='o',color=color_sct[0], label=f"Presynaptic neuron currents")
	if limit_range:
		plt.xlim(-3,3)
		plt.ylim(-3,3)		    
	plt.xlabel('weight pre_syn 1')
	plt.ylabel('weight pre_syn 2')
	plt.title(f'{plt_title}')
	plt.legend()
	plt.grid(True)
	plt.savefig(f'{plt_title}_correct_2.png')
	print(f'plot "{plt_title}" saved')
	plt.clf()
	plt.close()


# Additional helper functions 

def generate_ds (s_x,s_y,theta,o,exp_num):
		# generates and saves the dataset of x,y pairs

		ds = np.array((np.random.normal(0,s_x,1000),np.random.normal(0,s_y,1000))).transpose()

		#does the rotation
		off_set = np.array((o,o)).transpose()
		rot_mtx = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
		ds = (ds @ rot_mtx.transpose())+off_set

		np.savetxt(f"./ds{exp_num}.txt",ds, delimiter="\t")	
		return ds

def extract_principal_components(ds):

		mean = np.mean(ds,axis=0)

		standardised_ds = ds - mean

		cov_matrix = np.cov(standardised_ds, rowvar=False)

		eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

		sorted_indices = np.argsort(eigenvalues)[::-1]

		sorted_eigenvectors = eigenvectors[:, sorted_indices]

		return sorted_eigenvectors


def run_simulation(synapses,time,step_size,s_x,s_y,theta,o,exp_num,q=None,limit_range=False):

		theta= np.radians(theta)
		
		try:
			ds=np.loadtxt(f"./ds{exp_num}.txt", delimiter="\t")
			print(f'dataset for experiment {exp_num} loaded')
		
		except FileNotFoundError:
			ds = generate_ds(s_x,s_y,theta,o,exp_num)
			print(f'dataset for experiment {exp_num} generated')

		ds_principal_components = extract_principal_components(ds)

		syn_01_weights = []
		syn_02_weights = []
		
		for x,y in ds:
			# updating currents and rate of presynaptics
			synapses[0].connection[0].current = x
			synapses[0].connection[0].update_rate()

			synapses[1].connection[0].current = y
			synapses[1].connection[0].update_rate()

			syn_01_weights.append(learn_rule(time,step_size,synapses[0]))
			syn_02_weights.append(learn_rule(time,step_size,synapses[1]))

		syn_01_weights=np.hstack(syn_01_weights)
		syn_02_weights=np.hstack(syn_02_weights)

		plot_weights(time=time,step_size=step_size,plt_title=f'experiment {exp_num} weight x vs. weight y',weights=[syn_01_weights,syn_02_weights],labels=['synapse','synapse'],current_ds=ds,pos_vec_line=True,scatter_currents=False,limit_range=limit_range)
		return syn_01_weights,syn_02_weights,ds_principal_components


def calculate_alignment(weights, pc1, pc2):
    # Ensure principal components are unit vectors
    pc1 = np.array(pc1) / np.linalg.norm(pc1)
    pc2 = np.array(pc2) / np.linalg.norm(pc2)
    
    # Calculate alignments
    for vec in weights.transpose():	
	    alignments_pc1=np.dot(vec, pc1) / np.linalg.norm(weights)
	    alignments_pc2=np.dot(vec, pc2) / np.linalg.norm(weights)
    
    # Compute mean and standard deviation for alignments
    return {
        'alignments_pc1': alignments_pc1,
        'alignments_pc2': alignments_pc2,
        'mean_alignment_pc1': np.mean(alignments_pc1),
        'mean_alignment_pc2': np.mean(alignments_pc2),
        'std_alignment_pc1': np.std(alignments_pc1),
        'std_alignment_pc2': np.std(alignments_pc2)
    }


if __name__ == '__main__':


	# parameters 
	np.random.seed(42)


	# # #Plain Hebb---------------------------------------------------------

	# heb_pre_syn_neuron = Neuron(current=0.5,rate=0)
	# heb_post_syn_neuron = Neuron(current=0.05,rate=0)
	# plain_hebb_01 = Plain_Hebb(weight=0,lr=0.1,connection=[heb_pre_syn_neuron,heb_post_syn_neuron])
	# plain_hebb_weights= learn_rule(100,0.1,plain_hebb_01)
	# plot_weights(100,0.1,'Plain hebb weigth update',weights=plain_hebb_weights,labels='dw/dt=mu*u*v')


	# #BCM Rule-------------------------------------------------------------
	# constant_currents = [0.2,0.4]
	
	# bcm_pre_syn_neuron = Neuron(current=0.5,rate=0)
	# bcm_post_syn_neurons = [Neuron(current=i,rate=0) for i in constant_currents]

	# # pre-syn current 0.2
	# bcm_01 = BCM(weight=0,lr=0.1,connection=[bcm_pre_syn_neuron,bcm_post_syn_neurons[0]])
	# bcm_01_weights = learn_rule(100,0.1,bcm_01,theta=0.3)
	# plot_weights(100,0.1,f'BCM rule weight update. pre_syn current={constant_currents[0]}',weights=bcm_01_weights,labels='dw/dt=mu*u*v*(u-theta)')

	# # pre-syn current 0.4
	# bcm_02 = BCM(weight=0,lr=0.1,connection=[bcm_pre_syn_neuron,bcm_post_syn_neurons[1]])
	# bcm_02_weights = learn_rule(100,0.1,bcm_02,theta=0.3)
	# plot_weights(100,0.1,f'BCM rule weight update. pre_syn current={constant_currents[1]}',weights=bcm_02_weights,labels='dw/dt=mu*u*v*(u-theta)')


	# #Oja Rule-------------------------------------------------------------
	# constant_currents = [0.5,0.7]
	# #two pre-synaptics
	# oja_pre_syn_neurons = [Neuron(current=i,rate=0) for i in constant_currents]
	# oja_post_syn_neuron = Neuron(current=0.1,rate=0)

	# oja_01 = Oja(weight=0,lr=0.1,connection=[oja_pre_syn_neurons[0],oja_post_syn_neuron])
	# oja_01_weights = learn_rule(100,0.1,oja_01,alpha=1)
	
	# plot_weights(100,0.1,f'Oja rule weight upadate. pre-syn with current={constant_currents[0]} ',weights=oja_01_weights,labels='dw/dt=mu*(u*v-(alpha*w*u^2))')

	# oja_02 = Oja(weight=0,lr=0.1,connection=[oja_pre_syn_neurons[1],oja_post_syn_neuron])
	# oja_02_weights = learn_rule(100,0.1,oja_02,alpha=1)
	# plot_weights(100,0.1,f'Oja rule weight upadate. pre-syn with current={constant_currents[1]}',weights=oja_02_weights,labels='dw/dt=mu*(u*v-(alpha*w*u^2))')


	# Exercise 1.6 part 01--------------------------------------------------
	

	# #experiment 1-----------------------------------------------------------	

	# instantiating the neurons and synapses
	# xp_6_1_post_syn_neuron = Neuron(current=0.1,rate=0)
	# xp_6_1_pre_syn_neuron_1 = Neuron(current=0,rate=0)
	# xp_6_1_pre_syn_neuron_2 = Neuron(current=0,rate=0)

	# xp_6_1_synapse_01 = Plain_Hebb(weight=0,lr=0.005,connection=[xp_6_1_pre_syn_neuron_1,xp_6_1_post_syn_neuron])
	# xp_6_1_synapse_02 = Plain_Hebb(weight=0,lr=0.005,connection=[xp_6_1_pre_syn_neuron_2,xp_6_1_post_syn_neuron])
	
	# time,step_size,s_x,s_y,theta,o,exp_num = 2,0.1,1,0.3,45,0,1

	# xp1_weights_x,xp1_weights_y,ds_principal_components = run_simulation(synapses=[xp_6_1_synapse_01,xp_6_1_synapse_02],time=time,step_size=step_size,s_x=s_x,s_y=s_y,theta=theta,o=o,exp_num=exp_num,limit_range=True)
	
	# # handle inf and -inf with nan. ignore nan and calculate mean vector as weight vector for alignment calculation
	# x_mean_nan = np.nanmean(np.where(np.isinf(xp1_weights_x),np.nan,xp1_weights_x))
	# y_mean_nan = np.nanmean(np.where(np.isinf(xp1_weights_y),np.nan,xp1_weights_y))
	# mean_vec =np.array((x_mean_nan,y_mean_nan))
	
	# stats=calculate_alignment(mean_vec, ds_principal_components[:,0], ds_principal_components[:,1])
	# print(stats)


	#experiment 2--------------------------------------------------------

	# xp_6_2_post_syn_neuron = Neuron(current=0.1,rate=0)
	# xp_6_2_pre_syn_neuron_1 = Neuron(current=0,rate=0)
	# xp_6_2_pre_syn_neuron_2 = Neuron(current=0,rate=0)

	# xp_6_2_synapse_01 = Plain_Hebb(weight=0,lr=0.005,connection=[xp_6_2_pre_syn_neuron_1,xp_6_2_post_syn_neuron])
	# xp_6_2_synapse_02 = Plain_Hebb(weight=0,lr=0.005,connection=[xp_6_2_pre_syn_neuron_2,xp_6_2_post_syn_neuron])
	
	# time,step_size,s_x,s_y,theta,o,exp_num = 2,0.1,1,0.3,20,0,2

	# xp2_weights_x,xp2_weights_y,ds_principal_components= run_simulation(synapses=[xp_6_2_synapse_01,xp_6_2_synapse_02],time=time,step_size=step_size,s_x=s_x,s_y=s_y,theta=theta,o=o,exp_num=exp_num,limit_range=True)


	# x_mean_nan = np.nanmean(np.where(np.isinf(xp2_weights_x),np.nan,xp2_weights_x))
	# y_mean_nan = np.nanmean(np.where(np.isinf(xp2_weights_y),np.nan,xp2_weights_y))
	# mean_vec =np.array((x_mean_nan,y_mean_nan))
	
	# stats=calculate_alignment(mean_vec, ds_principal_components[:,0], ds_principal_components[:,1])
	# print(stats)


	# #experiment 3--------------------------------------------------------

	# xp_6_3_post_syn_neuron = Neuron(current=0.1,rate=0)
	# xp_6_3_pre_syn_neuron_1 = Neuron(current=0,rate=0)
	# xp_6_3_pre_syn_neuron_2 = Neuron(current=0,rate=0)

	# xp_6_3_synapse_01 = Plain_Hebb(weight=0,lr=0.005,connection=[xp_6_3_pre_syn_neuron_1,xp_6_3_post_syn_neuron])
	# xp_6_3_synapse_02 = Plain_Hebb(weight=0,lr=0.005,connection=[xp_6_3_pre_syn_neuron_2,xp_6_3_post_syn_neuron])
	
	# time,step_size,s_x,s_y,theta,o,exp_num = 2,0.1,1,0.3,-45,2,3

	# xp3weights_x,xp3_weights_y,ds_principal_components= run_simulation(synapses=[xp_6_3_synapse_01,xp_6_3_synapse_02],time=time,step_size=step_size,s_x=s_x,s_y=s_y,theta=theta,o=o,exp_num=exp_num,limit_range=True)
	
	# x_mean_nan = np.nanmean(np.where(np.isinf(xp3weights_x),np.nan,xp3weights_x))
	# y_mean_nan = np.nanmean(np.where(np.isinf(xp3_weights_y),np.nan,xp3_weights_y))
	# mean_vec =np.array((x_mean_nan,y_mean_nan))

	# stats=calculate_alignment(mean_vec, ds_principal_components[:,0], ds_principal_components[:,1])
	# print(stats)


	# # # Exercise 1.6 part 02---------------------------------------------
	
	# xp_6_4_post_syn_neuron = Neuron(current=0.1,rate=0)
	# xp_6_4_pre_syn_neuron_1 = Neuron(current=0,rate=0)
	# xp_6_4_pre_syn_neuron_2 = Neuron(current=0,rate=0)

	# xp_6_4_synapse_01 = Covrule(weight=0,lr=0.005,q=0.1,connection=[xp_6_4_pre_syn_neuron_1,xp_6_4_post_syn_neuron])
	# xp_6_4_synapse_02 = Covrule(weight=0,lr=0.005,q=0.1,connection=[xp_6_4_pre_syn_neuron_2,xp_6_4_post_syn_neuron])
	
	# time,step_size,s_x,s_y,theta,o,exp_num = 2,0.1,1,0.3,-45,2,4

	# xp4weights_x,xp4weights_y,ds_principal_components=run_simulation(synapses=[xp_6_4_synapse_01,xp_6_4_synapse_02],time=time,step_size=step_size,s_x=s_x,s_y=s_y,theta=theta,o=o,exp_num=exp_num)

	# x_mean_nan = np.nanmean(np.where(np.isinf(xp4weights_x),np.nan,xp4weights_x))
	# y_mean_nan = np.nanmean(np.where(np.isinf(xp4weights_y),np.nan,xp4weights_y))
	# mean_vec =np.array((x_mean_nan,y_mean_nan))

	# stats=calculate_alignment(mean_vec, ds_principal_components[:,0], ds_principal_components[:,1])
	# print(stats)
