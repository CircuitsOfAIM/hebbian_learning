# Additional helper functions 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)



def generate_ds (s_x,s_y,theta,o,exp_num):
	# generates or loading dataset and saves the dataset of x,y pairs
	try:
		ds=np.loadtxt(f"./ds{exp_num}.txt", delimiter="\t")
		print(f'dataset for experiment {exp_num} loaded')
	
	except FileNotFoundError:
		print(f'dataset for experiment {exp_num} generated')
		ds = np.array((np.random.normal(0,s_x,1000),np.random.normal(0,s_y,1000))).transpose()

		#does the rotation
		theta = np.radians(theta)
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


def run_simulation(time,step_size,synapse,verbose=False,**kwargs):
	'''applies learning rule on the list of synapses (can be different) for the given time and step size
	args:
		time > float
		step_size > float
		synapse > list
		verbose > bool
		**kwargs > additional arguments for the learning rule

	returns > a dataframe of : time, synapse, weight
	'''

	weights={}

	for syn in synapse:
		weights[f'weight_syn_{id(syn)}'] = []

	for step in np.arange(0, time, step_size):
		for syn in synapse:
			if verbose:
				print(f'time>{step:8f}\nsynapse>{syn.__class__.__name__}\nrate_j>{syn.connection[0].rate}\nrate_i>{syn.connection[1].rate}\nweight_ij>{syn.weight}')
				print('--------------')
			
			syn.update_weight(step_size,**kwargs)
			syn.connection[1].update_rate()

			#storing weights for plotting
			weights[f'weight_syn_{id(syn)}'].append(syn.weight)
		
		# Convert weights to DataFrame
		df = pd.DataFrame(weights)
	return df


def plot_weights(time_axis,weights_df,plt_title,labels,legends=False,w_trajectory=False,current_ds=None,pos_vec_line=False,scatter_currents=False,limit_range=3,verbose=True):
	'''plots the weights of the single or multiple synapses either as time series or trajectory'''
	
	plt.figure()
	# plotting a single plot
	for i,col in enumerate(weights_df.columns):
		plt.plot(time_axis, weights_df[col], label=legends[i] if legends else None)
		
	if w_trajectory:
		#plotting trajectory of weights
		plt.clf()
		plt.plot(weights_df.iloc[:, 0],weights_df.iloc[:, 1])
		
		if pos_vec_line:
			xfirst_inf_index = np.where(np.isinf(weights[0]))[0][0]
			yfirst_inf_index = np.where(np.isinf(weights[1]))[0][0]
			plt.plot([0,0],[weights[0][xfirst_inf_index-1],weights[1][yfirst_inf_index-1]],linestyle='--',color='lightgray')

		color_sct = ['#883e03','#034f8c']

		if scatter_currents:
			# fig, axs = plt.subplots(1,2, figsize=(6, 4))
			# axs[0].plot(weights[0],weights[1])
			# axs[1].scatter(current_ds[:,0], current_ds[:,1], s=5,marker='o',color=color_sct[0], label=f"Presynaptic neuron currents")
			# for ax in axs:
			# 	ax.set_aspect('equal')			
			plt.scatter(current_ds[:,0], current_ds[:,1], s=5,marker='o',color=color_sct[0], label=f"Presynaptic neuron currents")
			
	if limit_range:
		plt.xlim(-limit_range,limit_range)
		plt.ylim(-limit_range,limit_range)

	plt.xlabel(labels[0])
	plt.ylabel(labels[1])
	# #gca=GET CURRENT AXES aspect
	# plt.gca().set_aspect('equal')
	plt.title(f'{plt_title}')
	plt.legend()
	plt.grid(True)
	plt.savefig(f'{plt_title}.png')
	
	if verbose:
		print(f'plot "{plt_title}" saved')
	
	plt.clf()
	plt.close()
