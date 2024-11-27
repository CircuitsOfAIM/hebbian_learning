#implementatino of the environment objects classes

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
		#connection [neruon1,...,neuronn]
		self.connection = connection
		self.weight = weight
		self.lr = lr
		self.set_connection(self.connection)

	def update_weight(self):
		pass
	
	def set_connection(self,connection):
		for neuron in connection:
			neuron.synapses.append(self)
	
