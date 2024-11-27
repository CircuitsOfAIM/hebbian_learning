# implementation of different synaptic update rules

from environment import Synapse

class Plain_Hebb(Synapse):
	#synapse with plain Hebb rule
	def __init__(self,*args, **kwargs):
		super().__init__(*args,**kwargs)

	def update_weight(self,dt=0):
		#check why put dt=0 ??? should be not
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

