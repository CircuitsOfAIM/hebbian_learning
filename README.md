## Objective
The primary objective of this tutorial was to explore how different Hebbian learning rules produce correlations between pre-synaptic and post-synaptic activity rates. Additionally, basic synchronization of these synaptic weights (correlations) based on characteristics of their probability distributions was evaluated using Principal Component Analysis.

## 1.1 Simulation Environment
Neuron and Synapse classes serve as the simulation environment.

- **Neuron Class:**
  - Each `Neuron` object has a `rate`, `current`, and a list of `synapses` it shares with other neurons.
  - Implements an `update_rate()` method, which updates the rate of the neuron based on the weights and current it acquires.

- **Synapse Class:**
  - Each `Synapse` object has a `weight`, `learning rate`, and `connection` (a two-element list of pre-synaptic and post-synaptic neuron objects that contribute to the synapse in order).
  - Implements:
    - `update_weight()` method as a polymorphic method in an inherited `Synapse` class.
    - `set_connection()` method that explicitly assigns neurons to the connection list.

---

## 1.2 Plain Hebbian Learning with Fixed Activities
The **Plain Hebbian learning rule** was implemented as an `update_weight()` method in the class `PlainHebb`. This rule combines the input rate (pre-synaptic rate) with the post-synaptic rate using a learning rate constant.

### Simulation Setup:
- Pre-synaptic neuron current: **0.5**
- Post-synaptic neuron current: **0.05**
- Simulation time: **100 seconds**
- Learning rate (μ): **0.1**
- Time step (∆t): **0.1**

### Results
The figure below shows how the synapse weight updates on each learning step during the simulation time:

![Alt Text]("Plain hebb weigth update_correct_2.png")

**Figure 1:** Time development of the weight over 100 seconds with Plain Hebbian  

As indicated, there is a positive correlation between the activity rates of pre-synaptic and post-synaptic neurons. The synaptic weight grows exponentially as time progresses.

---

## 1.3 Bienenstock-Cooper-Munro Rule
This learning rule incorporates an additional term to Plain Hebb, `(rate_i − θ)`, which acts as a threshold for strengthening weights. For example:
- When the post-synaptic rate is below the threshold or zero, the term becomes negative, reducing the weight update.

### Results
**Figure 2:** Weight update by BCM with post-synaptic current 0.2  
**Figure 3:** Weight update by BCM with post-synaptic current 0.4  

The learning rule prolonged the weight update period compared to the plain Hebbian rule. The post-synaptic current value reinforces the learning rule, where higher currents increase the speed of positive weight updating.

---

## 1.4 Oja’s Rule
This rule subtracts a term `(- α ⋅ weight_{ij} ⋅ rate_i^2)` from Plain Hebb, which acts as a weight decay, reducing the synaptic weights over time in response to excessive growth.

### Results
**Figure 4:** Weight update with Oja rule with current = 0.5  
**Figure 5:** Weight update with Oja rule with current = 0.7  

Oja’s rule shows a linear behavior compared to other rules, taking much longer to reach a certain weight. Higher current values cause the weight decay term to become more effective, meaning weight development takes significantly longer with a current of 0.7 compared to 0.5.

---

## 1.5 Random Number Generation
A `generate_dataset()` helper function was implemented to generate a dataset with:
- **1000 data points**
- **2 features (x and y)** representing currents for two pre-synaptic neurons.

### Results
- The mean is close to zero as generated.
- For a random seed = 42:
  - The directions correspond to the principal components of the covariance of these distributions.
  - The component with maximal variance is the principal component with the highest eigenvalue of the covariance matrix.
  - The component with minimal variance has the lowest.

---

## 1.6 Principal Component Analysis
### Part 1: Simulation with Plain Hebbian Learning
This simulation investigates how two distinct synapses from two pre-synaptic neurons, which share a post-synaptic neuron, relate to each other based on their current distributions. These currents, generated randomly in the previous exercise, undergo various transformations to demonstrate different relationships between weights.

#### Experiments:
- **Experiment 1:** `sx = 1`, `sy = 0.3`, `ϕ = 45°`, `o = 0`  
  **Figure 6:** Weight update against each other. Currents scattered. Plot is limited to weight range -0.5 to 0.5  

  For these parameters, a negative correlation between the two synapses is observed, such that they inhibit each other.

- **Experiment 2:** `sx = 1`, `sy = 0.3`, `ϕ = 20°`, `o = 0`  
  **Figure 7:** Weight update against each other. Currents scattered. Plot is limited to weight range -0.5 to 0.5  

  Little anti-clockwise twist is observed compared to the previous experiment. Despite the negative correlation, the weights also show smoother strength in updates.

- **Experiment 3:** `sx = 1`, `sy = 0.3`, `ϕ = -45°`, `o = 2`  
  **Figure 8:** Weight update against each other. Currents scattered. Plot is limited to weight range -0.5 to 0.5  

  A -45-degree rotation on the dataset and shift results in learned data showing a **positive correlation** with a high degree of strength. This is inferred as the weight vector aligns highly with the principal components of the covariance matrix of the random currents.

### Part 2: Covariance Rule
The covariance rule applies the covariance of each neuron’s rate as the learning rule.

#### Results
Weights **x correlation with y** is much more positive. Small changes in x cause significant updates in weight y, indicating that the current of pre-synaptic neuron x has more effect on the synapse.
