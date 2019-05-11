import gym
import random
import copy
import numpy as np
import itertools
import heapq
import time
from operator import attrgetter

class Population:
	def __init__(self):
		self.chromosomes = []
	def calculate_selection_probability(self):
		#First normalize the rewards to [0,1]
		normalized_reward = []
		min_rewards = float("inf")
		sum_rewards = 0
		for i in self.chromosomes:
			if i.fitness_value <= min_rewards:
				min_rewards = i.fitness_value
		for i in self.chromosomes:
			sum_rewards += i.fitness_value  + abs(min_rewards) + 1
		for i in self.chromosomes:
			normalized_reward.append((i.fitness_value + abs(min_rewards) + 1) / sum_rewards)
		#Then creates the distribution
		for i in range(len(normalized_reward)):
			self.chromosomes[i].set_selection_probability(normalized_reward[i])
	def sample_chromosome_pairs(self):
		#Create a list of additive probabilities in order to facilitate
		#the selection of the pair
		additive_probabilities = []
		partial_sum = 0
		for i in range(len(self.chromosomes)):
			partial_sum += self.chromosomes[i].selection_probability
			additive_probabilities.append(partial_sum)
		random_number_1 = random.uniform(0.0, 1.0)
		random_number_2 = random.uniform(0.0, 1.0)
		selected_chromosome_1 = -1
		selected_chromosome_2 = -1
		#Iterate through the lists to get the chromosomes
		for i in range(len(additive_probabilities)):
			if random_number_1 <= additive_probabilities[i]:
				selected_chromosome_1 = i
				break
		for i in range(len(additive_probabilities)):
			if random_number_2 <= additive_probabilities[i]:
				selected_chromosome_2 = i
				break
		return selected_chromosome_1, selected_chromosome_2
	def sample_chromosome_pairs_2(self, elite):
		#Create a list of additive probabilities in order to facilitate
		#the selection of the pair
		additive_probabilities = []
		partial_sum = 0
		for i in range(len(elite)):
			partial_sum += self.chromosomes[i].selection_probability
			additive_probabilities.append(partial_sum)
		random_number_1 = random.uniform(0.0, 1.0)
		random_number_2 = random.uniform(0.0, 1.0)
		selected_chromosome_1 = -1
		selected_chromosome_2 = -1
		#Iterate through the lists to get the chromosomes
		for i in range(len(additive_probabilities)):
			if random_number_1 <= additive_probabilities[i]:
				selected_chromosome_1 = i
				break
		for i in range(len(additive_probabilities)):
			if random_number_2 <= additive_probabilities[i]:
				selected_chromosome_2 = i
				break
		return selected_chromosome_1, selected_chromosome_2
	def playout(self, N_CHROMOSOMES, N_ACTIONS, env):
		for i in range(N_CHROMOSOMES):
			env.reset()
			sum_rewards = 0
			for _ in range(500//N_ACTIONS):
				for j in range(N_ACTIONS):
					action = self.chromosomes[i].actions[j]
					_, reward, _, _ = env.step(action)
					sum_rewards += reward
			self.chromosomes[i].calculate_fitness_value(sum_rewards)
	def mutation(self, N_ACTIONS,MUTATION_PERCENTAGE, N_MUTATIONS):
		for chromosome in self.chromosomes:
			will_it_mutate = random.randint(0, 100)
			if will_it_mutate >= MUTATION_PERCENTAGE*100:
				continue
			#generate the action indexes to be fully mutated
			mutation_indexes = random.sample(range(N_ACTIONS), N_MUTATIONS)
			for m in mutation_indexes:
				chromosome.actions[m] = env.action_space.sample()

	def crossover(self, N_ACTIONS, elite):
		new_population = []
		new_population.extend(elite)
		#Ignoring the elite, iterate until the end of the chromosomes
		
		for i in range(len(elite), len(self.chromosomes), 2):
			#Selection stage - Select two chromosomes for crossover (elite CAN be selected)
			#index_1, index_2 = self.sample_chromosome_pairs()
			#Crossover
			new_chromosome_1 = self.chromosomes[index_1].crossover_between_chromosomes(self.chromosomes[index_2], N_ACTIONS)
			new_chromosome_2 = self.chromosomes[index_2].crossover_between_chromosomes(self.chromosomes[index_1], N_ACTIONS)

			new_population.append(new_chromosome_1)
			new_population.append(new_chromosome_2)
		self.chromosomes = copy.deepcopy(new_population)
	def crossover_2(self, N_ACTIONS, elite):
		new_population = []
		new_population.extend(elite)
		#Ignoring the elite, iterate until the end of the chromosomes
		random_indexes = list(range(len(elite)))
		random.shuffle(random_indexes)
		for i in range(0, len(elite), 2):
			#Selection stage - Select two chromosomes for crossover (elite CAN be selected)
			#index_1, index_2 = self.sample_chromosome_pairs_2(elite)
			index_1 = random_indexes[i]
			index_2 = random_indexes[i+1]
			#print(index_1, index_2)
			#Crossover
			new_chromosome_1 = self.chromosomes[index_1].crossover_between_chromosomes(self.chromosomes[index_2], N_ACTIONS)
			new_chromosome_2 = self.chromosomes[index_2].crossover_between_chromosomes(self.chromosomes[index_1], N_ACTIONS)

			new_population.append(new_chromosome_1)
			new_population.append(new_chromosome_2)
		self.chromosomes = copy.deepcopy(new_population)

	def elitism(self, ELITISM_PERCENTAGE, N_CHROMOSOMES):
		return self.chromosomes[:int(N_CHROMOSOMES*ELITISM_PERCENTAGE)]

class Chromosome:
	def __init__(self, actions, N_ACTIONS):
		self.actions = copy.deepcopy(actions)
		self.selection_probability = 0
		self.fitness_value = 0
	def set_selection_probability(self, selection_probability):
		self.selection_probability = selection_probability
	def calculate_fitness_value(self, fitness_value):
		self.fitness_value = fitness_value
	def crossover_between_chromosomes(self, chromosome, N_ACTIONS):
		new_actions = []
		#75% of self and 25% of the other
		proportion = (3 * N_ACTIONS) // 4
		for i in range(proportion):
			aux = self.actions[i]
			new_actions.append(aux)
		for i in range(proportion, N_ACTIONS):
			aux = chromosome.actions[i]
			new_actions.append(aux)
		new_chromosome = Chromosome(new_actions, N_ACTIONS)
		return new_chromosome
	def imprime(self):
		print('Fitness value: ', self.fitness_value, ' Probability of being selected: %.15f' % self.selection_probability)

N_GENERATIONS = 100
N_CHROMOSOMES = 100 #Even number
N_ACTIONS = 40 #Has to be a number multiple of 4
MUTATION_PERCENTAGE = 0.50 #Chance of a chromosome to mutate
N_MUTATIONS = 1 #Number of mutations inside a chromosome - Has to be a value lower than N_ACTIONS
ELITISM_PERCENTAGE = 0.50 #Result of the multiplication of this value and N_CHROMOSOMES has to be even
env = gym.make('BipedalWalker-v2')
population = Population()
env.reset()
#Initial population is initialized randomly
for i in range(N_CHROMOSOMES):
	random_actions = []
	for _ in range(N_ACTIONS):
		random_actions.append(env.action_space.sample())
	chromosome = Chromosome(random_actions, N_ACTIONS)
	sum_rewards = 0
	for i in range(N_ACTIONS):
		_, reward, _, _ = env.step(chromosome.actions[i])
		sum_rewards += reward
	chromosome.calculate_fitness_value(sum_rewards)
	population.chromosomes.append(chromosome)
population.calculate_selection_probability()
env.reset()

#Main loop
for generation_counter in range(N_GENERATIONS):
	print('Geração', generation_counter+1)
	generation_counter += 1
	population.playout(N_CHROMOSOMES, N_ACTIONS, env)
	population.calculate_selection_probability()
	population.chromosomes.sort(key=lambda x: x.fitness_value, reverse=True)
	print('Imprimindo os três melhores')
	population.chromosomes[0].imprime()
	population.chromosomes[1].imprime()
	population.chromosomes[2].imprime()
	print('Imprimindo os três piores')
	population.chromosomes[47].imprime()
	population.chromosomes[48].imprime()
	population.chromosomes[49].imprime()
	#Elitism
	elite = population.elitism(ELITISM_PERCENTAGE, N_CHROMOSOMES)
	#Crossover
	population.crossover_2(N_ACTIONS, elite)
	#Mutation
	population.mutation(N_ACTIONS, MUTATION_PERCENTAGE, N_MUTATIONS)