import numpy as np

# Objective function: Example - Maximize f(x) = x^2; x in [-5, 5]
def objective_function(x):
    return x ** 2

# Genetic representation: Convert float to binary (chromosome)
def float_to_binary(x, lower_bound, upper_bound, bits=16):
    decimal_part = int((x - lower_bound) / (upper_bound - lower_bound) * ((2 ** bits) - 1))
    return format(decimal_part, f'0{bits}b')

# Convert binary (chromosome) to float
def binary_to_float(binary, lower_bound, upper_bound, bits=16):
    decimal_part = int(binary, 2)
    return lower_bound + decimal_part * (upper_bound - lower_bound) / ((2 ** bits) - 1)

# Create initial population
def create_population(pop_size, lower_bound, upper_bound, bits=16):
    return [float_to_binary(np.random.uniform(lower_bound, upper_bound), lower_bound, upper_bound, bits) for _ in range(pop_size)]

# Crossover
def crossover(parent1, parent2, crossover_rate=0.7):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

# Mutation
def mutate(chromosome, mutation_rate=0.01):
    mutated = ''.join(['1' if bit == '0' and np.random.rand() < mutation_rate else '0' if bit == '1' and np.random.rand() < mutation_rate else bit for bit in chromosome])
    return mutated

# Selection
def selection(population, fitnesses):
    # Roulette wheel selection
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return population[np.random.choice(range(len(population)), p=selection_probs)]

# Main GA loop
def genetic_algorithm(objective, pop_size, generations, lower_bound, upper_bound, bits=16, crossover_rate=0.7, mutation_rate=0.01):
    # Create initial population
    population = create_population(pop_size, lower_bound, upper_bound, bits)
    
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [objective(binary_to_float(chromosome, lower_bound, upper_bound, bits)) for chromosome in population]
        
        new_population = []
        for _ in range(pop_size // 2):
            # Selection
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            
            # Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population

    # Return the best solution
    fitnesses = [objective(binary_to_float(chromosome, lower_bound, upper_bound, bits)) for chromosome in population]
    best_index = np.argmax(fitnesses)
    return binary_to_float(population[best_index], lower_bound, upper_bound, bits), fitnesses[best_index]

# Parameters
pop_size = 100
generations = 100
lower_bound = -5
upper_bound = 5

# Run GA
best_solution, best_fitness = genetic_algorithm(objective_function, pop_size, generations, lower_bound, upper_bound)
print(f"Best Solution: x = {best_solution}, f(x) = {best_fitness}")
