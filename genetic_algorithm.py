import random
import numpy as np

import time


class GeneticAlgorithm:
    def __init__(
        self,
    ):  # More parameters as needed
        self.population_size = 30
        self.elite_rate = 0.3
        self.crossover_probability = 0.9
        self.mutation_probability = 0.05
        self.unchanged_gens = 0
        self.mutation_count = 0
        self.do_precise_mutate = True
        self.best_value = None
        self.best = []
        self.current_generation = 0
        self.current_best_index = None
        self.population = []
        self.values = []
        self.fitness_values = []
        self.roulette = []
        self.distances = []
        self.points = []

    def init(self, points: list[dict]):

        self.points = points

        self.count_distances()  # You'll need to implement this method

        # Reset population
        self.population = []
        for _ in range(self.population_size):
            self.population.append(self.random_individual(len(self.points)))

        self.compute_best_generation()

    def run_generation(self):
        self.current_generation += 1

        self.selection_step()  # You'll need to implement this method
        self.crossover_step()  # You'll need to implement this method
        self.mutation_step()  # You'll need to implement this method

        self.compute_best_generation()

        # Print best value
        print(f"Generation: {self.current_generation}, Best value: {self.best_value}")

    def selection_step(self) -> None:
        """
        Performs the selection process in the genetic algorithm.

        This method constructs a list of parents by applying various selection techniques.
        The selection process includes elitism, mutation, and roulette wheel selection.

        Returns:
            None

        """
        # Construct list of parents
        parents = []
        # Elitism - Add elites directly
        parents.append(self.population[self.current_best_index])
        parents.append(self.do_mutate(self.best.copy()))
        parents.append(self.push_mutate(self.best.copy()))
        # TODO: Why do we add the best solution twice
        parents.append(self.best.copy())

        # Roulette wheel selection
        self.update_roulette()

        # Start from index 4 because we already added the elites
        # This removes the 4 oldest individuals
        for _ in range(4, self.population_size):
            parents.append(self.population[self.sample_from_roulette()])

        self.population = parents

    def crossover_step(self):
        """
        Perform the crossover step of the genetic algorithm.

        This function selects individuals from the population for crossover based on a probability threshold.
        It shuffles the selected individuals and performs crossover between pairs of individuals.

        Crossover is a genetic operator that combines the genetic material of two individuals to create new offspring.
        In this implementation, the crossover operation is performed using the `crossover` method.

        Returns:
            None
        """

        # Create a list of individuals to be crossed over
        crossover_indices = []
        for i in range(self.population_size):
            # Sample to see if we should do crossover
            if np.random.rand() < self.crossover_probability:
                crossover_indices.append(i)

        # Shuffle the crossover indices, so that we cross random pairs
        np.random.shuffle(crossover_indices)
        for i in range(0, len(crossover_indices) - 1, 2):
            self.crossover(crossover_indices[i], crossover_indices[i + 1])
            # self.oxCrossover(queue[i], queue[i+1])

    def crossover(self, x: int, y: int):
        """
        Perform crossover between two parent solutions.

        Args:
            x (int): Index of the first parent solution in the population.
            y (int): Index of the second parent solution in the population.

        Returns:
            None
        """
        child1 = self.compute_child_forward(x, y)
        child2 = self.compute_child_backwards(x, y)

        # Replace the parents with the children
        self.population[x] = child1
        self.population[y] = child2

    def compute_child_forward(self, x: int, y: int) -> np.ndarray:
        return self._compute_child_helper(self._next, x, y)

    def compute_child_backwards(self, x: int, y: int) -> np.ndarray:
        return self._compute_child_helper(self._prev, x, y)

    def _compute_child_helper(self, iterate_fn: any, x: int, y: int) -> np.ndarray:
        """
        Computes the child solution by performing crossover between two parent solutions.

        Parameters:
        - iterate_fn (function): A function that iterates over the parent solution and returns the next point to consider.
        - x (int): Index of the first parent solution in the population.
        - y (int): Index of the second parent solution in the population.

        Returns:
        - solution (np.ndarray): The child solution obtained after crossover.

        Description:
        - This function performs crossover between two parent solutions to generate a child solution.
        - It starts by copying the parent solutions and selecting a random point to start the crossover.
        - It then iteratively selects the next point to add to the child solution based on the distance between the current point and the potential next points.
        - The process continues until all points are added to the child solution.
        - The child solution is returned as a numpy array.

        """

        # Copy the parents
        px = self.population[x].copy()  # np.ndarray
        py = self.population[y].copy()  # np.ndarray

        dx, dy = None, None
        # Select a random point to start the crossover
        px_index = random.randint(0, len(px) - 1)
        c = px[px_index]

        # Find the index of the selected point in the other parent, py (numpy array)
        solution = np.zeros(shape=px.shape, dtype=int)
        solution[0] = c
        # Loop until all points are added to the solution
        i = 1
        while len(px) > 1:
            px_index = np.where(px == c)[0][0]
            py_index = np.where(py == c)[0][0]

            dx = iterate_fn(px_index, px)
            dy = iterate_fn(py_index, py)

            # Remove the selected points from the parents
            px = np.delete(px, px_index)
            py = np.delete(py, py_index)

            # Choose the next point to add to the solution
            # Select the point that is closer to the current point\
            c = int(dx if self.distances[c][dx] < self.distances[c][dy] else dy)

            solution[i] = c
            i += 1

        return np.array(solution)

    def _next(self, index: int, array: np.ndarray) -> any:
        """Helper function to get the next element in the array

        Args:
            index (int): The index of the current element
            array (np.ndarray): The array of elements

        Returns:
            any: The next element in the array
        """
        if index >= len(array) - 1:
            return array[0]
        else:
            return array[index + 1]

    def _prev(self, index: int, array: np.ndarray) -> any:
        """Helper function to get the previous element in the array

        Args:
            index (int): The index of the current element
            array (np.ndarray): The array of elements

        Returns:
            any: The previous element in the array
        """
        if index == 0:
            return array[len(array) - 1]
        else:
            return array[index - 1]

    def mutation_step(self):
        """
        Applies mutation to the individuals in the population.

        This function iterates through each individual in the population and applies mutation
        with a certain probability. The mutation can be either a push mutation or a do mutation,
        determined randomly. The mutated individual replaces the original individual in the population.

        Parameters:
            None

        Returns:
            None

        Comments:
            - The function can mutate the same individual multiple times.
        """
        for i in range(self.population_size):
            # Decide whether to apply mutation
            if np.random.rand() < self.mutation_probability:
                # Decide whether to apply push mutation or do mutation
                if np.random.rand() > 0.5:
                    self.population[i] = self.push_mutate(self.population[i])
                else:
                    self.population[i] = self.do_mutate(self.population[i])

                # Decrement counter so that we can mutate the same individual again
                i -= 1

    # def preciseMutate(self, orseq):
    #     seq = orseq.copy()
    #     if random.random() > 0.5:
    #         seq.reverse()
    #     bestv = self.evaluate(seq)
    #     for i in range(len(seq) >> 1):
    #         for j in range(i + 2, len(seq) - 1):
    #             new_seq = self.swap_seq(seq, i, i + 1, j, j + 1)
    #             v = self.evaluate(new_seq)
    #             if v < bestv:
    #                 bestv = v
    #                 seq = new_seq
    #     return seq

    # def preciseMutate1(self, orseq):
    #     seq = orseq.copy()
    #     bestv = self.evaluate(seq)
    #     for i in range(len(seq) - 1):
    #         new_seq = seq.copy()
    #         new_seq.swap(i, i + 1)
    #         v = self.evaluate(new_seq)
    #         if v < bestv:
    #             bestv = v
    #             seq = new_seq
    #     return seq

    # def swap_seq(self, seq, p0, p1, q0, q1):
    #     seq1 = seq[:p0]
    #     seq2 = seq[p1 + 1:q1]
    #     seq2.append(seq[p0])
    #     seq2.append(seq[p1])
    #     seq3 = seq[q1:]
    #     return seq1 + seq2 + seq3

    def do_mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutates an individual by reversing a random subset of elements in the array.

        Parameters:
            individual (np.ndarray): The individual to be mutated.

        Returns:
            np.ndarray: The mutated individual.

        Comments:
            - The function updates the count of mutations.
            - It randomly selects two indices, m and n, such that m < n.
            - It swaps the elements between indices m and n in the individual array.
        """

        # Update count of mutations
        self.mutation_count += 1

        m, n = 1, 0
        while m >= n:
            m = np.random.randint(0, len(individual) - 2)
            n = np.random.randint(2, len(individual))
        for i in range((n - m + 1) >> 1):
            individual[m + i], individual[n - i] = individual[n - i], individual[m + i]

        return individual

    def push_mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Applies push mutation to an individual in the population.

        Args:
            individual (np.ndarray): The individual to be mutated.

        Returns:
            np.ndarray: The mutated individual.

        Raises:
            None

        Detailed Explanation:
        - The push mutation randomly selects two points, m and n, in the individual's genome.
        - It then pushes the subsequence between m and n to the beginning of the genome.
        - The resulting individual is returned as the mutated individual.
        """

        self.mutation_count += 1

        # Randomly select two points, m and n, in the individual's genome
        m, n = 1, 0
        while m >= n:
            m = np.random.randint(0, len(individual) >> 1)
            n = np.random.randint(0, len(individual))

        # Construct the mutated individual
        s1 = individual[:m]
        s2 = individual[m:n]
        s3 = individual[n:]

        return np.concatenate((s2, s1, s3))

    def compute_best_generation(self):
        """
        Computes the best generation in the population.

        This function evaluates each individual in the population and updates the stored best solution if a better solution is found.
        It also keeps track of the number of unchanged generations.

        Parameters:
        - None

        Returns:
        - None
        """
        # Evaluate each individual in the population
        self.values = [self.evaluate(individual) for individual in self.population]

        current_best_index, current_best_value = self.get_current_best()

        # Found a better solution
        if self.best_value is None or self.best_value > current_best_value:

            # Update the stored best solution
            self.current_best_index = current_best_index
            self.best = self.population[current_best_index].copy()
            self.best_value = current_best_value

            # Reset unchanged generations
            self.unchanged_gens = 0
        else:
            self.unchanged_gens += 1

    def get_current_best(self):
        """Get the current best individual in the population

        Returns:
            Tuple[np.ndarray, float]: The best individual and its value
        """

        best_index = np.argmin(self.values)
        best_value = self.values[best_index]

        return best_index, best_value

    def update_roulette(self):
        """
        Update the roulette wheel selection probabilities based on the fitness values.

        This method calculates the fitness values for each individual in the population,
        normalizes the fitness values, and calculates the cumulative sum of the normalized
        values to create the roulette wheel selection probabilities.

        Parameters:
            None

        Returns:
            None
        """

        self.fitness_values = [1.0 / value for value in self.values]

        # Calculate the roulette values
        sum = 0
        # First, calculate the sum of all fitness values
        for i in range(len(self.fitness_values)):
            sum += self.fitness_values[i]

        # Second, normalize the fitness values
        self.roulette = [
            self.fitness_values[i] / sum for i in range(len(self.fitness_values))
        ]

        # Third, calculate the cumulative sum of the roulette values
        for i in range(1, len(self.roulette)):
            self.roulette[i] += self.roulette[i - 1]

    def sample_from_roulette(
        self,
    ) -> int:
        """
        Select an individual from the population using roulette wheel selection.

        Returns:
            int: The index of the selected individual in the population.

        Comments:
            - The method uses the roulette wheel selection probabilities to select an individual.
            - It generates a random number between 0 and 1 and uses it to select an individual.
        """
        rand = np.random.rand()

        for i in range(len(self.roulette)):
            if rand <= self.roulette[i]:
                return i

    def random_individual(self, n: int) -> np.ndarray:
        a = np.arange(n)
        np.random.shuffle(a)
        return a

    def evaluate(self, individual: np.ndarray) -> float:
        # Calculate the distance between the points in the order of the individual
        # First, calculate the distance between the first point and 0
        sum = np.linalg.norm(
            self.points[individual[0]]["start_coord"] - np.array([0, 0])
        )

        # Then, calculate the distance between the rest of the points
        sum += np.sum(self.distances[individual[:-1], individual[1:]])

        return sum

    def count_distances(self):

        length = len(self.points)

        # Create a 2D array to store the distances between points
        # We can use this so that we don't have to calculate the distance between points multiple times
        self.distances = np.zeros((length, length))

        for i in range(length):
            for j in range(length):
                self.distances[i][j] = np.linalg.norm(
                    self.points[i]["start_coord"] - self.points[j]["end_coord"]
                )
