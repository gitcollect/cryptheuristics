import numpy as np
import random as rn

from bisect import bisect


class CandidateSolution(object):

    _mutation_probability = 0.03
    _crossover_probability = 0.97

    @property
    def mutation_probability(self):
        return self.__class__._mutation_probability

    @mutation_probability.setter
    def mutation_probability(self, val):
        self.__class__._mutation_probability = val

    @property
    def crossover_probability(self):
        return self.__class__._crossover_probability

    @crossover_probability.setter
    def crossover_probability(self, val):
        self.__class__._crossover_probability = val

    @property
    def solution(self):
        return self._solution

    @property
    def fitness(self):

        if self._fitness is None:
            self.evaluate()

        return self._fitness

    def __init__(self, **kwargs):

        if 'mutation_probability' in kwargs:
            self.mutation_probability = kwargs['mutation_probability']
        if 'crossover_probability' in kwargs:
            self.crossover_probability = kwargs['crossover_probability']

        self._fitness = None

    def evaluate(self):
        raise NotImplementedError("This method should be implemented in child "
                                  "class.")

    def mutate(self):
        raise NotImplementedError("This method should be implemented in child "
                                  "class.")

    def crossover(self, candidate_solution):
        raise NotImplementedError("This method should be implemented in child "
                                  "class.")

    def crossover_mutate_evaluate(self, candidate_solution):

        child_x, child_y = self.crossover(candidate_solution)

        child_x.mutate()
        child_y.mutate()

        child_x.evaluate()
        child_y.evaluate()

        return child_x, child_y


class PermutationCandidateSolution(CandidateSolution):

    def __init__(self, size, solution=None, **kwargs):

        super(PermutationCandidateSolution, self).__init__(**kwargs)

        self.size = size
        self._solution = self.spawn() if solution is None else solution

    def spawn(self):
        return list(np.random.permutation(self.size))

    def mutate(self):
        ix, iy = np.random.randint(self.size, size=2)
        self.solution[ix], self.solution[iy] = \
            self.solution[iy], self.solution[ix]

    def evaluate(self):
        raise NotImplementedError("This method should be implemented in child "
                                  "class.")

    def crossover(self, candidate_solution):
        raise NotImplementedError("This method should be implemented in child "
                                  "class.")


class CXCandidateSolution(PermutationCandidateSolution):

    def crossover(self, candidate_solution):

        solution_x, solution_y = self.solution, candidate_solution.solution
        indexes = - np.ones(len(solution_x), dtype=int)

        start = solution_x[0]
        index = 0
        indexes[index] = 0
        while not solution_y[index] == start:
            index = solution_x.index(solution_y[index])
            indexes[index] = 0

        child_x = [solution_y[i] if not val else solution_x[i]
                   for i, val in enumerate(indexes)]
        child_y = [solution_x[i] if not val else solution_y[i]
                   for i, val in enumerate(indexes)]

        return CXCandidateSolution(self.size, solution=child_x), \
            CXCandidateSolution(self.size, solution=child_y)

    def evaluate(self):
        raise NotImplementedError("This method should be implemented in child "
                                  "class.")


class OXCandidateSolution(PermutationCandidateSolution):

    def crossover(self, candidate_solution):

        solution_x, solution_y = self.solution, candidate_solution.solution
        ix, iy = np.random.randint(len(solution_x), size=2)
        ix, iy = (iy, ix) if ix > iy else (ix, iy)

        child_x = solution_x[ix:] + solution_x[:ix]
        child_y = solution_y[ix:] + solution_y[:ix]
        child_x = [x for x in child_x if x not in solution_y[ix:iy]]
        child_y = [x for x in child_y if x not in solution_x[ix:iy]]
        child_x = child_x[:ix] + solution_y[ix:iy] + child_x[ix:]
        child_y = child_y[:ix] + solution_x[ix:iy] + child_y[ix:]

        return OXCandidateSolution(self.size, solution=child_x), \
            OXCandidateSolution(self.size, solution=child_y)

    def evaluate(self):
        raise NotImplementedError("This method should be implemented in child "
                                  "class.")


class PMXCandidateSolution(PermutationCandidateSolution):

    def crossover(self, candidate_solution):

        solution_x, solution_y = self.solution, candidate_solution.solution
        ix, iy = np.random.randint(len(solution_x), size=2)
        ix, iy = (iy, ix) if ix > iy else (ix, iy)

        child_x, child_y = solution_x[:], solution_y[:]

        for i in range(ix, iy):
            index_x = child_x.index(solution_y[i])
            child_x[i], child_x[index_x] = child_x[index_x], child_x[i]
            index_y = child_y.index(solution_x[i])
            child_y[i], child_y[index_y] = child_y[index_y], child_y[i]

        return PMXCandidateSolution(self.size, solution=child_x), \
            PMXCandidateSolution(self.size, solution=child_y)

    def evaluate(self):
        raise NotImplementedError("This method should be implemented in child "
                                  "class.")


class GABase(object):

    def __init__(self, solution_class):
        self.solution_class = solution_class

    def selection(self):
        raise NotImplementedError("This method should be implemented in child "
                                  "class.")


class RouletteWheelSelection:

    def selection(self, population, size):
        """
        Select pool of solutions to breed using roulette wheel selection
        (selecting each solution sequentially with probability proportional to
        its fitness value).

        :param size: amount of solutions to be generated
        :type size: int
        :return: breed pool
        :rtype: iter of solutions
        """

        total_fitness = sum((x[1] for x in population))
        cumdist = np.cumsum([x[1] for x in population])
        return (population[bisect(cumdist, rn.random() * total_fitness)][0]
                for _ in range(size))


class TruncationSelection:

    def truncation_selection(self, population, size):
        """
        Select pool of solutions to breed choosing **at most** *size* best
        solutions.

        :param size: amount of solutions to be generated
        :type size: int
        :return: breed pool
        :rtype: iter of solutions
        """

        return (list(x) for x in np.random.permutation([x[0]
                for x in population[:size]]))


class TournamentSelection:

    _tournament_probability = 0.85
    _tournament_size = 2

    @property
    def tournament_probability(self):
        return self._tournament_probability

    @tournament_probability.setter
    def tournament_probability(self, val):
        self._tournament_probability = val

    @property
    def tournament_size(self):
        return self._tournament_size

    @tournament_size.setter
    def tournament_size(self, val):
        self._tournament_size = val

    def __init__(self, **kwargs):

        if 'tournament_probability' in kwargs:
            self.tournament_probability = kwargs['tournament_probability']
        if 'tournament_size' in kwargs:
            self.tournament_size = kwargs['tournament_size']

    def tournament_selection(self, population, size):
        """
        Select pool of solutions to breed using tournament selection rule.

        :param size: amount of solutions to be generated
        :type size: int
        :return: breed pool
        :rtype: iter of solutions
        """

        return iter([self.tournament(population) for _ in range(size)])

    def tournament(self, population):
        """
        Perform single tournament selecting solution from population using
        tournament selection rule.

        :return: solution from population
        :rtype: solution
        """

        if self.tournament_size == 2:
            solution_a, solution_b = rn.sample(population, 2)
            if solution_b[1] >= solution_a[1]:
                solution_a, solution_b = solution_b, solution_a
            if rn.random() < self.tournament_probability:
                return solution_a[0]
            else:
                return solution_b[0]

        indexes = rn.sample(range(len(population)), self.tournament_size)
        candidates = [solution[0] for index, solution
                      in enumerate(population)
                      if index in indexes]

        for candidate in candidates:
            if rn.random() < self.tournament_probability:
                return candidate
        return candidates[-1]


class StochasticUniversal_Sampling:

    def stochastic_universal_sampling(self, population, size):
        """
        Select pool of solutions to breed using stochastic universal sampling
        rule.

        :param size: amount of solutions to be generated
        :type size: int
        :return: breed pool
        :rtype: iter of solutions
        """

        total_fitness = sum([x[1] for x in population])
        cumdist = np.cumsum([x[1] for x in population])
        distance = total_fitness / size
        start = rn.random() * distance
        pointers = [start + i * distance for i in range(size)]
        indexes = [bisect(cumdist, pointer) for pointer in pointers]

        return (list(x) for x in np.random.permutation([population[idx][0]
                for idx in indexes]))