import random
import json

# Class definiing an invidiual within the population
# It has a chromosome and fitness_score
# It can crossover, mutate, and
class Individual:
    CHROMOSOME_COUNT = 11

    TEST_FILE = open('./data_saham.json')
    TEST_DATA = json.loads(TEST_FILE.read())['data']

    def __init__(self, *args):
        if len(args) == 0:
            self.chromosome = self.generate_chromosome()
        elif len(args) == 1:
            self.chromosome = args[0].get_chromosome()
        elif len(args) == 2:
            chromosome_choices = (
                args[0].get_chromosome(), args[1].get_chromosome())
            self.chromosome = list(
                chromosome_choices[random.randint(0, 1)][i] for i in range(Individual.CHROMOSOME_COUNT))

    # internal functions
    def generate_gene(self):
        return random.uniform(-1, 1)

    def generate_chromosome(self):
        return list(
            self.generate_gene() for _ in range(Individual.CHROMOSOME_COUNT))

    def calculate_fitness(self, verbose=False):
        deviation_sum = 0
        deviation_count = 0

        for i in range(0, len(Individual.TEST_DATA)-10):
            fx = self.chromosome[0]
            for j in range(0, 10):
                fx += self.chromosome[j+1] * Individual.TEST_DATA[j+i]

            target = Individual.TEST_DATA[j+i+1]
            deviation = abs(((fx - target) / target) * 100)
            deviation_count += 1
            deviation_sum += deviation

            if verbose:
                print('%d. Prediction: %.2f, Real: %.2f, Deviation: %.3f' %
                      (deviation_count, fx, target, deviation))

        fitness_score = deviation_sum / deviation_count
        if verbose:
            print('Total Test: %d, Average Deviation: %.3f' %
                  (deviation_count, fitness_score))

        return fitness_score

    def mutate(self, verbose=False):
        gene_to_mutate = random.randrange(0, Individual.CHROMOSOME_COUNT)
        self.chromosome[gene_to_mutate] = self.generate_gene()

    # getters
    def get_chromosome(self):
        return self.chromosome

    def get_fitness_score(self):
        if not hasattr(self, 'fitness_score'):
            self.fitness_score = self.calculate_fitness()

        return self.fitness_score

    def get_chromosome(self):
        return self.chromosome


class Population:
    def __init__(self, size):
        self.size = size
        self.generation_count = 0
        self.mutation_count = 0
        self.members = list(Individual() for _ in range(size))

        self.sort_population()

        self.best_members = []
        self.best_members.append(self.members[0])

    # internal function
    def sort_population(self):
        self.members = sorted(
            self.members, key=lambda i: i.get_fitness_score())
        self.isSorted = True

    # getters
    def get_size(self):
        return len(self.members)

    def get_generation_count(self):
        return self.generation_count

    def get_mutation_count(self):
        return self.mutation_count

    def get_members(self):
        return self.members

    def get_member(self, index):
        return self.members[index]

    def get_best_member(self):
        return self.best_members[-1]

    def get_best_members(self):
        return self.best_members

    # GA Functions
    def iterate_generation(self, verbose=False):
        if verbose:
            print('Interating Generation %d | Current Best : %.3f' % (
                self.get_generation_count(),
                self.get_best_member().get_fitness_score(),
            )
            )

        parents = random.sample(self.members, self.size)

        while(len(parents) > 0):
            p1 = parents.pop()
            p2 = parents.pop()
            self.members.append(Individual(p1, p2))

        self.sort_population()
        self.members = self.members[0:self.size]

        if random.randint(0, 100) < 5:
            self.get_member(
                random.randrange(0, self.size)
            ).mutate()
            self.mutation_count += 1

        if self.members[0] != self.best_members[-1]:
            self.best_members.append(self.members[0])

        self.generation_count += 1


if __name__ == '__main__':
    POPULATION_SIZE = 1000
    GENERATION_COUNT = 20000

    print('Running GA with population size %d over %d generations...' %
          (POPULATION_SIZE, GENERATION_COUNT))

    population = Population(POPULATION_SIZE)

    for i in range(GENERATION_COUNT):
        population.iterate_generation(verbose=True)

    # REPORT
    best_members = population.get_best_members()
    best_member = best_members[-1]

    print()
    print('-----------')
    print('Population Summary')
    print('Mutation count       : %d' % (population.get_mutation_count()))
    print('Best member changes  : %d' % (len(best_members)))
    print('Best member history  :')

    for i in range(len(best_members)):
        print('%d. Fitness Score: %.3f, Chromosome: %s' % (
            i+1,
            best_members[i].get_fitness_score(),
            best_members[i].get_chromosome())
        )

    print()
    print('-----------')
    print('Final Best Member')

    print('Chromosome:')
    print(best_member.get_chromosome())

    print('Test Result:')
    best_member.calculate_fitness(verbose=True)