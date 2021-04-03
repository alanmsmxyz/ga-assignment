import random
import json

import os.path
from os import path, walk

# CONFIGURE HERE
INPUT_DIR = './input'
OUTPUT_DIR = './output'

CHROMOSOME_LENGTH = 11

POPULATION_SIZE = 400
GENERATION_COUNT = 10000

MUTATION_PERCENTAGE = 25


# GeneticAlgorithm class
# Used to intilize genetic algorithm processing
# And also save it's result to a JSON file
class GeneticAlgorithm:
    # Constructior, initalize files used for the genetic algorithm
    # @param input_file     - path to a JSON file containing data for fitness testing
    # @param output_file    - path to JSON file that will be used to save the result to a JSON file
    # @param log_file       - optional, path to a text file that will contains the log of GA process
    def __init__(self, input_file, output_file, log_file=None):
        assert path.isfile(input_file)

        data_file = open(input_file, "r")
        self.test_data = json.loads(data_file.read())["data"]
        data_file.close()

        self.output_file = output_file

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.f_log = open(log_file, "w")

            self.logger = lambda entry: self.f_log.write("%s\n" % (entry))
        else:
            self.logger = lambda *args: None

        self.population = None

        self.logger("__init__: input file %s, output file %s" %
                    (input_file, output_file))

    # Destructor, used to cleanup file handles used during the GA process
    def __del__(self):
        self.logger("__del__: cleaning up file handler...\n")
        if hasattr(self, "log_file"):
            self.f_log.close()

    # Run the GA process
    # @params generation_count - number of generation to iterate
    def run(self, population, generation, mutation_percentage):
        self.population = Population(population, mutation_percentage,
                                     self.fitness_tester, self.logger)

        for _ in range(generation):
            self.population.iterate_generation()

    # Return a JSON containing GA result
    def toJSON(self):
        if not self.population:
            print("Please run the instance first!")

        result = {
            "solution": {
                "chromosome":
                self.population.get_best_member().get_chromosome(),
                "fitness_report":
                self.population.get_best_member().get_fitness_report(),
            },
            "population": {
                "size": self.population.get_size(),
                "generation_count": self.population.get_generation_count(),
                "mutation_percentage":
                self.population.get_mutation_percentage(),
                "mutation_count": self.population.get_mutation_count(),
                "best_members": [],
            },
        }

        best_members = self.population.get_best_members()
        for i in range(len(best_members)):
            result["population"]["best_members"].append({
                "fitness_score":
                best_members[i].get_fitness_score(),
                "chromosome":
                best_members[i].get_chromosome(),
            })

        return result

    # Write the GA result into the output file
    def write_output(self):
        if not self.population:
            print("Please run the instance first!")

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        f_output = open(self.output_file, 'w')

        json.dump(self.toJSON(), f_output, indent=2)

        f_output.close()

    # Test the individual fitness scores
    # Fitness score is defined as average percentage of deviation from the target value
    # Lesser fitness scores means *better* individual
    # @returns - a dictionary containing the test report
    def fitness_tester(self, individual):
        test_score = 0
        test_results = []

        deviation_sum = 0
        deviation_count = 0

        chromosome = individual.get_chromosome()

        for i in range(0, len(self.test_data) - 10):
            fx = chromosome[0]
            for j in range(0, 10):
                fx += chromosome[j + 1] * self.test_data[j + i]

            target = self.test_data[j + i + 1]

            deviation = abs(((fx - target) / target) * 100)

            deviation_count += 1
            deviation_sum += deviation

            test_results.append({
                "prediction": fx,
                "target": target,
                "deviation": deviation
            })

        test_score = deviation_sum / deviation_count

        return {"score": test_score, "test_result": test_results}


# Individual Class
# Used to model an individual within population.
# Each individual will have a chromosome
class Individual:
    # Constructor, initialize individual based on parameters given
    # @param parents        - optional, will use parents crossover chromosome if supplied
    # @param chromosome     - optional, will use this chromosome if supplied
    # @param fitness_tester - function to test fitness scores supplied by GeneticAlgorithm
    def __init__(self,
                 fitness_tester,
                 chromosome=None,
                 parents=None,
                 population=None):
        # if parents is supplied, derived chromosome from the parents using crossover
        if parents:
            self.__construct_from_crossover(parents)

        # if chromosome is supplied, use the supplied chromosome
        elif chromosome:
            self.__construct_from_chromosome(chromosome)

        # if nothing is supplied, generate chromosome randomly
        else:
            self.__construct_from_random()

        self.fitness_tester = fitness_tester
        self.fitness_report = self.fitness_tester(self)

    # Called by constructor to intialize with random chromosome
    def __construct_from_random(self):
        self.chromosome = self.generate_chromosome()

    # Called by constructor to intialize with supplied
    def __construct_from_chromosome(self, chromosome):
        self.chromosome = chromosome

    # Called by constructor to intialize with crossover
    def __construct_from_crossover(self, parents):
        # determine which gene to choose from which parents
        # if chromosome_selector[gene_index] is 0, chromosome[gene_index] will derived the from parent1
        # and vice versa for parent2
        chromosome_selector = [
            random.randint(0, 1) for i in range(CHROMOSOME_LENGTH)
        ]

        # create new chromosome based on the chromosome_selector
        self.chromosome = [
            parents[chromosome_selector[i]][i] for i in chromosome_selector
        ]

    # Generate a random gene, a float between -1 and 1
    def generate_gene(self):
        return random.uniform(-1, 1)

    # Generate a chromosome, consisting of configured length
    def generate_chromosome(self):
        return [self.generate_gene() for _ in range(CHROMOSOME_LENGTH)]

    # Mutate a single gene in the chromosome
    # Randomly select a gene within the chromosome, and change it with new random value
    def mutate(self):
        gene_to_mutate = random.randrange(0, CHROMOSOME_LENGTH)
        self.chromosome[gene_to_mutate] = self.generate_gene()

        # Regenerate fitness report after mutation occurs
        self.fitness_report = self.fitness_tester(self)

    # Getter functions
    def get_chromosome(self):
        return self.chromosome

    def get_fitness_report(self):
        return self.fitness_report

    def get_fitness_score(self):
        return self.fitness_report["score"]


# Population Class
# Used to manage Individuals as it's members
class Population:
    # Intialize population
    # @param size - number of members within the population
    def __init__(self, size, mutation_percentage, fitness_tester, logger):
        self.size = size
        self.mutation_percentage = mutation_percentage

        self.generation_count = 0
        self.mutation_count = 0

        self.fitness_tester = fitness_tester
        self.logger = logger

        # generate population members and sort it from best members
        members_temp = [
            Individual(fitness_tester=fitness_tester) for _ in range(size)
        ]

        self.members = sorted(members_temp,
                              key=lambda i: i.get_fitness_score())

        # add best members from intial generation to best_members dictionary
        # best_members dictionary is used to keep track of best_member changes within the population
        self.best_members = []
        self.set_best_member(self.members[0])

    # Since the population is always kept at certain size due to elimination
    # Parent selection will only create a randomize copy of current population members
    def parents_selection(self):
        return random.sample(self.members, self.size)

    # Members selection choose which members to keep for next generation
    # This keep population size constant by eliminating worst members of the population
    def members_selection(self):
        return self.members[0:self.size]

    # Add a new member into population
    # New member will be inserted into certain positions according to its fitness score
    # This makes the members list sorted
    def add_member(self, new_member):
        pos = len(self.members)
        while (new_member.get_fitness_score() <
               self.members[pos - 1].get_fitness_score() and pos > 0):
            pos -= 1
        self.members.insert(pos, new_member)

    # Set Best Member
    # The best member is saved as a new instance of supplied member in best_members list
    # @param member - member set as best_member
    def set_best_member(self, member):
        self.best_members.append(
            Individual(self.fitness_tester,
                       chromosome=member.get_chromosome()))

    # Iterate generation of the population
    # The process consists of:
    # - Parents selection
    # - Breeding with crossover
    # - Mutation
    # - Members selection / elimination
    # - Best member selection
    def iterate_generation(self, verbose=False):
        self.logger(
            "iterate_generation: generation %d, best score %.3f, best chromosome %s"
            % (
                self.get_generation_count(),
                self.get_best_member().get_fitness_score(),
                self.get_best_member().get_chromosome(),
            ))

        # Select parents for the breeding process
        parents = self.parents_selection()

        # Breed the selected parents to generate offsprings Individual
        while len(parents) > 0:
            p1 = parents.pop().get_chromosome()
            p2 = parents.pop().get_chromosome()
            child1 = Individual(parents=(p1, p2),
                                fitness_tester=self.fitness_tester)
            child2 = Individual(parents=(p1, p2),
                                fitness_tester=self.fitness_tester)

            self.add_member(child1)
            self.add_member(child2)

        # Randomly select a member within the population to mutate within the configured probability
        # Best member (current solution) are protected from mutation to avoid setbacks
        if random.randint(0, 100) < self.mutation_percentage:
            # remove member that will be mutated from the members list
            index_to_mutate = random.randrange(1, self.size)
            member_to_mutate = self.members.pop(index_to_mutate)

            # mutate the selected member, then add it back to members list
            member_to_mutate.mutate()
            self.add_member(member_to_mutate)

            self.mutation_count += 1

        # Eliminate worst member excess of population size
        self.members = self.members_selection()

        # Select best members from the population
        if self.members[0].get_chromosome(
        ) != self.best_members[-1].get_chromosome():
            self.set_best_member(self.members[0])

        # Keep track of the generations count
        self.generation_count += 1

    # Getter Functions
    def get_size(self):
        return len(self.members)

    def get_generation_count(self):
        return self.generation_count

    def get_mutation_percentage(self):
        return self.mutation_percentage

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


if __name__ == "__main__":
    _, _, input_files = next(walk(INPUT_DIR))

    for i in range(len(input_files)):
        filename = input_files[i]
        name, ext = path.splitext(filename)

        if ext == '.json':
            print("Processing file %s..." % (filename))

            ga = GeneticAlgorithm(path.join(INPUT_DIR, filename),
                                  path.join(OUTPUT_DIR, filename),
                                  path.join(OUTPUT_DIR, "%s.log" % (name)))
            ga.run(
                population=POPULATION_SIZE,
                generation=GENERATION_COUNT,
                mutation_percentage=MUTATION_PERCENTAGE,
            )

            ga.write_output()

            del ga
        else:
            print("Skipping file %s..." % (filename))
