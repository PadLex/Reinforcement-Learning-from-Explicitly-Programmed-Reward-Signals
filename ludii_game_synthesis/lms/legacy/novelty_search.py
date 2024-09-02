class Archive:

    def __init__(self, individuals=[], fitnesses=[], behavior_characteristics=[], k: int = 5):
        self.individuals = individuals
        self.fitnesses = fitnesses
        self.behavior_characteristics = behavior_characteristics
        self.k = k

    def sample_individual(self):
        num_individuals = len(self.individuals)
        idx = np.random.randint(num_individuals)
        return self.individuals[idx]

    def add(self, individual, fitness, behavior_characteristic):
        self.individuals.append(individual)
        self.fitnesses.append(fitness)
        self.behavior_characteristics.append(behavior_characteristic)

    def dist(self, query = None):
        import torch

        stacked = torch.stack(self.behavior_characteristics).view(len(self.behavior_characteristics), -1)

        if query is None:
            query = stacked

        dist = torch.cdist(query, stacked, p=2)
        return dist

    def topk(self, query, k: Optional[int] = None):
        if k is None:
            k = self.k
        dist = self.dist(query)
        return [self.indivudals[idx] for idx in dist.topk(k, largest=False).indices.squeeze()], dist.topk(k, largest=False).values.squeeze().mean(-1).item()

@dataclass
class Fitness:
    behavior_characteristic: Any
    fitness: float

class NoveltySearch:
    def __init__(
        self,
        minimum_criteria: float,
        population_size: int = 100,
        k: int = 5
    ):
        self.minimum_criteria = minimum_criteria
        self.archive = self.initialize_population(population_size)
        self.k = k

    def initialize_population(self, population_size: int):
        """Return initialized archive

        Args:
            population_size (int): _description_
        """
        raise NotImplementedError("Not implemented yet!")

    def mutate(self, individual):
        raise NotImplementedError("Mutate function is not implemented!")

    def evaluate(self, individual: str) -> Fitness:
        raise NotImplementedError("Evaluate function is not implemented!")

    def evolve(self, epochs: int):
        for _ in range(epochs):
            sampled_individual = self.archive.sample_individual()
            mutated_individual = self.mutate(sampled_individual)
            fitness_obj = self.evaluate(mutated_individual)

            behavior_characteristic = fitness_obj.behavior_characteristic
            fitness = fitness_obj.fitness

            _, distance = self.archive.topk(mutated_individual, k=self.k)
            if distance < self.minimum_criteria:
                self.archive.add(mutated_individual, fitness, behavior_characteristic)