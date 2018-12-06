import statistics
from  mlapi.utilities import invert_dictionary


class DiscriminatingFacetsAlgo(object):
    def __init__(self):
        self.min_documents_per_facet = 3
        self.max_standard_deviation = 25
        self. min_values_per_facet = 2

    def get_discriminating_facets(self, facets_by_document):
        documents_by_facet = invert_dictionary(facets_by_document)
        unique_documents_by_facet = self.get_unique_documents_by_facet(documents_by_facet)
        documents_by_discriminating_facets = self.execute_discriminating_facets_algorithm(unique_documents_by_facet)
        if len(self.get_values_by_facet(documents_by_discriminating_facets)) <= 1:
            self.adjust_parameters()
            unique_documents_by_facet.clear()
            documents_by_discriminating_facets.clear()
            unique_documents_by_facet = self.get_unique_documents_by_facet(documents_by_facet)
            documents_by_discriminating_facets = self.execute_discriminating_facets_algorithm(unique_documents_by_facet)
        return self.sort_discriminating_facets(documents_by_discriminating_facets)

    def get_unique_documents_by_facet(self, documents_by_facet):
        unique_documents_by_facet = {}
        unique_facets = []
        for facet, documents in documents_by_facet.items():
            unique_facet = (facet.name, facet.value)
            if unique_facet not in unique_facets:
                unique_documents_by_facet[unique_facet] = documents
                unique_facets.append(unique_facet)
            elif unique_facet in unique_facets:
                unique_documents_by_facet[unique_facet].append(documents)
        return unique_documents_by_facet

    def get_facet_names(self, unique_documents_by_facet):
        return list({facet[0] for facet in unique_documents_by_facet.keys()})

    def get_values_by_facet(self,unique_documents_by_facet):
        values_by_facet = {}
        unique_facets = []
        if unique_documents_by_facet:
            for facet, documents in unique_documents_by_facet.items():
                if facet[0] not in unique_facets:
                    values_by_facet[facet[0]] = [facet[1]]
                    unique_facets.append(facet[0])
                elif facet[0] in unique_facets:
                    values_by_facet[facet[0]].append(facet[1])
        return values_by_facet

    def get_documents_count_by_facet_name(self, unique_documents_by_facet):
        facet_names = self.get_facet_names(unique_documents_by_facet)
        documents_count_by_facet_name = {}
        for index in range(len(facet_names)):
            unique_documents = set()
            for facet, documents in unique_documents_by_facet.items():
                if facet[0] is facet_names[index]:
                    unique_documents.update(documents)
            documents_count_by_facet_name[facet_names[index]] = len(unique_documents)
        return documents_count_by_facet_name

    def get_facet_sample(self, unique_documents_by_facet):
        documents_count_by_facet_name = self.get_documents_count_by_facet_name(unique_documents_by_facet)

        return {facet: documents for (facet, documents) in unique_documents_by_facet.items()
                if (documents_count_by_facet_name[facet[0]] >= self.min_documents_per_facet)}

    def get_uniformly_distributed_facets(self, unique_documents_by_facet):
        standard_deviation = 50
        unique_facet_names = self.get_facet_names(unique_documents_by_facet)
        facet_values_by_facet_name = self.get_values_by_facet(unique_documents_by_facet)
        documents_count_by_facet_value = {facet: len(documents) for (facet, documents) in unique_documents_by_facet.items()}
        for unique_facet_name in unique_facet_names:
            counts = []
            for facet, nbr in documents_count_by_facet_value.items():
                if facet[0] == unique_facet_name:
                    counts.append(nbr)
            if len(counts)>= 2:
                standard_deviation = statistics.stdev(counts)
            if standard_deviation > self.max_standard_deviation:
                for facet_name, values in facet_values_by_facet_name.items():
                    if (facet_name == unique_facet_name):
                        for value in values:
                            unique_documents_by_facet.pop((unique_facet_name, value))
        return unique_documents_by_facet

    def get_facets_with_max_values(self, unique_documents_by_facet):
        facet_values_by_facet_name = self.get_values_by_facet(unique_documents_by_facet)
        return {facet: documents for (facet, documents) in unique_documents_by_facet.items()
                if len(facet_values_by_facet_name[facet[0]]) >= self.min_values_per_facet}

    def adjust_parameters(self):
        self.max_standard_deviation = self.max_standard_deviation + 10

    def execute_discriminating_facets_algorithm(self, unique_documents_by_facet):
        facets_sample = self.get_facet_sample(unique_documents_by_facet)
        uniformly_distributed_facets = self.get_uniformly_distributed_facets(facets_sample)
        return self.get_facets_with_max_values(uniformly_distributed_facets)

    def sort_discriminating_facets(self, unique_documents_by_facet):
        values_by_facet = self.get_values_by_facet(unique_documents_by_facet)
        documents_count_by_facet_name = self.get_documents_count_by_facet_name(unique_documents_by_facet)
        return sorted(values_by_facet.items(),
                      key=lambda values_by_facet : len(values_by_facet[1]) * documents_count_by_facet_name[values_by_facet[0]],
                      reverse= True)