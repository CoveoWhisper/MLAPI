from mlapi.utilities import invert_dictionary


class FacetDictionary(object):

    @staticmethod
    def create_facet_dict(facets_by_document):
        facets_inverted = invert_dictionary(facets_by_document)

        facets = dict()
        for key in facets_inverted:
            name = key.name
            value = key.value

            if name in facets:
                facets[name].append(value)
            else:
                facets.update({name: [value]})

        for key in facets:
            facets[key] = sorted(facets.get(key), key=lambda s: s.casefold())

        return dict(sorted(facets.items()))
