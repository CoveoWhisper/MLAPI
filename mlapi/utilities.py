def invert_dictionary(facets_by_document):
    documents_by_facet = {}
    for k, v in facets_by_document.items():
        for value in v:
            documents_by_facet[value] = documents_by_facet.get(value, [])
            documents_by_facet[value].append(k)
    return documents_by_facet