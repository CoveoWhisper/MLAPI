class DocumentFilter(object):

    @staticmethod
    def keep_documents_with_facets(documents, must_have_facets):
        documents_with_facets = {}
        for document, facets in documents.items():
            if not set(must_have_facets.keys()).issubset([facet for facet in facets]):
                continue

            if any([set(facets[facet]).isdisjoint(must_have_facets[facet]) for facet in facets if facet in must_have_facets]):
                continue
            documents_with_facets[document] = facets

        return documents_with_facets

    @staticmethod
    def keep_documents_without_facets(documents, must_not_have_facets):
        documents_with_facets = {}
        for document, facets in documents.items():
            if any([set(facets[facet]).intersection(must_not_have_facets[facet]) for facet in facets if facet in must_not_have_facets]):
                continue
            documents_with_facets[document] = facets

        return documents_with_facets
