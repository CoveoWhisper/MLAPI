class DocumentFilter(object):

    @staticmethod
    def keep_documents_with_facets(documents, must_have_facets):
        documents_with_facets = {}
        for document, facets in documents.items():
            if any([facet.value not in must_have_facets[facet.name] for facet in facets if facet.name in must_have_facets]):
                continue
            documents_with_facets[document] = facets

        return documents_with_facets

    @staticmethod
    def keep_documents_without_facets(documents, must_not_have_facets):
        documents_with_facets = {}
        for document, facets in documents.items():
            if not all([facet.value not in must_not_have_facets[facet.name] for facet in facets if facet.name in must_not_have_facets]):
                continue
            documents_with_facets[document] = facets

        return documents_with_facets
