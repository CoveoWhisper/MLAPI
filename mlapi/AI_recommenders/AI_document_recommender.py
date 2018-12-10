from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pickle
from mlapi.AI_recommenders.text_data_mining_utilities import parseText
from mlapi.extractors.factory import ExtractorFactory
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors

HEADERS = {'Authorization': 'Bearer xx50034238-5a30-4808-98d3-4ef3dc9ec7cc'}

class DocumentRecommender(object):
    def __init__(self, tf_idf_vectorizer, k_means_clustering_model):
        self.tfidf_vectorizer = tf_idf_vectorizer
        self.k_means = k_means_clustering_model
        self.extracted_uris = []

    def get_recommended_documents(self, query, document_uris):
        extracted_documents = self.extract_documents(document_uris)
        parsed_documents = {index: parseText(text) for index, text in extracted_documents.items()}
        print(self.extracted_uris)

        # linear kernel similarity
        similarity_documents = self.get_linear_kernel_simillar_documents(query, document_uris, parsed_documents)
        print('similarity recommender :',similarity_documents)
        # unsupervised knn
        neighbors_documents = self.get_unsupervised_knn_neighbors(query, document_uris, parsed_documents)
        print('neighbors recommender :', neighbors_documents)

        # k_means
        k_means_documents = self.get_k_means_query_cluster_documents(query, document_uris, parsed_documents)
        print('k means recommender : ',k_means_documents)
        # final scores
        final_recommended_documents = self.get_final_scores(k_means_documents, neighbors_documents, similarity_documents)
        print(final_recommended_documents)

        return self.get_json_response(final_recommended_documents)


    def get_json_response(self, recommended_documents):
        return [
            {
                "Document": {
                    "Uri": document,
                    "Title": ''
                },
                "Score": score
            }
            for document, score in recommended_documents
        ]

    def extract_documents(self, uris):
        factory = ExtractorFactory()
        extracted_documents = {}
        for index in range(len(uris)):
            extractor = factory.fromFilePath(uris[index], 'html', HEADERS)
            if extractor != None:
                extracted_text = extractor.extractAllText()
                if extracted_text:
                    extracted_documents[index] = extracted_text
                    self.extracted_uris.append(uris[index])
        return  extracted_documents

#**************************************************** calculate scores *****************************************
    def get_k_means_cluster_documents_with_score(self, labels, document_uris):
        query_cluster = labels[0]
        documents_clusters= labels[1:]
        recommended_documents = [(document_uris[index], 1) for index in range(len(documents_clusters))
                                 if (documents_clusters[index]==query_cluster) and (document_uris[index] in self.extracted_uris)]
        return recommended_documents

    def get_linear_kernel_similarity_documents_with_scores(self, linear_kernel_similarities, document_uris):
        scores = linear_kernel_similarities[1:]
        recommended_documents = [(document_uris[index], 1) for index in range(len(scores))
                                 if (document_uris[index] in self.extracted_uris) and scores[index]]
        return recommended_documents

    def get_unsupervised_knn_neighbors_with_scores(self, indices, document_uris):
        indices = indices.tolist()[0]
        print('knn indices :', indices)
        print('knn indice 0 :', (indices[0]+1)*0.1)
        return [(document_uris[index], (len(indices)-index) * 0.1) for index in range(len(indices)) if
                document_uris[index] in self.extracted_uris]

    def unify_scores(self, document_scores):
        all_document_scores = {document_uri:score for document_uri, score in document_scores}
        for uri in self.extracted_uris:
            if uri not in all_document_scores.keys():
                all_document_scores[uri] = 0
        return all_document_scores

    def get_final_scores(self, k_means_documents, neighbors_documents, similarity_documents):
        final_recommendation = [(document_uri, (k_means_documents[document_uri]*0.1)+ (neighbors_documents[document_uri]*0.5) + (similarity_documents[document_uri]*0.4))
                                for document_uri in self.extracted_uris]
        return sorted(final_recommendation, key=lambda x:x[1], reverse=True)

#******************************************************* The 3 recommenders ********************************************
    def get_k_means_query_cluster_documents(self, query, document_uris, parsed_documents):

        query = parseText(query)
        predicting_data = []
        predicting_data.append(query)
        predicting_data.extend(parsed_documents.values())
        tf_idf_matrix = self.tfidf_vectorizer.transform(predicting_data)
        labels = self.k_means.predict(tf_idf_matrix)
        print(labels)
        recommended_documents = self.get_k_means_cluster_documents_with_score(labels, document_uris)
        return self.unify_scores(recommended_documents)

    def get_linear_kernel_simillar_documents(self, query, document_uris, parsed_documents):
        query = parseText(query)
        predicting_data = []
        predicting_data.append(query)
        predicting_data.extend(parsed_documents.values())
        tf_idf_matrix = self.tfidf_vectorizer.transform(predicting_data)
        linear_kernel_similarities = linear_kernel(tf_idf_matrix[0:1], tf_idf_matrix).flatten()
        recommended_documents = self.get_linear_kernel_similarity_documents_with_scores(linear_kernel_similarities, document_uris)

        return self.unify_scores(recommended_documents)

    def get_unsupervised_knn_neighbors(self, query, document_uris, parsed_documents):
        query = parseText(query)

        tf_idf_matrix = self.tfidf_vectorizer.transform(parsed_documents.values())

        neighbors = NearestNeighbors(n_neighbors=10)
        neighbors.fit(tf_idf_matrix)
        distances, indices = neighbors.kneighbors(self.tfidf_vectorizer.transform([query]), 10)

        recommended_documents = self.get_unsupervised_knn_neighbors_with_scores(indices, document_uris)

        return self.unify_scores(recommended_documents)