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
        self.extracted_documents = []

    # cette fonction fait le lien entre les 3 recommandeurs
    def get_recommended_documents(self, query, uris):
        recommended_documents = []
        scores = []
        return self.get_json_response(recommended_documents, scores)

    def get_json_response(self, recommended_documents, scores):
        return [
            {
                "Document": {
                    "Uri": recommended_documents[index],
                    "Title": ''
                },
                "Score": scores[index]
            }
            for index in range(len(recommended_documents))
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
                    self.extracted_documents.append(uris[index])
        return  extracted_documents

    def get_k_means_cluster_documents(self, labels, urls):
        query_cluster = labels[0]
        documents_clusters= labels[1:]
        related_documents = [urls[index] for index in range(len(documents_clusters))
                             if (documents_clusters[index]==query_cluster) and (urls[index] in self.extracted_documents)]
        return related_documents


    def get_k_means_query_cluster_documents(self, query, uris):
        extracted_documents = self.extract_documents(uris)
        parsed_documents = {index : parseText(text) for index, text in extracted_documents.items()}

        query = parseText(query)
        predicting_data = []
        predicting_data.append(query)
        predicting_data.extend(parsed_documents.values())
        tf_idf_matrix_test = self.tfidf_vectorizer.transform(predicting_data)
        labels = self.k_means.predict(tf_idf_matrix_test)
        print('all clusters : ', labels.tolist())
        cluster_documents = self.get_k_means_cluster_documents(labels, uris)
        print('cluster documents : ', cluster_documents)
        return cluster_documents

    def get_cosine_simillar_documents(self, query, uris):
        extracted_documents = self.extract_documents(uris)
        parsed_documents = {index : parseText(text) for index, text in extracted_documents.items()}

        query = parseText(query)
        predicting_data = []
        predicting_data.append(query)
        predicting_data.extend(parsed_documents.values())
        tf_idf_matrix = self.tfidf_vectorizer.transform(predicting_data)
        print(tf_idf_matrix)
        cosine_similarities = linear_kernel(tf_idf_matrix[0:1], tf_idf_matrix).flatten()
        print(cosine_similarities * 100)
        return cosine_similarities

    def get_unsupervised_knn_neighbors(self, query, uris):
        extracted_documents = self.extract_documents(uris)
        parsed_documents = {index : parseText(text) for index, text in extracted_documents.items()}

        query = parseText(query)
        print(query)
        predicting_data = []
        # predicting_data.append(query)
        predicting_data.extend(parsed_documents.values())
        tf_idf_matrix = self.tfidf_vectorizer.transform(predicting_data)

        nbrs = NearestNeighbors(n_neighbors=2)
        nbrs.fit(tf_idf_matrix)
        distances, indices = nbrs.kneighbors(self.tfidf_vectorizer.transform([query]), 2)
        print('indices of document : ', indices)
        print('distance : ', distances)

        return indices