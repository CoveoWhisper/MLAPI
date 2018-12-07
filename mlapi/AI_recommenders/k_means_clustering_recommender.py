from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pickle
from mlapi.AI_recommenders.text_data_mining_utilities import parseText
from mlapi.extractors.factory import ExtractorFactory

HEADERS = {'Authorization': 'Bearer xx50034238-5a30-4808-98d3-4ef3dc9ec7cc'}

class KmeansClusteringRecommender(object):
    def __init__(self, tf_idf_vectorizer, k_means_clustering_model):
        self.tfidf_vectorizer = tf_idf_vectorizer
        self.k_means = k_means_clustering_model

    def extract_documents(self, urls):
        factory = ExtractorFactory()
        extracted_documents = {}
        for index in range(len(urls)):
            extractor = factory.fromFilePath(urls[index], 'html', HEADERS)
            if extractor != None:
                extracted_text = extractor.extractAllText()
                if extracted_text:
                    extracted_documents[index] = extracted_text
        return  extracted_documents

    def get_related_documents(self, labels, urls):
        query_cluster = labels[0]
        documents_clusters= labels[1:]
        related_documents = [urls[index] for index in range(len(documents_clusters)) if documents_clusters[index]==query_cluster]
        return related_documents

    def get_suggested_documents(related_documents, scores):
        return [
            {
                "Document": {
                    "Uri": related_documents[index],
                    "Title": ''
                },
                "Score": scores[index]
            }
            for index in range(len(related_documents))

        ]

    def get_recommended_documents(self, query, urls):
        extracted_documents = self.extract_documents(urls)
        parsed_documents = {index : parseText(text) for index, text in extracted_documents.items()}

        query = 'Hello, can I push analytics to the organization?'
        query = parseText(query)
        predicting_data = []
        predicting_data.append(query)
        predicting_data.extend(parsed_documents.values())
        tf_idf_matrix_test = self.tfidf_vectorizer.transform(predicting_data)
        labels = self.k_means.predict(tf_idf_matrix_test)
        print('all clusters : ', labels.tolist())
        related_documents = self.get_related_documents(labels, urls)
        scores = [1 for document in related_documents]
        return self.get_suggested_documents(related_documents, scores)


