import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import math
from collections import defaultdict


# Custom Binary Vectorizer
class CustomBinaryVectorizer(CountVectorizer):
    def __init__(self, **kwargs):
        super().__init__(binary=True, **kwargs)

    def fit_transform(self, raw_documents, y=None):
        return super().fit_transform(raw_documents, y)


# Custom TFIDF Vectorizer
class CustomTFIDFVectorizer(CountVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df = None

    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents, y)
        self.df = np.diff(X.tocsc().indptr)
        self.df[self.df == 0] = 1  # Avoid division by zero
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        if self.df is None:
            raise ValueError("The fit method must be called before transform.")

        X = super().transform(raw_documents)
        tf = X.copy()

        # Compute TF-IDF
        for i, j in zip(*X.nonzero()):
            tf[i, j] = (1 + math.log(X[i, j])) / (1 + math.log(X[i].sum()))

        idf = np.reciprocal(self.df.astype(float))  # Custom IDF calculation
        tfidf = tf.multiply(idf)

        return tfidf


# Function to load the dataset
def loadCrabfieldDataset():
    with open('cran.all', 'r') as f:
        documents = f.read().split('.I ')[1:]
    with open('query.text', 'r') as f:
        queries = f.read().split('.I ')[1:]
    with open('qrels.text', 'r') as f:
        qrels = f.readlines()

    docs = []
    for doc in documents:
        docs.append(doc.split('.W\n')[1].strip())

    qrs = []
    for qr in queries:
        qrs.append(qr.split('.W\n')[1].strip())

    return docs, qrs, qrels


# Function to preprocess text
def preprocessText(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)


# Function to calculate cosine similarity and euclidean distance
def calculateSimilarities(vectorizer, distance_measure, docs, queries):
    X_docs = vectorizer.fit_transform(docs)
    X_queries = vectorizer.transform(queries)
    if distance_measure == 'cosine':
        return cosine_similarity(X_queries, X_docs)
    elif distance_measure == 'euclidean':
        return pairwise_distances(X_queries, X_docs, metric='euclidean')


# Function to get top-k indices
def get_top_k_indices(similarities, top_k=10):
    return np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]


# Function to calculate precision, recall, and F-score
def calculateMetrics(top_k_indices, qrel_dict):
    precision, recall, fscore = [], [], []
    for query_id, retrieved_docs in enumerate(top_k_indices):
        rel_docs = qrel_dict.get(query_id + 1, [])
        relevant_retrieved = [1 if doc in rel_docs else 0 for doc in retrieved_docs]
        p = sum(relevant_retrieved) / len(retrieved_docs)
        r = sum(relevant_retrieved) / len(rel_docs) if len(rel_docs) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precision.append(p)
        recall.append(r)
        fscore.append(f)
    return np.array(precision), np.array(recall), np.array(fscore)


# Function to display mean and max precision, recall, and F-score
def displayMetrics(precision, recall, fscore):
    meanP = np.mean(precision)
    maxP = np.max(precision)
    meanR = np.mean(recall)
    maxR = np.max(recall)
    meanF = np.mean(fscore)
    maxF = np.max(fscore)
    return {
        'p': (meanP, maxP),
        'r': (meanR, maxR),
        'f': (meanF, maxF)
    }


# Function to plot and save graphs
def plot_and_save(metric_values, metric_name, vectorizerType, distance_type):
    plt.bar(range(len(metric_values)), metric_values)
    plt.xlabel('Query Index')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} for {vectorizerType} vectorizer using {distance_type} distance')
    plt.savefig(f'{metric_name}_{vectorizerType}_{distance_type}.png')
    plt.clf()


# Load and preprocess data
docs, queries, qrels = loadCrabfieldDataset()
docs = [preprocessText(doc) for doc in docs]
queries = [preprocessText(query) for query in queries]

# Initialize dictionaries/lists to store top-k indices
top_k_indices = {
    'tfidf_cosine': np.zeros((len(queries), 10), dtype=int),
    'tfidf_euclidean': np.zeros((len(queries), 10), dtype=int),
    'binary_cosine': np.zeros((len(queries), 10), dtype=int),
    'binary_euclidean': np.zeros((len(queries), 10), dtype=int)
}

# Load qrels into a dictionary
qrel_dict = defaultdict(list)
for line in qrels:
    record = list(map(int, line.split()))
    qrel_dict[record[0]].append(record[1])

metricsSummary = {}

for model in ['tfidf', 'binary']:
    for measure in ['euclidean', 'cosine']:
        vectorizerType = f'Custom{model.capitalize()}Vectorizer'
        distance_type = measure

        if model == 'tfidf':
            vectorizer = CustomTFIDFVectorizer(binary=False, lowercase=True)
        elif model == 'binary':
            vectorizer = CustomBinaryVectorizer(stop_words='english', lowercase=True)

        similarities = calculateSimilarities(vectorizer, measure, docs, queries)
        top_k_indices[f'{model}_{measure}'] = get_top_k_indices(similarities)

# Extract top-k indices for each query
topKIndicesTFIDFCos = top_k_indices['tfidf_cosine']
topKIndicesTFIDFEuc = top_k_indices['tfidf_euclidean']
topKIndicesBinaryCos = top_k_indices['binary_cosine']
topKIndicesBinaryEuc = top_k_indices['binary_euclidean']

# Calculate metrics for each model-measure combination
precisionTFIDFCos, recallTFIDFCos, fscoreTFIDFCos = calculateMetrics(topKIndicesTFIDFCos, qrel_dict)
precisionTFIDFEuc, recallTFIDFEuc, fscoreTFIDFEuc = calculateMetrics(topKIndicesTFIDFEuc, qrel_dict)
precisionBinaryCos, recallBinaryCos, fscoreBinaryCos = calculateMetrics(topKIndicesBinaryCos, qrel_dict)
precisionBinaryEuc, recallBinaryEuc, fscoreBinaryEuc = calculateMetrics(topKIndicesBinaryEuc, qrel_dict)

# Display metrics for each model-measure combination and save in summary dictionary
metricsSummary['Binary'] = {
    'f': {
        'cos': displayMetrics(precisionBinaryCos, recallBinaryCos, fscoreBinaryCos),
        'euc': displayMetrics(precisionBinaryEuc, recallBinaryEuc, fscoreBinaryEuc)
    },
    'p': {
        'cos': displayMetrics(precisionBinaryCos, recallBinaryCos, fscoreBinaryCos),
        'euc': displayMetrics(precisionBinaryEuc, recallBinaryEuc, fscoreBinaryEuc)
    },
    'r': {
        'cos': displayMetrics(recallBinaryCos, precisionBinaryCos, fscoreBinaryCos),
        'euc': displayMetrics(recallBinaryEuc, precisionBinaryEuc, fscoreBinaryEuc)
    }
}

metricsSummary['TFIDF'] = {
    'f': {
        'cos': displayMetrics(precisionTFIDFCos, recallTFIDFCos, fscoreTFIDFCos),
        'euc': displayMetrics(precisionTFIDFEuc, recallTFIDFEuc, fscoreTFIDFEuc)
    },
    'p': {
        'cos': displayMetrics(precisionTFIDFCos, recallTFIDFCos, fscoreTFIDFCos),
        'euc': displayMetrics(precisionTFIDFEuc, recallTFIDFEuc, fscoreTFIDFEuc)
    },
    'r': {
        'cos': displayMetrics(recallTFIDFCos, precisionTFIDFCos, fscoreTFIDFCos),
        'euc': displayMetrics(recallTFIDFEuc, precisionTFIDFEuc, fscoreTFIDFEuc)
    }
}

# Format and print the metrics summary
for vectorizerType, metrics in metricsSummary.items():
    print(f"{vectorizerType}:")
    for metric_type, distances in metrics.items():
        print(f"  {metric_type}:")
        for distance_type, values in distances.items():
            print(f"    {distance_type}: (", end="")
            for key, (mean_val, max_val) in values.items():
                print(f"'{key}': ({mean_val:.3f}, {max_val:.1f}), ", end="")
            print(")")

# Plot and save metrics
plot_and_save(precisionTFIDFCos, 'precision', 'TFIDF', 'cosine')
plot_and_save(recallTFIDFCos, 'recall', 'TFIDF', 'cosine')
plot_and_save(fscoreTFIDFCos, 'fscore', 'TFIDF', 'cosine')

plot_and_save(precisionTFIDFEuc, 'precision', 'TFIDF', 'euclidean')
plot_and_save(recallTFIDFEuc, 'recall', 'TFIDF', 'euclidean')
plot_and_save(fscoreTFIDFEuc, 'fscore', 'TFIDF', 'euclidean')

plot_and_save(precisionBinaryCos, 'precision', 'Binary', 'cosine')
plot_and_save(recallBinaryCos, 'recall', 'Binary', 'cosine')
plot_and_save(fscoreBinaryCos, 'fscore', 'Binary', 'cosine')

plot_and_save(precisionBinaryEuc, 'precision', 'Binary', 'euclidean')
plot_and_save(recallBinaryEuc, 'recall', 'Binary', 'euclidean')
plot_and_save(fscoreBinaryEuc, 'fscore', 'Binary', 'euclidean')
