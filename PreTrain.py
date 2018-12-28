import sys
import os
import numpy as np
import time
import gensim
from sklearn.metrics.pairwise import cosine_similarity

ModelPath = "./Model/"
TrainingQueryPath = "./data/Query_Training/"
DocumentPath = "./data/100000Doc/"

LoadAllWord = True

LoadWord2VecModel = True

Load_Embedding_Matrix = False

# Word2Vec param
Word2Vec_size = 300
Word2Vec_window = 10
Word2Vec_worker = 1000
Word2Vec_iter = 100

def ConsoleLog(msg):
    print("[", time.strftime('%H:%M:%S', time.localtime(time.time())), "] " + msg)

def Transform(corpus, w2v_model):
    new_corpus = []
    for doc in corpus:
        new_doc = np.zeros(Word2Vec_size, dtype=float) # w2v size
        count = 0
        for word in doc:
            try:
                new_doc += w2v_model[word]
                count += 1
            except:
                pass
        new_doc = new_doc / count
        new_corpus.append(new_doc)
    return np.array(new_corpus)

def LoadingTrainingData():
    word = []
    query_filename = []
    query_sentence = []
    document_filename = []
    document_sentence = []
    documents = []
    queries = []

    for filename in os.listdir(TrainingQueryPath):
        query_filename += [filename]
        with open(TrainingQueryPath + filename) as f:
            query = ' '.join([word for line in f.readlines() for word in line.split()[:-1]])
            queries.append(query)
            query_sentence += [query.split()]
            word += query.split()

    for filename in os.listdir(DocumentPath):
        document_filename += [filename]
        with open(DocumentPath + filename) as f:
            doc = ' '.join([word for line in f.readlines()[3:] for word in line.split()[:-1]])
            documents.append(doc)
            document_sentence += [doc.split()]
            word += doc.split()
    
    if(LoadAllWord):
        word = np.load(ModelPath + 'all_words.npy')
    else:
        word = list(set(word))
    
    return word, query_filename, query_sentence, queries, document_filename, document_sentence, documents


if __name__ == '__main__':
    # 讀取query, document等data
    ConsoleLog("Loading Training Data")
    # query_filename : query的檔案名稱list
    # query_sentence : query將每個檔案分成一層list，再將每個list裡面的字分成list (string of list of list)
    # queries :　query將每個檔案分成一層list　(string of list)
    # document_filename : document的檔案名稱list
    # document_sentence : document將每個檔案分成一層list，再將每個list裡面的字分成list (string of list of list)
    # documents : document將每個檔案分成一層list　(string of list)
    word_index, query_filename, query_sentence, queries, document_filename, document_sentence, documents = LoadingTrainingData()
    if(LoadAllWord == False):
        np.save(ModelPath + 'all_words', word_index)

    ConsoleLog("Word2Vec")
    w2v_model = None
    if(LoadWord2VecModel):
        w2v_model = gensim.models.Word2Vec.load(ModelPath + "word2vec.model")
    else:
        train_sentences = document_sentence + query_sentence
        w2v_model = gensim.models.Word2Vec(sentences = train_sentences, min_count = 1, 
                            size = Word2Vec_size, window = Word2Vec_window, workers = Word2Vec_worker, iter = Word2Vec_iter, hs = 0)
        w2v_model.save(ModelPath + "word2vec.model")

    ConsoleLog("embeddings")
    if(Load_Embedding_Matrix):
        embedding_matrix = np.load(ModelPath + 'embedding_matrix.npy')
    else:
        embeddings_index = {}
        word_vector = w2v_model.wv
        for word,vocab_obj in w2v_model.wv.vocab.items():
                embeddings_index[word] = word_vector[word]
        
        embedding_matrix = np.zeros((len(word_index) + 1, w2v_model.vector_size))
        for i in range(len(word_index)):
            embedding_vector = embeddings_index.get(word_index[i])
            if (embedding_vector is not None):
                embedding_matrix[i] = embedding_vector
        np.save(ModelPath + 'embedding_matrix', embedding_matrix)
