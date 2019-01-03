import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from PreTrain import TrainingQueryPath

OutputPath = 'simmat'

def build_sim_matrix(query, document, w2v_model):
    matrix = np.zeros((len(query), len(document)))
    for i, w_q in enumerate(query):
        for j, w_d in enumerate(document):
            try:
                matrix[i][j] = w2v_model.similarity(w_q, w_d)
            except:
                matrix[i][j] = 0  # 如果字沒在model裡 則設為0

    return matrix

def create_sim_npy(query_filename, queries, documents, words, w2v_model):
    # Create dir
    if not os.path.isdir(OutputPath):
        os.makedirs(OutputPath)
    if not os.path.isdir(OutputPath + '/query_idf'):
        os.makedirs(OutputPath + '/query_idf')
    if not os.path.isdir(OutputPath + '/desc_doc_mat'):
        os.makedirs(OutputPath + '/desc_doc_mat')

    # Construct vectorizer
    vectorizer = TfidfVectorizer(
        use_idf=True,
        norm='l2',
        smooth_idf=False,
        sublinear_tf=False,  # tf = 1+ln(tf)
        binary=False,
        max_features=None,
        token_pattern=r"(?u)\b\w+\b"
    )

    # get word idf
    vectorizer.fit_transform(words)
    idf = vectorizer.idf_
    word2idf = dict(zip(vectorizer.get_feature_names(), idf))
    
    # get query solution
    solution = []
    with open('data/solution_Training.txt') as f:
        f.readline()
        solution = [line[line.find(',') + 1:].split() for line in f.readlines()]

    qid = 1
    for query in queries:
        # save qeury word idf
        query_idf = np.array(list(map(lambda x: word2idf[x], query)))
        print(query_idf.shape)
        np.save("%s/%s/%s.npy" % (OutputPath, 'query_idf', qid), query_idf)

        # create doc dir
        if not os.path.isdir("%s/%s/%d" % (OutputPath, 'desc_doc_mat', qid)):
            os.makedirs("%s/%s/%d" % (OutputPath, 'desc_doc_mat', qid))
        # if doc in the solution, build simmat and save
        for doc_id in solution[int(query_filename[qid - 1]) - 1]:
            if int(doc_id) >= len(documents): # solution doc does not exist
                continue
            doc = documents[int(doc_id)]
            sim_mat = build_sim_matrix(query, doc, w2v_model)
            np.save("%s/%s/%d/%s.npy" % (OutputPath, 'desc_doc_mat',qid, doc_id), sim_mat)
            print(sim_mat.shape)
                
        qid += 1



if __name__ == '__main__':
    from PreTrain import LoadingTrainingData, ModelPath
    import gensim
    word_index, query_filename, query_sentence, queries, document_filename, document_sentence, documents = LoadingTrainingData()
    w2v_model = gensim.models.Word2Vec.load(ModelPath + "word2vec.model")

    create_sim_npy(query_filename, query_sentence, document_sentence, word_index, w2v_model)

