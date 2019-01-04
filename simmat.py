import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from PreTrain import TrainingQueryPath
from itertools import chain
import threading

OutputPath = 'simmat'

# 子執行緒類別
class MyThread(threading.Thread):
  def __init__(self, min, max):
    threading.Thread.__init__(self)
    self.min = min
    self.max = max
    self.w2v_model = gensim.models.Word2Vec.load(ModelPath + "word2vec.model")

  def run(self):
    create_sim_npy(self.w2v_model, self.min, self.max)

def build_sim_matrix(query, document, w2v_model):
    matrix = np.zeros((len(query), len(document)))
    for i, w_q in enumerate(query):
        for j, w_d in enumerate(document):
            try:
                matrix[i][j] = w2v_model.similarity(w_q, w_d)
            except:
                matrix[i][j] = 0  # 如果字沒在model裡 則設為0

    return matrix

def create_sim_npy(w2v_model, min, max):
    print('start', min, max)

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
    vectorizer.fit_transform(word_index)
    idf = vectorizer.idf_
    word2idf = dict(zip(vectorizer.get_feature_names(), idf))
    
    # get query solution
    solution = []
    with open('data/solution_Training.txt') as f:
        f.readline()
        solution = [line[line.find(',') + 1:].split() for line in f.readlines()]

    qid = 1
    for query in query_sentence:
        if(qid < min):
            qid += 1
            continue
        # save qeury word idf
        query_idf = np.array(list(map(lambda x: word2idf[x], query)))
        print(query_idf.shape)
        np.save("%s/%s/%s.npy" % (OutputPath, 'query_idf', qid), query_idf)

        # create doc dir
        if not os.path.isdir("%s/%s/%d" % (OutputPath, 'desc_doc_mat', qid)):
            os.makedirs("%s/%s/%d" % (OutputPath, 'desc_doc_mat', qid))
        #for s in range(len(solution)):
        # if doc in the solution, build simmat and save
        all_doc = set(list(chain.from_iterable(solution)))
        for doc_id in all_doc:
            if doc_id not in document_filename: # solution doc does not exist
                continue
            label = 0
            if doc_id in solution[int(query_filename[qid - 1]) - 1]:
                label = 1
            doc = document_sentence[int(document_filename.index(doc_id))]
            sim_mat = build_sim_matrix(query, doc, w2v_model)
            np.save("%s/%s/%d/%s_%d.npy" % (OutputPath, 'desc_doc_mat',qid, doc_id, label), sim_mat)
            print(sim_mat.shape)
                
        qid += 1
        if qid > max:
            break
    print('finish', min, max)



from PreTrain import LoadingTrainingData, ModelPath
import gensim
word_index, query_filename, query_sentence, queries, document_filename, document_sentence, documents = LoadingTrainingData()
# w2v_model = gensim.models.Word2Vec.load(ModelPath + "word2vec.model")

threads = []
for i in range(1, 150,5):
    print(i, i + 4)
    threads.append(MyThread(i, i + 4))

for i in range(len(threads)):
    threads[i].start()

for i in range(len(threads)):
    threads[i].join()
# query_filename, queries, doc_filename, documents, words, 