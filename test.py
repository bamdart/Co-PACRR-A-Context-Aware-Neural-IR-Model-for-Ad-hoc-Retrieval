import numpy as np
import select_doc_pos
import glob 

MAX_QUERY_LENGTH = 16#param_val['maxqlen'] 
SIM_DIM = 800#param_val['simdim']
usetopic = True
usedesc = True
doc_mat_dir = 'simmat\\cosine\\'


def load_query_idf(qids):
    qid_desc_idf = dict()
    qid_topic_idf = dict()
    for qid in qids:
        qid_desc_idf[qid] = np.load(doc_mat_dir + 'query_idf\\desc_term_idf\\%d.npy' % qid)
        qid_topic_idf[qid] = np.load(doc_mat_dir + 'query_idf\\topic_term_idf\\%d.npy' % qid)
    return qid_topic_idf, qid_desc_idf

def _load_doc_mat_desc(qids, qid_topic_idf, qid_desc_idf):
    qid_cwid_simmat = dict()
    qid_term_idf = dict()
    for qid in sorted(qids):
        qid_cwid_simmat[qid]=dict()
        topic_idf_arr, desc_idf_arr = qid_topic_idf[qid], qid_desc_idf[qid]
        descmax = maxqlen = MAX_QUERY_LENGTH
        didxs = list(range(len(desc_idf_arr)))
        mi = []
        if usetopic:
            descmax = maxqlen - len(topic_idf_arr)
            mi.append(topic_idf_arr)
        if usedesc:
            if len(didxs) > descmax:
                didxs = np.sort(np.argsort(desc_idf_arr)[::-1][:descmax])
            mi.append(desc_idf_arr[didxs])
        qid_term_idf[qid] = np.concatenate(mi, axis=0).astype(np.float32)

        cwid_list = glob.glob('/topic_doc_mat/%d/'%(qid))
        for cwid in cwid_list: ##############  這邊要改成讀取資料夾內的所有npy
            topic_cwid_f = doc_mat_dir + '/topic_doc_mat/%d/%s.npy'%(qid, cwid)
            desc_cwid_f = doc_mat_dir + '/desc_doc_mat/%d/%s.npy'%(qid, cwid)
            topic_mat, desc_mat = np.empty((0,0), dtype=np.float32), np.empty((0,0), dtype=np.float32)
            if usetopic:
                topic_mat = np.load(topic_cwid_f)
            if usedesc:
                desc_mat = np.load(desc_cwid_f)[didxs]
            empty = True
            m = []
            if usetopic:
                m.append(topic_mat)
                if topic_mat.shape[1] > 0:
                    empty = False
            if usedesc:
                m.append(desc_mat)
                if desc_mat.shape[1] > 0:
                    empty = False
            if usetopic and usedesc and topic_mat.shape[1] != desc_mat.shape[1]:
                empty = True
            if not empty:
                qid_cwid_simmat[qid][cwid] = np.concatenate(m, axis=0).astype(np.float32)
    return qid_cwid_simmat, qid_term_idf


def convert_cwid_udim_simmat(qids, qid_cwid_rmat, select_pos_func, qid_term_idf, n_grams):
    qid_cwid_mat=dict()
    qid_context = {}
    qid_ext_idfarr = dict()
    pad_value = 0

    qid_cwid_qermat = None
    dim_sim = SIM_DIM
    max_query_term = MAX_QUERY_LENGTH

    for qid in qids:
        qid_cwid_mat[qid]=dict()
        for cwid in qid_cwid_rmat[qid]:
            len_doc = qid_cwid_rmat[qid][cwid].shape[1]
            len_query = qid_cwid_rmat[qid][cwid].shape[0]
            if qid not in qid_ext_idfarr:
                qid_ext_idfarr[qid] =  np.pad(qid_term_idf[qid], pad_width=((0,max_query_term-len_query)), mode='constant', constant_values=-np.inf)     
            for n_gram in n_grams:  
                if n_gram not in qid_cwid_mat[qid]:
                    qid_cwid_mat[qid][n_gram]=dict()  
                    if len_doc > dim_sim:
                        rmat = np.pad(qid_cwid_rmat[qid][cwid],  pad_width=((0,max_query_term-len_query),(0, 1)), mode='constant', constant_values=pad_value).astype(np.float32)
                        selected_inds = select_pos_func(qid_cwid_rmat[qid][cwid], dim_sim, n_gram)
                        if qid_cwid_qermat is None:
                            qid_cwid_mat[qid][n_gram][cwid] = (rmat[:, selected_inds])
                        else:
                            qermat = np.pad(qid_cwid_qermat[qid][cwid],  pad_width=((0,max_query_term-len_query),(0, 1)), mode='constant', constant_values=pad_value).astype(np.float32)
                            qid_cwid_mat[qid][n_gram][cwid] = (rmat[:, selected_inds],\
                                    qermat[:, selected_inds])
                elif len_doc < dim_sim:
                    if qid_cwid_qermat is None:
                        qid_cwid_mat[qid][n_gram][cwid] = \
                                (np.pad(qid_cwid_rmat[qid][cwid],\
                                pad_width=((0,max_query_term-len_query),\
                                (0, dim_sim - len_doc)),mode='constant', \
                                constant_values=pad_value).astype(np.float32))
                    else:
                        qid_cwid_mat[qid][n_gram][cwid] = \
                                (np.pad(qid_cwid_rmat[qid][cwid],\
                                pad_width=((0,max_query_term-len_query),\
                                (0, dim_sim - len_doc)),mode='constant', \
                                constant_values=pad_value).astype(np.float32),\
                                np.pad(qid_cwid_qermat[qid][cwid],\
                                pad_width=((0,max_query_term-len_query),\
                                (0, dim_sim - len_doc)),mode='constant', \
                                constant_values=pad_value).astype(np.float32))    
                else:
                    if qid_cwid_qermat is None:
                        qid_cwid_mat[qid][n_gram][cwid] = (np.pad(qid_cwid_rmat[qid][cwid],\
                                        pad_width=((0,max_query_term-len_query),(0, 0)),\
                                mode='constant', constant_values=pad_value).astype(np.float32))
                    else:
                        qid_cwid_mat[qid][n_gram][cwid] = (np.pad(qid_cwid_rmat[qid][cwid],\
                                        pad_width=((0,max_query_term-len_query),(0, 0)),\
                                mode='constant', constant_values=pad_value).astype(np.float32),\
                                np.pad(qid_cwid_qermat[qid][cwid],\
                                pad_width=((0,max_query_term-len_query),(0, 0)),\
                                mode='constant', constant_values=pad_value).astype(np.float32))    
    return qid_cwid_mat, qid_ext_idfarr

# 看你有幾個query  從1.npy開始讀取
qids = []
for i in range(1, 2):
    qids.append(i)

qid_topic_idf, qid_desc_idf = load_query_idf(qids)
qid_cwid_rmat, qid_term_idf = _load_doc_mat_desc(qids, qid_topic_idf, qid_desc_idf) # 讀取doc的npy

select_pos_func = getattr(select_doc_pos, 'select_pos_firstk')
mat_ngrams = [3]#[max(N_GRAMS)]

qid_wlen_cwid_mat, qid_ext_idfarr = convert_cwid_udim_simmat(qids, qid_cwid_rmat, select_pos_func, qid_term_idf, mat_ngrams)

pass