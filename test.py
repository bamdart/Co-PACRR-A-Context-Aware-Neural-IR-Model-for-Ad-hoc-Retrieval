import numpy as np
import select_doc_pos
import os 
from collections import Counter
import logging

MAX_QUERY_LENGTH = 16#param_val['maxqlen'] 
SIM_DIM = 800#param_val['simdim']
usetopic = False
usedesc = True
doc_mat_dir = 'simmat\\'

logger = logging.getLogger('pacrr')

def load_query_idf(qids):
    qid_desc_idf = dict()
    qid_topic_idf = dict()
    for qid in qids:
        qid_desc_idf[qid] = np.load(doc_mat_dir + 'query_idf\\%d.npy' % qid)
        # qid_topic_idf[qid] = np.load(doc_mat_dir + 'query_idf\\topic_term_idf\\%d.npy' % qid)
    return qid_topic_idf, qid_desc_idf

def _load_doc_mat_desc(qids, qid_topic_idf, qid_desc_idf):
    qid_cwid_simmat = dict()
    qid_term_idf = dict()
    for qid in sorted(qids):
        qid_cwid_simmat[qid]=dict()
        # topic_idf_arr, desc_idf_arr = qid_topic_idf[qid], qid_desc_idf[qid]
        topic_idf_arr, desc_idf_arr = None, qid_desc_idf[qid]
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

        # cwid_list = glob.glob('simmat/desc_doc_mat/' + str(qid) + '/*.npy')
        # print('simmat/desc_doc_mat/' + str(qid) + '/')
        # print(cwid_list)

        dirPath = 'simmat/desc_doc_mat/' + str(qid)
        cwid_list = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
        # print(cwid_list)
        for cwid in cwid_list: ##############  這邊要改成讀取資料夾內的所有npy
            cwid = cwid[:-4]
            topic_cwid_f = doc_mat_dir + 'topic_doc_mat/%d/%s.npy'%(qid, cwid)
            desc_cwid_f = doc_mat_dir + 'desc_doc_mat/%d/%s.npy'%(qid, cwid)
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

'''
還不能用
qid_wlen_cwid_mat : query在不同n-gram跟doc的similar matrix，丟qid_wlen_cwid_mat進來
qid_cwid_label : query跟doc是否有關，目前沒有這個
query_idf : query的idf，丟qid_ext_idfarr進來
sample_qids : query列表，丟qids進來
'''
def sample_train_data_weighted(qid_wlen_cwid_mat, qid_cwid_label, query_idfs, sample_qids):
    label2tlabel = {4:2,3:2,2:2,1:1,0:0,-2:0}#這裡我們應該只有1跟0
    sample_label_prob = {2:0.5,1:0.5}#這裡我們應該只有1
    random_seed = 14
    n_batch = 32
    NUM_NEG = 10
    n_query_terms = MAX_QUERY_LENGTH
    n_dims = SIM_DIM
    random_shuffle = True
    binarysimm =  True
    
    np.random.seed(random_seed)
    qid_label_cwids=dict()
    label_count = dict()
    label_qid_count = dict()
    for qid in sample_qids:

        # if qid not in qid_cwid_label or qid not in qid_wlen_cwid_mat:
        #     logger.error('%d in qid_cwid_label %r, in qid_cwid_mat %r'%\
        #             (qid,qid in qid_cwid_label, qid in qid_wlen_cwid_mat))
        #     continue

        if qid not in qid_wlen_cwid_mat:
            logger.error('%d, in qid_cwid_mat %r'%\
                    (qid, qid in qid_wlen_cwid_mat))
            continue

        qid_label_cwids[qid - 1]=dict()
        print(qid)
        print(qid_wlen_cwid_mat[qid])
        print(qid)
        wlen_k = list(qid_wlen_cwid_mat[qid].keys())[0]

        # wlen_k = len(qid_wlen_cwid_mat[qid])
        # wlen_k = 1

        for cwid in qid_cwid_label[qid - 1]:
            # l = label2tlabel[qid_cwid_label[qid][cwid]]
            l = 1
            # if cwid not in qid_wlen_cwid_mat[qid][wlen_k]:
            #     logger.error('%s not in %d in qid_wlen_cwid_mat'%(cwid, qid))
            #     continue

            if l not in qid_label_cwids[qid - 1]:
                qid_label_cwids[qid - 1][l] = list()

            qid_label_cwids[qid - 1][l].append(cwid)

            if l not in label_qid_count:
                label_qid_count[l] = dict()

            if qid not in label_qid_count[l]:
                label_qid_count[l][qid]=0

            label_qid_count[l][qid] += 1

            if l not in label_count:
                label_count[l] = 0

            label_count[l] += 1
       
    if len(sample_label_prob) == 0:
        total_count = sum([label_count[l] for l in label_count if l > 0])
        sample_label_prob = {l:label_count[l]/float(total_count) for l in label_count if l > 0}
        logger.error('nature sample_label_prob', sample_label_prob)

    label_qid_prob = dict()
    for l in label_qid_count:
        if l > 0:
            total_count = label_count[l]
            label_qid_prob[l] = {qid:label_qid_count[l][qid]/float(total_count) for qid in label_qid_count[l]}
            
    sample_label_qid_prob = {l:[label_qid_prob[l][qid] if qid in label_qid_prob[l] else 0 for qid in sample_qids] for l in label_qid_prob}

    while 1:
        pos_batch = dict()
        neg_batch = dict()
        qid_batch = list()
        pcwid_batch = list()
        ncwid_batch = list()
        qidf_batch = list()
        ys = list()

        selected_labels = np.random.choice([l for l in sorted(sample_label_prob)], size=n_batch, replace=True, p=[sample_label_prob[l] for l in sorted(sample_label_prob)])

        label_counter = Counter(selected_labels)
        total_train_num = 0

        for label in label_counter:
            nl_selected = label_counter[label]

            if nl_selected == 0:
                continue
            # print(sample_label_qid_prob[label])
            selected_qids = np.random.choice(sample_qids, size=nl_selected, replace=True, p=sample_label_qid_prob[label])
            # selected_qids = np.random.choice(sample_qids, size=nl_selected, replace=True, p=sample_label_qid_prob[1])
            qid_counter = Counter(selected_qids)

            for qid in qid_counter: 
                pos_label = 0
                nq_selected = qid_counter[qid]

                if nq_selected == 0:
                    continue
                for nl in reversed(range(label)):
                    if nl in qid_label_cwids[qid - 1]:
                        pos_label = label
                        neg_label = nl
                        break

                if pos_label != label:
                    continue

                pos_cwids = qid_label_cwids[qid - 1][label]
                # pos_cwids = qid_label_cwids[qid - 1][1]
                neg_cwids = qid_label_cwids[qid - 1][nl]
                n_pos, n_neg = len(pos_cwids), len(neg_cwids)
                idx_poses = np.random.choice(list(range(n_pos)),size=nq_selected, replace=True)
                min_wlen = min(qid_wlen_cwid_mat[qid])

                for wlen in qid_wlen_cwid_mat[qid]:
                    if wlen not in pos_batch:
                        pos_batch[wlen] = list()

                    for pi in idx_poses:
                        p_cwid = pos_cwids[pi]
                        pos_batch[wlen].append(qid_wlen_cwid_mat[qid][wlen][p_cwid])

                        if wlen == min_wlen:
                            ys.append(1)

                for neg_ind in range(NUM_NEG):
                    idx_negs = np.random.choice(list(range(n_neg)),size=nq_selected, replace=True)
                    min_wlen = min(qid_wlen_cwid_mat[qid])

                    for wlen in qid_wlen_cwid_mat[qid]:
                        if wlen not in neg_batch:
                            neg_batch[wlen] = dict()

                        if neg_ind not in neg_batch[wlen]:
                            neg_batch[wlen][neg_ind]=list()

                        for ni in idx_negs:
                            n_cwid = neg_cwids[ni]
                            neg_batch[wlen][neg_ind].append(qid_wlen_cwid_mat[qid][wlen][n_cwid])

                qidf_batch.append(query_idfs[qid].reshape((1,n_query_terms,1)).repeat(nq_selected, axis=0))

        total_train_num = len(ys)

        if random_shuffle:
            shuffled_index=np.random.permutation(list(range(total_train_num)))
        else:
            shuffled_index = list(range(total_train_num))

        train_data = dict()
        labels = np.array(ys)[shuffled_index]

        getmat = lambda x: np.array(x)
        
        for wlen in pos_batch:
            train_data['pos_wlen_%d'%wlen] = getmat(pos_batch[wlen])[shuffled_index,:]
            for neg_ind in range(NUM_NEG):
                train_data['neg%d_wlen_%d'%(neg_ind,wlen)] = np.array(getmat(neg_batch[wlen][neg_ind]))[shuffled_index,:]

        if binarysimm:
            for k in train_data:
                assert k.find("_wlen_") != -1, "data contains non-simmat objects"
                train_data[k] = (train_data[k] >= 0.999).astype(np.int8)


        train_data['query_idf'] = np.concatenate(qidf_batch, axis=0)[shuffled_index,:]

        train_data['permute'] = np.array([[(bi, qi) for qi in np.random.permutation(n_query_terms)]
                                          for bi in range(n_batch)], dtype=np.int)
        # yield (train_data, labels)
        return train_data, labels

def load_training_data():
    # 看你有幾個query  從1.npy開始讀取
    qids = []
    for i in range(1, 151):
        qids.append(i)

    qid_topic_idf, qid_desc_idf = load_query_idf(qids)
    qid_cwid_rmat, qid_term_idf = _load_doc_mat_desc(qids, qid_topic_idf, qid_desc_idf) # 讀取doc的npy

    qid_cwid_label = np.ones((150, 32))
    select_pos_func = getattr(select_doc_pos, 'select_pos_firstk')
    mat_ngrams = [3]#[max(N_GRAMS)]

    qid_wlen_cwid_mat, qid_ext_idfarr = convert_cwid_udim_simmat(qids, qid_cwid_rmat, select_pos_func, qid_term_idf, mat_ngrams)

    train_data_generator = sample_train_data_weighted(qid_wlen_cwid_mat, qid_cwid_label, qid_ext_idfarr, qids)

    return train_data_generator
