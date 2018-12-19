from keras.models import Sequential, Model
from keras.layers import Permute, Activation, Dense, Dropout, Embedding, \
Flatten, Input, merge, Lambda, Reshape, Convolution2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras import backend
from .model_base import MODEL_BASE
import tensorflow as tf
from utils.ngram_nfilter import get_ngram_nfilter
'''
輸入設定(照順序)：
pos_doc ：          Shape = (p['maxqlen'], p['simdim'])
pos_doc_context ：  Shape = (p['maxqlen'], p['simdim'])
neg_doc ：          Shape = (p['maxqlen'], p['simdim'])
neg_doc_context ：  Shape = (p['maxqlen'], p['simdim'])
r_query_idf ：      Shape = (p['maxqlen'], 1)
permute_input ：    Shape = (p['maxqlen'], 2)

參數設定：
'maxqlen' :16                     # 一個doc有幾個row ?
'simdim'  :800                    # 一個doc的長度

'nfilter' :32                     # convolution的filter數
'combine' :32                     # Dense layer的filter數   如果是0的話 則用LSTM輸出

'numneg'  :1                      # neg doc有幾個 (用幾個doc來跟pos doc比較)
'kmaxpool':3                      # 做kmax時 要對每個doc 取前幾個最大的

'context' :True                   # 做完kmax後要不要把context接在後面
'shuffle' :True                   # 要不要做random permutation

'distill' :'firstk'               # 不知道是幹嘛的  如果不是firstk 你會需要多輸入很多doc

'winlen' :3                       # 設定kernel size  3就是(1, 1) , (2, 2), (3, 3)
'qproximity' :0                   # 似乎是增加一個(maxqlen, qproximity)大小的kernel
'xfilters' :''                    # 增加一個指定大小的kernel
'''
class PACRR(MODEL_BASE):
    
    params = MODEL_BASE.common_params + ['distill', 'winlen', 'nfilter', 'kmaxpool', 'combine',
                                         'qproximity', 'context', 'shuffle', 'xfilters', 'cascade']

    def __init__(self, *args, **kwargs):
        super(PACRR, self).__init__(*args, **kwargs)
        # 建立kernel size的list
        self.NGRAM_NFILTER, _ = get_ngram_nfilter(self.p['winlen'], self.p['qproximity'],
                                                  self.p['maxqlen'], self.p['xfilters'])
        self.NGRAMS = sorted(self.NGRAM_NFILTER.keys())
        if self.p['qproximity'] > 0:
            self.NGRAMS.append(self.p['qproximity'])


    def _cascade_poses(self):
        '''
        initialize the cascade positions, over which
        we max-pool after the cnn filters.
        the outcome is a list of document positions.
        when the list only includes the SIM_DIM, it 
        is equivalent to max-pool over the whole document
        '''
        doc_poses = list()
        pos_arg = str(self.p['cascade'])
        if len(pos_arg) > 0:
            poses = pos_arg.split('.')
            for p in poses:
                if len(p) > 0:
                    p = int(p)
                    if p <= 0 or p > 100:
                        raise ValueError("Cascade positions are outside (0,100]: %s"%pos_arg)
            doc_poses.extend([int((int(p)/100)*self.p['simdim']) for p in poses if len(p)>0])

        if self.p['simdim'] not in doc_poses:
            doc_poses.append(self.p['simdim'])
            
        return doc_poses

    def _create_inputs(self, prefix):
        '''
        新增input layer
        '''
        p = self.p
        if p['distill'] == 'firstk':
            ng = max(self.NGRAMS)
            shared = Input(shape = (p['maxqlen'], p['simdim']), name='%s_wlen_%d' % (prefix, ng)) 
            inputs = {ng: shared}
        else:
            inputs = {} # NGRAMS有多少  就需要輸入多少doc
            for ng in self.NGRAMS:
                inputs[ng] = Input(shape = (p['maxqlen'], p['simdim']), name='%s_wlen_%d' % (prefix, ng))
            
        return inputs

    def build_doc_scorer(self, r_query_idf, permute_idxs):
        '''
        定義Co PACRR的所有layer  回傳一個function連接Co PACRR網路
        '''
        p = self.p
        ng_fsizes = self.NGRAM_NFILTER # 輸入kernel size的設定  (1, 1) (2, 2) (3, 3)
        maxpool_poses = self._cascade_poses() # 算出doc的dim

        ##### 建立kernel size的list
        filter_sizes = list() 
        added_fs = set() # 避免kernel size重複
        for ng in sorted(ng_fsizes):
            for n_x, n_y in ng_fsizes[ng]:
                dim_name = self._get_dim_name(n_x, n_y)
                if dim_name not in added_fs:
                    filter_sizes.append((n_x,n_y)) 
                    added_fs.add(dim_name)
        
        ##### 定義layer的功能  只是定義而已

        # 1. 定義以下這些功能
        # re_input ： 把input reshpe
        # cov_sim_layers ： 做convolution
        # pool_sdim_layer ： 做kmax pooling
        # pool_sdim_layer_context ： 做k max pooling 並將context連接在後面
        # pool_filter_layer ： 做max pooling filters
        # re_lq_ds ：刪除維度2(長度為1) 將寬取代為CNN filter的結果
        # ex_filter_layer ： 做維度的改變  沒用到
        re_input, cov_sim_layers, pool_sdim_layer, pool_sdim_layer_context, pool_filter_layer, ex_filter_layer, re_lq_ds =\
        self._cov_dsim_layers(p['simdim'], p['maxqlen'], filter_sizes, p['nfilter'], top_k=p['kmaxpool'], poses=maxpool_poses, selecter=p['distill'])

        # 2. SOFTMAX(IDF)
        query_idf = Reshape((p['maxqlen'], 1))(Activation('softmax', name='softmax_q_idf')(Flatten()(r_query_idf)))

        # 3. Dense layer or RNN layer (p['combine']控制要用哪個)
        if p['combine'] < 0:
            raise RuntimeError("combine should be 0 (LSTM) or the number of feedforward dimensions")
        elif p['combine'] == 0:
            rnn_layer = LSTM(1, dropout=0.0, recurrent_regularizer=None, recurrent_dropout=0.0, unit_forget_bias=True, \
                    name="lstm_merge_score_idf", recurrent_activation="hard_sigmoid", bias_regularizer=None, \
                    activation="tanh", recurrent_initializer="orthogonal", kernel_regularizer=None, kernel_initializer="glorot_uniform")
        else:
            d1 = Dense(p['combine'], activation='relu', name='dense_1')
            d2 = Dense(p['combine'], activation='relu', name='dense_2')
            dout = Dense(1, name='dense_output')
            rnn_layer = lambda x: dout(d1(d2(Flatten()(x))))    
        ##### 

        def _permute_scores(inputs):
            scores, idxs = inputs
            return tf.gather_nd(scores, backend.cast(idxs, 'int32'))

        ##### 定義一個function 傳進input layer  連接上面定義過的layer
        self.vis_out = None
        self.visout_count = 0
        print(max(ng_fsizes))
        print('')
        def _scorer(doc_inputs, dataid):
            self.visout_count += 1
            self.vis_out = {}
            doc_qts_scores = [query_idf] # SOFTMAX(IDF)
            for ng in sorted(ng_fsizes): # doc_inputs連接PACRR
                if p['distill'] == 'firstk':
                    input_ng = max(ng_fsizes)
                else:
                    input_ng = ng
                    
                for n_x, n_y in ng_fsizes[ng]:
                    dim_name = self._get_dim_name(n_x, n_y)
                    # 1. 連接convolution和max pooling layer
                    if n_x == 1 and n_y == 1: 
                        doc_cov = doc_inputs[input_ng] # 如果kernel size是1的話 那就直接把輸出等於輸入
                        re_doc_cov = doc_cov
                    else:
                        doc_cov = cov_sim_layers[dim_name](re_input(doc_inputs[input_ng])) # 對輸入做convolution
                        re_doc_cov = re_lq_ds[dim_name](pool_filter_layer[dim_name](Permute((1, 3, 2))(doc_cov))) # 對convolution後的feature map做max pooling
                    self.vis_out['conv%s' % ng] = doc_cov
                    
                    # 2. 選擇要不要使用context  連接k-max layer
                    if p['context']: 
                        ng_signal = pool_sdim_layer_context[dim_name]([re_doc_cov, doc_inputs['context']]) # 做k max pooling 並將context連接在最後
                    else:
                        ng_signal = pool_sdim_layer[dim_name](re_doc_cov) # 單純只做k max pooling而已
                    
                    doc_qts_scores.append(ng_signal)
            
            # 3. 本來是依據kernel size的大小 個別進行運算輸出  現在把他concat回來
            if len(doc_qts_scores) == 1:                
                doc_qts_score = doc_qts_scores[0]# 代表輸出只有query_idf的輸入而已
            else:
                doc_qts_score = Concatenate(axis=2)(doc_qts_scores)# 將query_idf和PACRR的輸出連接在一起
    
            # 4. 把結果根據permute_idxs來做random permute
            if permute_idxs is not None:
                doc_qts_score = Lambda(_permute_scores)([doc_qts_score, permute_idxs])

            # 5. 最後連接上Dense layer或LSTM 輸出
            doc_score = rnn_layer(doc_qts_score)
            return doc_score

        return _scorer

    def build_vis(self):
        '''
        可視化整個model的架構  沒用到
        '''
        assert self.visout_count == 1, "cannot vis when _scorer called multiple times (%s)" % self.visout_count
        
        p = self.p
        
        doc_inputs = self._create_inputs('doc')
        if p['context']:
            doc_inputs['context'] = Input(shape=(p['maxqlen'], p['simdim']), name='doc_context')

        r_query_idf = Input(shape = (p['maxqlen'], 1), name='query_idf')
        doc_scorer = self.build_doc_scorer(r_query_idf, permute_idxs=None)

        doc_score = doc_scorer(doc_inputs, 'doc')
        doc_input_list = [doc_inputs[name] for name in doc_inputs]
        visout = [self.vis_out[ng] for ng in sorted(self.vis_out)]
        print("visout:", sorted(self.vis_out))
        self.model = Model(inputs = doc_input_list + [r_query_idf], outputs = [doc_score] + visout)
        return self.model
    
    def build_predict(self):
        '''
        建立Co PACRR model  
        輸入query和1個doc
        輸出的是doc的score (build是輸出比較結果)
        '''
        p = self.p
        
        # 定義doc input layer
        doc_inputs = self._create_inputs('doc')
        if p['context']:
            doc_inputs['context'] = Input(shape=(p['maxqlen'], p['simdim']), name='doc_context')

        # 定義query input layer
        r_query_idf = Input(shape = (p['maxqlen'], 1), name='query_idf')

        # 定義Co PACRR網路
        doc_scorer = self.build_doc_scorer(r_query_idf, permute_idxs=None)

        # 以doc_inputs作為輸入 連接Co PACRR網路
        doc_score = doc_scorer(doc_inputs, 'doc')

        #  建立input layer list
        doc_input_list = [doc_inputs[name] for name in doc_inputs]
        self.model = Model(inputs = doc_input_list + [r_query_idf], outputs = [doc_score])
        return self.model
    
    def build(self):
        '''
        建立Co PACRR model  
        輸入query、2個doc(pos、neg  neg可設定成n個)和random permutation index
        輸出的是pos score > neg score的機率
        '''
        p = self.p

        r_query_idf = Input(shape = (p['maxqlen'], 1), name='query_idf') # querysim_d
        if p['shuffle']:
            permute_input = Input(shape=(p['maxqlen'], 2), name='permute', dtype='int32') # 設定做random permutation的index  (由這個輸入來指定要換到哪裡去)
        else:
            permute_input = None
        
        # 定義好Co PACRR所有layer  回傳一個function 用來連接定義好的layer
        doc_scorer = self.build_doc_scorer(r_query_idf, permute_idxs=permute_input)

        # 定義pos的input layer (一個doc)
        pos_inputs = self._create_inputs('pos')        
        if p['context']:
            pos_inputs['context'] = Input(shape=(p['maxqlen'], p['simdim']), name='pos_context')
        
        # 定義neg的input layer (另n個doc)
        neg_inputs = {}
        for neg_ind in range(p['numneg']):
            neg_inputs[neg_ind] = self._create_inputs('neg%d' % neg_ind)
            if p['context']:
                neg_inputs[neg_ind]['context'] = Input(shape=(p['maxqlen'], p['simdim']),
                                                       name='neg%d_context' % neg_ind)

        # pos_inputs作為input layer  連接Co PACRR網路 輸出pos score
        pos_score = doc_scorer(pos_inputs, 'pos')
        # neg_inputs作為input layer  連接Co PACRR網路 輸出neg score
        neg_scores = [doc_scorer(neg_inputs[neg_ind], 'neg_%s'%neg_ind) for neg_ind in range(p['numneg'])]

        # 將pos score和neg score做softmax  出來的結果是pos > neg的機率
        pos_neg_scores = [pos_score] + neg_scores
        pos_prob = Lambda(self.pos_softmax, name='pos_softmax_loss')(pos_neg_scores)
        
        #  建立input layer list，有query、pos、neg和permute_input
        pos_input_list = [pos_inputs[name] for name in pos_inputs]
        neg_input_list = [neg_inputs[neg_ind][ng] for neg_ind in neg_inputs for ng in neg_inputs[neg_ind]]
        inputs = pos_input_list + neg_input_list + [r_query_idf]
        if p['shuffle']:
            inputs.append(permute_input)
        
        self.model = Model(inputs = inputs, outputs = [pos_prob])

        self.model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])
        return self.model
