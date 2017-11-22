import sys
import theano
import theano.tensor as T
import collections
import numpy as np
import random 
import time
import operator
import cPickle

floatX=theano.config.floatX

class GRUClassifier_batched_2layer(object):
    def __init__(self, n_words, n_classes,batch_size):

        random_seed = 10
        state_embedding = 200
        self.recurrent_size = max(2 * n_words, 256)
        self.context_size = 48
        decay = 0.9

        
        self.rng = np.random.RandomState(random_seed)
        
        self.params = collections.OrderedDict()
        self.rms = collections.OrderedDict()
        
        input_indicies = T.lmatrix('input_indices')
        target_class = T.lvector('target_class')
        learningrate = T.dscalar('learningrate')
        
        state_embeddings = self.create_parameter_matrix('state_embeddings', (n_words,self.context_size))
        
        input_vectors = state_embeddings[input_indicies]
        
        #GRU
        def gru_step(x,h_prev,h2_prev,W_xr,W_hr,W_xz,W_hz,W_xh,W_hh,b_r,b_z,b_h,b_r2,b_z2,b_h2):
            #m = T.nnet.sigmoid(T.dot(x,W_xm) + T.dot(h_prev, W_hm)) 
            r = T.nnet.sigmoid(T.dot(x,W_xr) + T.dot(h_prev, W_hr[0])+b_r)
            z = T.nnet.sigmoid(T.dot(x,W_xz) + T.dot(h_prev, W_hz[0])+b_z)
            _h = T.tanh(T.dot(x, W_xh) + T.dot(r * h_prev, W_hh[0])+b_h) 
            h = z * h_prev + (1.0 - z) * _h
            r2 = T.nnet.sigmoid(T.dot(h,W_hr[1]) + T.dot(h2_prev, W_hr[2])+b_r2)
            z2 = T.nnet.sigmoid(T.dot(h,W_hz[1]) + T.dot(h2_prev, W_hz[2])+b_z2)
            _h2 = T.tanh(T.dot(h, W_hh[1]) + T.dot(r2 * h_prev, W_hh[2])+b_h2) 
            h2 = z2 * h2_prev + (1.0 - z2) * _h2
            return h, h2
        ##Net Parameters
        W_xr = self.create_parameter_matrix('W_xr', (self.context_size, self.recurrent_size))
        W_hr = self.create_parameter_matrix('W_hr', (3,self.recurrent_size, self.recurrent_size))
        W_xz = self.create_parameter_matrix('W_xz', (self.context_size, self.recurrent_size))
        W_hz = self.create_parameter_matrix('W_hz', (3,self.recurrent_size, self.recurrent_size))
        W_xh = self.create_parameter_matrix('W_xh', (self.context_size, self.recurrent_size))
        W_hh = self.create_parameter_matrix('W_hh', (3,self.recurrent_size, self.recurrent_size))
        b_r = self.create_parameter_matrix('b_r', (self.recurrent_size))
        b_z = self.create_parameter_matrix('b_z', (self.recurrent_size))
        b_h = self.create_parameter_matrix('b_h', (self.recurrent_size))
        b_r2 = self.create_parameter_matrix('b_r2', (self.recurrent_size))
        b_z2 = self.create_parameter_matrix('b_z2', (self.recurrent_size))
        b_h2 = self.create_parameter_matrix('b_h2', (self.recurrent_size))
        
        initial_hidden_vec = np.zeros((batch_size,self.recurrent_size),dtype=floatX)
        initial_hidden_vec2 = np.zeros((batch_size,self.recurrent_size),dtype=floatX)

        [hidden_vector, hidden_vector2], _ = theano.scan(
            gru_step,
            sequences = input_vectors,
            outputs_info = [initial_hidden_vec, initial_hidden_vec2],
            non_sequences = [W_xr, W_hr,W_xz, W_hz, W_xh, W_hh,b_r,b_z,b_h,b_r2,b_z2,b_h2]
        )
        hidden_vector2 = hidden_vector2[-1]
        
        W_output = self.create_parameter_matrix('W_output', (self.recurrent_size, n_classes))
        output = T.nnet.softmax(T.dot(hidden_vector2, W_output))
        predicted_class = T.argmax(output, axis = 1)
        
        #cost = -1.0 * T.mean(T.log(output[T.arange(batch_size),target_class]))
        cost = -1.0 * T.mean(T.log(output[T.arange(batch_size),target_class]))
        #cost = T.nnet.binary_crossentropy(output, target_class).mean()
        for m in self.params.values():
            cost += 2.5e-5 * T.sqr(m).sum()

        mstate_embeddings = self.create_rms_matrix('mstate_embeddings', (n_words,self.context_size))
        mW_xr = self.create_rms_matrix('mW_xr', (self.context_size, self.recurrent_size))
        mW_hr = self.create_rms_matrix('mW_hr', (3,self.recurrent_size, self.recurrent_size))
        mW_xz = self.create_rms_matrix('mW_xz', (self.context_size, self.recurrent_size))
        mW_hz = self.create_rms_matrix('mW_hz', (3,self.recurrent_size, self.recurrent_size))
        mW_xh = self.create_rms_matrix('mW_xh', (self.context_size, self.recurrent_size))
        mW_hh = self.create_rms_matrix('mW_hh', (3,self.recurrent_size, self.recurrent_size))    
        mb_r = self.create_rms_matrix('mb_r', (self.recurrent_size))
        mb_z = self.create_rms_matrix('mb_z', (self.recurrent_size))
        mb_h = self.create_rms_matrix('mb_h', (self.recurrent_size))
        mb_r2 = self.create_rms_matrix('mb_r2', (self.recurrent_size))
        mb_z2 = self.create_rms_matrix('mb_z2', (self.recurrent_size))
        mb_h2 = self.create_rms_matrix('mb_h2', (self.recurrent_size))
        mW_output = self.create_rms_matrix('mW_output', (self.recurrent_size, n_classes))
        
        
        gradients = T.grad(cost, self.params.values())
        
        
#        updates = [(p,(p - learningrate * g )) for p,g,m in zip(self.params.values(),gradients,self.rms.values())]
        updates = [(p,(p - (learningrate * (g / (T.sqrt(((.9 * m)+(.1*g**2)))+1e-6))))) for p,g,m in zip(self.params.values(),gradients,self.rms.values())] + [(m,((.9 * m)+(.1*g**2))) for m,g in zip(self.rms.values(),gradients)]
#         for m,g in zip(self.rms.values(),gradients): 
#             updates.append((m,((.5 * m)+(.5*g**2))))
        
        self.train = theano.function([input_indicies, target_class, learningrate], [cost, predicted_class], updates=updates, allow_input_downcast = True)
        self.test = theano.function([input_indicies, target_class], [cost,predicted_class], allow_input_downcast=True)
        self.predclass = theano.function([input_indicies], [predicted_class], allow_input_downcast=True)
        self.testProb = theano.function([input_indicies], [output], allow_input_downcast=True)
        self.getParams = theano.function([],[state_embeddings,W_xr, W_hr, W_xz,W_hz, W_xh,W_hh,b_r,b_z,b_h,W_output])
        self.getmParams = theano.function([],[mstate_embeddings, mW_xh,mW_hh])
        self.input = theano.function([input_indicies], [input_vectors], allow_input_downcast=True)
    
    
    def create_parameter_matrix(self, name, size):
        vals = np.asarray(self.rng.normal(loc=0.0, scale = np.sqrt(1./128), size = size), dtype=floatX)
        self.params[name] = theano.shared(vals, name)
        return self.params[name]
    def create_rms_matrix(self, name, size): 
        vals = np.ones(size, dtype=floatX)
        self.rms[name] = theano.shared(vals, name)
        return self.rms[name]
    def create_id_matrix(self, name, size): 
        vals = np.identity(size, dtype=floatX)
        self.params[name] = theano.shared(vals, name)
        return self.params[name]
    def create_rmsid_matrix(self, name, size): 
        vals = np.identity(size, dtype=floatX)
        self.rms[name] = theano.shared(vals, name)
        return self.rms[name]

def batch_training(gru, batched_seqs, batched_labels, epochs=50, lr=.005,return_costs=False):
    ##Batch Training
    epochs = epochs
    lr = lr
    deltacost = []
    for epoch in xrange(epochs):
        cost_sum = 0.0
        correct = 0
        examples = 0
        #predict conversions
        cnf_cc_t = 0 
        #wrong predicted conversions
        cnf_cn_t = 0
        #predict nonconvert
        cnf_nn_t = 0 
        #wrong predicted nonconvert
        cnf_nc_t = 0
        for index, data in enumerate(batched_seqs):
            seqMat = np.matrix(data).transpose()
            convMat = np.array(batched_labels[index])
            cost, predicted_class = gru.train(seqMat, convMat,lr)
            cost_sum += cost
            try:
                costs.append(cost)
            except NameError:
                pass
            for c,p in zip(convMat, predicted_class):
                examples += 1
                if p == c:
                    correct += 1
                    if p == 1: 
                        cnf_cc_t += 1
                    else: 
                        cnf_nn_t += 1
                else:
                    if p == 1: 
                        cnf_cn_t += 1
                    else: 
                        cnf_nc_t += 1
        deltacost.append(cost_sum)
        print "Epoch: " + str(epoch) + "\tCost: " + str(deltacost[-1]) + "\tAccuracy: " + str(float(correct)/(examples)) + "\tLR: "+ str(lr)
        print "Correct Predictions: " + str(cnf_cc_t + cnf_nn_t)
        print "Predicted Conversions Correct: " + str(cnf_cc_t)
        print "Predicted Conversions Wrong: " + str(cnf_cn_t)
        print "Predicted N-Conversions Correct: " + str(cnf_nn_t)
        print "Predicted N-Conversions Wrong: " + str(cnf_nc_t)
        print "True Positive Rate: " + str(float(cnf_cc_t)/(cnf_cc_t+cnf_cn_t+1e-9))
        print "True Negative Rate: " + str(float(cnf_nn_t)/(cnf_nc_t+cnf_nn_t+1e-9))
        if epoch > 6 and epoch % 4 == 0 :
            if deltacost[-1]/deltacost[-3] > .99 :
                lr = lr / 1.2
    if return_costs == True:
        return costs;

def batch_testing(gru, batched_seqs, batched_labels):
    # Testing
    cost_sum = 0.0
    correct = 0
    examples = 0
    #predict conversions
    cnf_cc = 0 
    #wrong predicted conversions
    cnf_cn = 0
    #predict nonconvert
    cnf_nn = 0 
    #wrong predicted nonconvert
    cnf_nc = 0
    for index, data in enumerate(batched_seqs):
        seqMat = np.matrix(data).transpose()
        convMat = np.array(batched_labels[index])
        predicted_class = gru.predclass(seqMat)
        for c,p in zip(convMat, predicted_class[0]):    
            examples += 1
            if p == c:
                correct += 1
                if p == 1: 
                    cnf_cc += 1
                else: 
                    cnf_nn += 1
            else:
                if p == 1: 
                    cnf_cn += 1
                else: 
                    cnf_nc += 1
    #         print(predicted_class)
    #         print(cnvt)
    #         print(sequence)
    print "Saw: " + str(examples) + "\tTest_accuracy: " + str(float(correct)/(correct+cnf_cn+cnf_nc))
    print "Correct Predictions: " + str(cnf_cc + cnf_nn)
    print "Predicted Conversions Correct: " + str(cnf_cc)
    print "Predicted Conversions Wrong: " + str(cnf_cn)
    print "Predicted N-Conversions Correct: " + str(cnf_nn)
    print "Predicted N-Conversions Wrong: " + str(cnf_nc)
    print "True Positive Rate: " + str(float(cnf_cc)/(cnf_cc+cnf_cn+1e-9))
    print "True Negative Rate: " + str(float(cnf_nn)/(cnf_nc+cnf_nn+1e-9))
    return cost_sum, correct, examples, cnf_cc, cnf_cn, cnf_nc, cnf_nn
