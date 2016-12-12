# Basic Deep Neural Network

import dspo
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError

from pybrain.structure.modules   import SoftmaxLayer

class BasicDNN(object):

    num_hidden_layers = 25

    def __init__(self, rawdatapath):
        self.rawdatapath = rawdatapath
        self.rawdata = dspo.util.parse_foldingcsv(rawdatapath)
        self.input_size = len(self.rawdata) - 1
        self.hidden_size = self.num_hidden_layers
        self.target_size = 1
        self.outename = 'PAPI_TOT_CYC'
        self.outhwc = []
        self.inenames = []
        self.inhwcs = []
        for ename, hwc in self.rawdata.items():
            if ename == self.outename:
                self.outhwc.append(hwc)
            else:
                self.inenames.append(ename)
                self.inhwcs.append(hwc)
        self.dataset = SupervisedDataSet( len(self.rawdata) - 1, 1 )
        self.dataset.setField('input', zip(*self.inhwcs))
        self.dataset.setField('target', zip(*self.outhwc))
        self.tstdata, self.trndata = self.dataset.splitWithProportion( 0.25 )

        self.network = buildNetwork( self.trndata.indim, self.trndata.indim, self.trndata.indim, self.trndata.indim, self.trndata.indim, self.trndata.outdim, bias = True, recurrent=True )
        self.trainer = BackpropTrainer( self.network, dataset=self.trndata)

    def train_network(self):
        #self.trainer.trainEpochs (50)

        #trainingErrors, validationErrors = self.trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 50, continueEpochs = 10 )
        trainingErrors, validationErrors = self.trainer.trainUntilConvergence( maxEpochs = 500, continueEpochs = 10 )

        print 'trainingErrors', trainingErrors
        print ''
        print 'validationErrors', validationErrors

