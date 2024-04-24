import numpy as np
from tqdm import tqdm
from .utils import neuronConvert_Grid_to_R2

class som:

    """
    Self Organizing Map
    input:
    - nrows: number of rows
    - ncols: number of columns
    - dimension: dimension of the neurons
    - vmin: minimum value of the neurons
    - vmax: maximum value of the neurons
    """

    def __init__(self,
                 nrows = 5,
                 ncols = 5,
                 dimension = 3,
                 vmin =  0,
                 vmax =  1,
                 vicinity = "rectangular", #"hexagonal"
                 learn_rate=.1,
                 radius_sq=1,
                 lr_decay=.1,
                 radius_decay=.1,
                 #epochs=10,
                 randomState = 0
                 ):
        self.RDS = np.random.RandomState(randomState)
        self.nrows = nrows
        self.ncols = ncols
        self.dimension = dimension
        self.vmin = vmin
        self.vmax = vmax
        self.vicinity = vicinity
        #self._createSOM()
        self.learn_rate = learn_rate
        self.radius_sq = radius_sq
        self.lr_decay = lr_decay
        self.radius_decay = radius_decay
        #self.epochs = epochs
        print('Som initialization')

    def _createSOM(self):
        SOM = self.RDS.rand(self.nrows,
                             self.ncols,
                             self.dimension)
        SOM = SOM * (self.vmax-self.vmin)+self.vmin
        self.SOM = SOM

    
    # def squareOfDistance(x):
    #     x1,x2 = x
    #     SOQ = np.square(x1-x2).sum(axis=2)
    #     return SOQ
   
    def find_BMU(self, x):
        '''
        x: input data
        return: row and col of the BMU
        '''
        distToNeurons = (np.square(self.SOM - x)).sum(axis=2)
        nrow,ncol = np.unravel_index(np.argmin(distToNeurons,
                                               axis=None),
                                     distToNeurons.shape)
        return nrow,ncol

    def find_BMUH(self,x):
        '''
        Function for Heskes99
        :param x:
        :return:
        '''
    
        return True

    def update_weights(self,
                       train_ex,
                       learn_rate_t,
                       radius_sq_t,
                       BMU_grid_position,
                       step=3):

        minimumDistanceNeurons = 1e-3
        
        # TODO: review STEP (neccessary?)
        
        bmu_row, bmu_col = BMU_grid_position

        if radius_sq_t < minimumDistanceNeurons: # only update BMU vector
            # TODO: place minimum distance between points
            self.SOM[bmu_row, bmu_col, :] += learn_rate_t * (train_ex - self.SOM[bmu_row, bmu_col, :])
            return self

        bmu_x_pos, bmu_y_pos = neuronConvert_Grid_to_R2(self.vicinity,bmu_row,bmu_col)

        for i in range(max(0, bmu_row - step), min(self.SOM.shape[0], bmu_row + step)):
            # i is neuron row
            for j in range(max(0, bmu_col - step), min(self.SOM.shape[1], bmu_col + step)):
                # j is neuron col
                neuron_x_pos, neuron_y_pos = neuronConvert_Grid_to_R2(self.vicinity, i, j)
                dist_sq = np.square(bmu_x_pos - neuron_x_pos) + np.square(bmu_y_pos - neuron_y_pos)
                if self.vicinity == 'rectangular':
                    dist_sq_2 = np.square(i - bmu_row) + np.square(j - bmu_col)
                    if dist_sq != dist_sq_2:
                        raise ValueError('distances do not match')

                dist = np.sqrt(dist_sq)
                dist_func = np.exp(-dist / 2 / radius_sq_t)
                self.SOM[i, j, :] += learn_rate_t * dist_func * (train_ex - self.SOM[i, j, :])

        return self

    def storeMapping(self, train_data):
        '''
        TODO: the SOM map can be trained multiple times
        with different data
        '''
        self.data_ = train_data
        self.data_map = [ [[] for c in range(self.ncols)] for r in range(self.nrows)]
        self.mat_count = np.zeros((self.nrows, self.ncols), np.int_)
        
        for index, x in enumerate(self.data_):
            r, c = self.find_BMU(x)
            self.data_map[r][c].append(index)
            self.mat_count[r,c] +=1

    def get_topologiaXY(self):
        topoXY = np.asarray([[[0,0] for j in range(self.ncols)] for i in range(self.nrows)])
        xc, yc = 0, 0 #Se inicializa el valor del centro del hexagono
        r = 1 #Valor del radio del hexagono
        a = r*np.sqrt(3)/2 # Valor de la apotema
        for i in range(self.nrows): 
            xc = 0 if i%2==0 else a  ## Se vuelve a poner la xc a su valor inicial
            for j in range(self.ncols):
                topoXY[i,j]=[xc,yc]
                xc = xc + 2*a
            yc = yc + r + a/2
        topoXY.resize((topoXY.shape[0]*topoXY.shape[1],topoXY.shape[2]))
        return topoXY

    def getCodes(self):
        _codes = self.SOM
        _codes = _codes.reshape(_codes.shape[0]*_codes.shape[1], _codes.shape[2])
        return _codes


    def getDataNeuron(self, row, col):
        return self.data_[self.data_map[row][col]]
    
    def getIndexDataNeur(self, row, col):
        return self.data_map[row][col]

    def getDistanceNode(self, r, c, data):
        ## Devuelve la distancia de las observaciones del array data a la neurona (r,c)
        
        return np.sqrt([np.dot((self.SOM[r][c]-x), (self.SOM[r][c]-x).T) for x in data])


    def train_SOM(self,train_data,epochs):
        
        self.dimension = train_data.shape[1]
        self._createSOM()


        lstEpocas = tqdm([i for i in range(0, epochs)])
        for epoch in lstEpocas:

            learn_rate_t = self.learn_rate * np.exp(-epoch * self.lr_decay)
            radius_sq_t = self.radius_sq * np.exp(-epoch * self.radius_decay)

            self.RDS.shuffle(train_data)
            for train_ex in train_data:
                bmu_row, bmu_col = self.find_BMU(train_ex)
                self.update_weights(train_ex,
                                    learn_rate_t,
                                    radius_sq_t,
                                    (bmu_row, bmu_col))

        self.storeMapping(train_data)
        return self

    def train_super_SOM(self, train_data, y_train, epochs):
        data_train = np.zeros((train_data.shape[0],train_data.shape[1]+1))
        for i, value in enumerate(zip(train_data, y_train)):
            for j, val in enumerate(value[0]):
                data_train[i,j]=val
            data_train[i, data_train.shape[1]-1] = value[1]

        self.train_SOM(train_data=data_train, epochs=epochs)

        self.SOM_atrib = np.zeros((self.SOM.shape[0],self.SOM.shape[1], self.SOM.shape[2]-1))
        self.SOM_label = np.zeros((self.SOM.shape[0],self.SOM.shape[1]))
        for i in range(self.SOM.shape[0]):
            for j in range(self.SOM.shape[1]):
                for k in range(self.SOM.shape[2]-1):
                    self.SOM_atrib[i,j,k]=self.SOM[i,j,k]
                self.SOM_label[i,j]=np.round(self.SOM[i,j,-1])

        return self

    def getNeuronaCercana(self, x):
        iNeu, jNeu = -1, -1
        dist = np.inf
        for i in range(self.SOM_atrib.shape[0]):
            for j in range(self.SOM_atrib.shape[1]):
                dif_ = self.SOM_atrib[i,j] - x
                w_dist = np.sqrt(np.dot(dif_.T,dif_))
                if w_dist < dist:
                   dist = w_dist
                   iNeu, jNeu = i, j
    
        return iNeu, jNeu

    def getPredict(self, X):
        
        y_pred = []
        for x_ in X:
            neur = self.getNeuronaCercana(x_)
            y_pred.append(self.SOM_label[neur])

        return np.asarray(y_pred)

    def score(self, y_pred, y_test):
        assert len(y_pred.shape) == len(y_test.shape), "y_pred e y_test deben ser 2 arrays de una dimensión"
        assert y_pred.shape[0] == y_test.shape[0], "y_pred e y_test deben ser 2 arrays de igual longitud"
        return len(y_pred[y_pred == y_test])/len(y_pred)

