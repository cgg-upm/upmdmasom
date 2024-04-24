import numpy as np

# def neuronaConv_rowCol_posR2(self,fila,columna):

def neuronConvert_Grid_to_R2(vicinity,row, col,ladoHex=1):
    '''
    Input: row, col of a neuron in a hexagonal grid
    Output: position in euclidean space R2
    '''
    posx = posy = 0

    if vicinity == 'hexagonal':

        # dimensiones del hexagono regular:
        # radio circulo cincunscrito
        ru = ladoHex  # igual a longitud de un lado
        # radio circulo inscrito
        ri = np.sqrt(3) / 2 * ru  # cos(30)*ru
        #
        Deltax = 2 * ri  # desplazamiento en x
        Deltay = 3 * ru / 2

        posyinicial = 0
        posy = posyinicial - row * Deltay

        if row % 2 == 0:
            # fila par
            posxinit = ri
        else:
            posxinit = 0

        posx = posxinit + col * Deltax

    elif vicinity == 'rectangular':
        posx = col
        posy = row
    else:
        raise ValueError('vicinity must be hexagonal or rectangular')

    return posx, posy

def neuronaConv_rowCol_index1D(self, fila, columna):
    '''
    Input: row,col of a neuron
    Output: 1D index of the neuron (as if in a 1dim vector)
    This function is inverse to neuronaConv_index1D_rowCol
    '''
    neuron_index = np.ravel_multi_index((fila, columna),
                                        dims=(self.nrows, self.ncols))
    return neuron_index


def neuronaConv_index1D_rowCol(self, neuronIndex):
    '''
    Input: 1D index of the neuron (as if in a 1dim vector)
    Output: row,col of a neuron
    This function is inverse to neuronaConv_rowCol_index1D
    '''
    nrow, ncol = np.unravel_index(neuronIndex, shape=(self.nrows, self.ncols))
    return nrow, ncol


def coordenadasNeurona(self, bmu_row, bmu_col):
    pass