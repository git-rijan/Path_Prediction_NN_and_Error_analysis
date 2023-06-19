
import os.path
import random
import numpy as np

class Neural_Net(object):

    def __init__(self,szes):
        self.no_layers=len(szes)
        self.szes = szes

        self.bses = [np.random.randn(z, 1) for z in szes[1:]]
        self.wghts = [np.random.randn(z, y)
                        for y, z in zip(szes[:-1], szes[1:])]

    def S_G_D(self, trn_data, epoch, mini_btch_sze, eta,tst_data=None):
        if tst_data: n_tst = len(tst_data)
        n = len(trn_data)
        for i in range(epoch):
            random.shuffle(trn_data)
            mini_btches = [
                trn_data[k:k+mini_btch_sze]
                for k in range(0, n, mini_btch_sze)]
            for mini_btch in mini_btches:
                self.Update_Mini_Btch(mini_btch, eta)
            if tst_data:
                print("Epoch {0}: {1} / {2}".format(
                    i, self.Evluate(tst_data), n_tst))
            else:
                print("Epoch {0} complete".format(i))

            # saving last biases and weights in file
            if i==epoch-1:
                with open('neural_net/biases.npy', 'wb') as f:
                    for obj in self.bses:
                        np.save(f,obj)

                with open('neural_net/weights.npy', 'wb') as f:
                    for obj in self.wghts:
                        np.save(f,obj)

    def Update_Mini_Btch(self, mini_btch, eta):
        # Perform a single mini-batch gradient descent update on the network's weights and biases using backpropagation.
        # The mini-batch is represented as a list of tuples containing input-output pairs (x, y),
        # and the learning rate is denoted as eta."
        nbla_b = [np.zeros(x.shape) for x in self.bses]
        nbla_w = [np.zeros(y.shape) for y in self.wghts]
        for y, z in mini_btch:
            dlta_nbla_b, dlta_nbla_w = self.Back_prop(y, z)
            nbla_b = [Nb+Dnb for Nb, Dnb in zip(nbla_b, dlta_nbla_b)]
            nbla_w = [Nw+Dnw for Nw, Dnw in zip(nbla_w, dlta_nbla_w)]
        self.wghts = [W-(eta/len(mini_btch))*Nw
                        for W, Nw in zip(self.wghts, nbla_w)]
        self.biases = [B-(eta/len(mini_btch))*Nb
                       for B, Nb in zip(self.bses, nbla_b)]

    def Back_prop(self, y, z):
        """Retrieve the grdient for  cost function C_x and return  tuple (nbla_b, nbla_w).
        The variables nbla_b and nbla_w are lists of numpy arrays, structured in a layer-by-layer manner,
        analogous to the self.biases and self.wghts attributes."""
        nbla_b = [np.zeros(B.shape) for B in self.bses]
        nbla_w = [np.zeros(W.shape) for W in self.wghts]
        # Feed_forward
        actvtion = y
        actvtions = [y] # layer by layer storage of activations in a list
        Zs = [] # layer by layer storage of z vectors in a list
        for B, W in zip(self.bses, self.wghts):
            Z = np.dot(W, actvtion)+B
            Zs.append(Z)
            actvtion = Sgmod(Z)
            actvtions.append(actvtion)
        # Backward_Pass
        dlta = self.Cst_drvtive(actvtions[-1], z) * \
            Sgmod_prime(Zs[-1])
        nbla_b[-1] = dlta
        nbla_w[-1] = np.dot(dlta, actvtions[-2].transpose())


        for l in range(2, self.no_layers):
            Z = Zs[-l]
            s_p = Sgmod_prime(Z)
            dlta = np.dot(self.wghts[-l+1].transpose(), dlta) * s_p
            nbla_b[-l] = dlta
            nbla_w[-l] = np.dot(dlta, actvtions[-l-1].transpose())
        return (nbla_b, nbla_w)



    def Cst_drvtive(self, otpt_actvtions, z):
        """Get vector of prtial drvtives (d(C_x )/d(a)
         with respect to the output activations and return it as a result."""
        return (otpt_actvtions-z)

    def Evluate(self, tst_data):
        tst_results = [(self.Feedfrwd(y), z)
                        for (y, z) in tst_data]

        i=0
        for y,z in tst_results:
            if np.array_equal(y,z):
                i=i+1
        return i
        # changes to be made wrt to allowd thresold for each noumber in output column, right now it should be equal

    def Feedfrwd(self, A):
        """Calculate output for input 'A' """
        for B, W in zip(self.bises, self.wghts):
            A = Sgmod(np.dot(W, A)+B)
        return A


# other required fuctions

def Sgmod(Z):
    """The Sigmoid mathematical Function"""
    return 1.0/(1.0+np.exp(-Z))

def Sgmod_prime(Z):
    """Differentition  of the Sigmoid """
    return Sgmod(Z)*(1-Sgmod(Z))
