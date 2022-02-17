import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time 

class CMAC:
    def __init__(self, x, y, generalization=5, num_weights=35, test_size=0.3):
        self.x=x
        self.y=y
        self.generalization_factor=generalization
        self.min_input = np.min(x)
        self.max_input = np.max(x)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=test_size, random_state=2)
        self.weights = np.ones(num_weights)
        self.num_weights = num_weights
        self.association_map = {}

    def plot(self,x,y):
        fig = plt.figure()
        ax1 = fig.add_subplot()
        ax1.plot(self.x, self.y, label=f'y=h(s)')

        idx_list = np.argsort(x)
        sorted_test_x = x[idx_list]
        d_sorted_predicted_y = [y[idx] for idx in idx_list]
        
        ax1.plot(sorted_test_x, d_sorted_predicted_y, 'p-', label='Model Inference')
        ax1.legend(loc='best')
        ax1.set(title=str(self.__class__.__name__), 
        ylabel='Prediction', 
        xlabel='Input')
        plt.show()

class DiscreteCMAC(CMAC):
    def __init__(self, x, y, generalization, num_weights,test_size):
        CMAC.__init__(self, x, y, generalization, num_weights,test_size)
        self.AssociationMap()
   
    def AssociationMap(self):
        num_association_vectors = self.num_weights - self.generalization_factor
        for i in range(len(self.x)):
            if x[i] < self.min_input:
                association_vec_idx = 1
            elif x[i] > self.max_input:
                association_vec_idx = num_association_vectors - 1
            else:
                proportion_idx = (num_association_vectors - 2) * ((self.x[i] - self.min_input) / (self.max_input - self.min_input)) + 1
                association_vec_idx = proportion_idx
            self.association_map[self.x[i]] = int(math.floor(association_vec_idx)) 


    def Predict(self,x):
        self.predicted = []

        if not len(self.association_map)>0:
            self.AssociationMap()

        for i in range(len(x)):
            weight_idx = self.association_map[x[i]]

            # Sum the weights in activated cells
            prediction = np.sum(self.weights[weight_idx : weight_idx + self.generalization_factor])

            self.predicted.append(prediction)

        return self.predicted

    def Train(self, epochs = 100, learning_rate = 0.01):
        
        #reset the model
        self.weights = np.ones(self.num_weights)
        current_epoch = 0

        prev_err = 0
        error = 99999
        converged = False
        while current_epoch <= epochs and not converged:
            prev_err = error

            for i in range(len(self.train_x)):
                # Get index for the beginning of generalization factor window
                weight_index = self.association_map[self.train_x[i]]

                # Output is the sum of the weights within the generalization factor window
                output = np.sum(self.weights[weight_index : weight_index + self.generalization_factor])
                error = self.train_y[i] - output
                correction = (learning_rate * error) / self.generalization_factor
                
                # Recalculate the weight vector values using the correction coeff
                self.weights[weight_index : weight_index + self.generalization_factor] = \
                    [(self.weights[idx] + correction) \
                    for idx in range(weight_index, (weight_index + self.generalization_factor))]

            predictions = self.Predict(self.test_x)
            error = mean_squared_error(self.test_y, predictions)
            val_accuracy=1-error
            
            if val_accuracy<0:
                val_accuracy=0

            if np.abs(prev_err - error) < 0.000001:
                converged = False
                        
            current_epoch = current_epoch + 1
        print(f'Discrete CMAC: \n  Generalization Factor: {self.generalization_factor} \
        \n  Epoch: {current_epoch} \n  Accuracy: {val_accuracy * 100}%')


if __name__=='__main__':
    x = np.linspace(0, 10, 100)
    y = np.zeros(x.shape)
    for i in range(0,len(x)):
        if x[i]<1:
            y[i]=10
        elif x[i]<2:
            y[i]=9
        elif x[i]<3:
            y[i]=7
        elif x[i]<4:
            y[i]=5
        elif x[i]<5:
            y[i]=4
        elif x[i]<6:
            y[i]=4
        elif x[i]<7:
            y[i]=5
        elif x[i]<8:
            y[i]=6
        elif x[i]<9:
            y[i]=7
  
    generalization_factor = 2 
    num_weights = 35
    test_size =0.3

    dCMAC = DiscreteCMAC(x,y,generalization_factor, num_weights, test_size)    
    dCMAC.Train(1000)

    inp = dCMAC.test_x
    prediction = dCMAC.Predict(inp)
    dCMAC.plot(inp,prediction)
    

