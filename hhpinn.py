# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import utilities
# import pickle

# # def customDerivative (x, t):
# #     x_dt = []
# #     for i in range(len(x) - 1):
# #         x_dt.append([(x[i]-x[i+1])/(t[i]-t[i+1])])
# #     x_dt.append([(x[-1]-x[-2])/(t[-1]-t[-2])])
# #     return x_dt

# # infile = open("hhdata.pkl", "rb")

# # data = pickle.load(infile)
# # t_all = data["time"]
# # V_all = data["Varr"]
# # h_all = data["harr"]
# # n_all = data["narr"]
# # m_all = data["marr"]
# # V_dt_all = customDerivative(V_all, t_all)
# # h_dt_all = customDerivative(h_all, t_all)
# # n_dt_all = customDerivative(n_all, t_all)
# # m_dt_all = customDerivative(m_all, t_all)
# # V_all = ((np.array(V_all) + 80)/ 110)
# # t = []
# # V = []
# # h = []
# # n = []
# # m = []
# # V_dt = []
# # h_dt = []
# # n_dt = []
# # m_dt = []
# # i = 0
# # while i < len(t_all):
# #     t.append(t_all[i])
# #     V.append(V_all[i])
# #     h.append(h_all[i])
# #     n.append(n_all[i])
# #     m.append(m_all[i])
# #     V_dt.append(V_dt_all[i])
# #     h_dt.append(h_dt_all[i])
# #     n_dt.append(n_dt_all[i])
# #     m_dt.append(m_dt_all[i])
# #     i = i + int((len(t_all))/1000)
# # print(len(V))
# # plt.plot(t, V)
# # plt.show()
# # V_tensor = torch.tensor(V)
# # print(t)

# class PhysicsInformedHHModel:
    
#     def __init__(self, layers, samples, end_time):

#         self.t = utilities.generate_grid_1d(end_time, samples)
#         self.model = utilities.build_model(1,layers,4)
#         self.differential_equation_loss_history = None
#         self.data_loss_history = None
#         self.boundary_condition_loss_history = None
#         self.total_loss_history = None
#         self.optimizer = None

#     def get_predictions(self, t):
#         u = self.model(t)
#         (V, h, n, m) = torch.tensor_split(u, 4, dim=1)
#         return (V, h, n, m)

    
#     def closureData(self):
#         self.optimizer.zero_grad()
#         V = self.get_predictions(self.t)
#         loss = self.costFunctionData(V, self.t)
#         loss.backward(retain_graph=True)
#         return loss
    
#     def closure(self):
#         self.optimizer.zero_grad()
#         V, h, n, m = self.get_predictions(self.t)
#         loss = self.costFunction(V, h, n, m, self.t)
#         loss = loss[0] + loss[1]
#         loss.backward(retain_graph=True)
#         return loss

#     def costFunctionData (self, V_pred_norm, time):
        
#         V_pred = (V_pred_norm * 110) - 80

#         data_loss = torch.sum((V_pred - V_tensor)**2)

#         return data_loss
        
    
#     def costFunction (self, V_pred_norm, h_pred, n_pred, m_pred, time):
            
#         gNa = 120
#         eNa = 115
#         gK = 36  
#         eK = -12
#         gL = 0.3  
#         eL = 10.6

#         V_pred = (V_pred_norm * 110) - 80
        
#         alphaM = ((2.5 - 0.1*(V_pred+65)) / (torch.exp(2.5 - 0.1*(V_pred+65)) - 1))
        
#         alphaN = ((0.1 - 0.01*(V_pred+65)) / (torch.exp(1 - 0.1*(V_pred+65)) - 1))
        
#         alphaH = 0.07 * (torch.exp(-(V_pred+65)/20))
        
#         betaM = 4.0*(torch.exp(-(V_pred+65)/18))
        
#         betaN = 0.125*(torch.exp(-(V_pred+65)/80))
        
#         betaH = 1.0/(torch.exp(3.0-0.1*(V_pred+65))+1)

#         V_pred_dt = utilities.get_derivative(V_pred, time, 1)
#         h_pred_dt = utilities.get_derivative(h_pred, time, 1)
#         m_pred_dt = utilities.get_derivative(m_pred, time, 1)
#         n_pred_dt = utilities.get_derivative(n_pred, time, 1)

#         I = 10
#         V_init = -74.924
#         h_init = 0.025
#         m_init = 0.682
#         n_init = 0.129
        
#         equation_loss = \
#               (torch.sum((V_pred_dt - \
#                         ((gK * (n_pred ** 4) * (eK - (V_pred + 65))) + \
#                         (gNa * (m_pred ** 3) * h_pred * (eNa - (V_pred + 65))) + \
#                         (gL * (eL - (V_pred + 65))) + \
#                          I)) ** 2) + \
#                torch.sum((m_pred_dt - \
#                         (alphaM * (1 - m_pred) - \
#                          betaM * m_pred)) ** 2) + \
#                torch.sum((n_pred_dt - \
#                         (alphaN * (1 - n_pred) - \
#                          betaN * n_pred)) ** 2) + \
#                torch.sum((h_pred_dt - \
#                         (alphaH * (1 - h_pred) - \
#                          betaH * h_pred)) ** 2)).view(1)

#         boundary_loss = \
#             ((V_pred[0] - V_init) ** 2) + \
#             (((m_pred[0] - m_init)*100) ** 2) + \
#             (((n_pred[0] - n_init)*100) ** 2) + \
#             (((h_pred[0] - h_init)*100) ** 2) # + \
#             # ((V_pred[-1] - V_init) ** 2) + \
#             # (((m_pred[-1] - m_init)*100) ** 2) + \
#             # (((n_pred[-1] - n_init)*100) ** 2) + \
#             # (((h_pred[-1] - h_init)*100) ** 2)

#         return equation_loss, boundary_loss

    
#     def train(self, epochs, optimizer='Adam', **kwargs):
#         """Train the model."""

#         # Set optimizer
#         if optimizer=='Adam':
#             self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        
#         elif optimizer=='LBFGS':
#             self.optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)

#         # Initialize history arrays
#         # self.differential_equation_loss_history = np.zeros(diff_epochs)
#         # self.initial_data_loss_history = np.zeros(data_epochs)
#         # self.data_loss_history = np.zeros(diff_epochs)

        
#         self.differential_equation_loss_history = np.zeros(epochs)
#         self.boundary_condition_loss_history = np.zeros(epochs)
#         self.total_loss_history = np.zeros(epochs)

#         # Training loop
#         for i in range(epochs):
#             # Predict displacements
#             V, h, n, m = self.get_predictions(self.t)

#             # Cost function calculation
#             differential_equation_loss, boundary_condition_loss = self.costFunction(V, h, n, m, self.t)

#             # Total loss
#             total_loss = differential_equation_loss + boundary_condition_loss

#             # Add energy values to history
#             # self.differential_equation_loss_history[i] += differential_equation_loss
#             # self.boundary_condition_loss_history[i] += boundary_condition_loss
#             # self.initial_data_loss_history[i] += data_loss

#             self.differential_equation_loss_history[i] += differential_equation_loss
#             self.boundary_condition_loss_history[i] += boundary_condition_loss
#             self.total_loss_history[i] += total_loss

#             # Print training state
#             self.print_training_state(i, epochs)

#             # Update parameters (Neural network train)
#             self.optimizer.step(self.closure)

#         # for i in range(diff_epochs):
#         #     V, h, n, m = self.get_predictions(self.t)

#         #     # Cost function calculation
#         #     data_loss = self.costFunctionData(V, h, n, m, self.t)
#         #     equation_loss = self.costFunction(V, h, n, m, self.t)

#         #     total_loss = data_loss + equation_loss

#         #     # Total loss
#         #     # total_loss = differential_equation_loss + boundary_condition_loss + data_loss

#         #     # Add energy values to history
#         #     self.differential_equation_loss_history[i] += equation_loss
#         #     # self.boundary_condition_loss_history[i] += boundary_condition_loss
#         #     self.data_loss_history[i] += data_loss
#         #     # self.total_loss_history[i] += total_loss

#         #     # Print training state
#         #     self.print_training_state(i, diff_epochs)

#         #     # Update parameters (Neural network train)
#         #     self.optimizer.step(self.closure)

#     def print_training_data_loss (self, epoch, epochs, print_every=1):
#          if epoch == 0 or epoch == (epochs - 1) or epoch % print_every == 0 or print_every == 'all':
#             # Prepare string
#             string = "Epoch: {}/{}\tData loss = {:2f}"

#             # Format string and print
#             print(string.format(epoch, epochs - 1, self.initial_data_loss_history[epoch]))


#     def print_training_state(self, epoch, epochs, print_every=1):
#         """Print the loss values of the current epoch in a training loop."""

#         if epoch == 0 or epoch == (epochs - 1) or epoch % print_every == 0 or print_every == 'all':
#             # Prepare string
#             string = "Epoch: {}/{}\t\tDifferential equation loss = {:2f}\t\boundary loss = {:2f}\t\tTotal loss = {:2f}"

#             # Format string and print
#             print(string.format(epoch, epochs - 1, self.differential_equation_loss_history[epoch], self.boundary_condition_loss_history[epoch], self.total_loss_history[epoch]))


# def run ():
#     samples = 1000
#     end_time = 14.64
#     pinnModel = PhysicsInformedHHModel([100, 100, 100], samples, end_time)
#     epochs = 3000
#     data_epochs = 1000
#     learningRate = 1e-2
    
#     pinnModel.train(epochs, lr=learningRate)
#     t_test = utilities.generate_grid_1d(end_time, samples)
#     V, h, n, m = pinnModel.get_predictions(t_test)
#     V = (V*110) - 80
#     # utilities.plot_displacements_bar(t_test, V)
#     V_plt = V.detach().numpy()
#     # m_plt = m.detach().numpy()
#     # n_plt = n.detach().numpy()
#     t_plt = t_test.detach().numpy()
#     plt.plot(t_plt, V_plt)
#     plt.show()
    
# torch.manual_seed(0)
# run()











import torch
import numpy as np
import matplotlib.pyplot as plt
import utilities



class PhysicsInformedHHModel:
    
    def __init__(self, layers, samples, end_time):

        self.t = utilities.generate_grid_1d(end_time, samples)
        self.model = utilities.build_model(1,layers,4)
        self.differential_equation_loss_history = None
        self.boundary_condition_loss_history = None
        self.total_loss_history = None
        self.optimizer = None

    def get_predictions(self, t):
        u = self.model(t)
        (V, h, n, m) = torch.tensor_split(u, 4, dim=1)
        return V, h, n, m

    
    def closure(self):
        self.optimizer.zero_grad()
        V, h, n, m = self.get_predictions(self.t)
        loss = self.costFunction(V, h, n, m)
        loss = loss[0] + loss[1]
        loss.backward(retain_graph=True)
        return loss
    
    def costFunction (self, V_pred_norm, h_pred, n_pred, m_pred):
    
        gNa = 120
        eNa = 115
        gK = 36  
        eK = -12
        gL = 0.3  
        eL = 10.6
        
        V_pred = (V_pred_norm * 110) - 80
        
        alphaM = ((2.5 - 0.1*(V_pred+65)) / (torch.exp(2.5 - 0.1*(V_pred+65)) - 1))
        
        alphaN = ((0.1 - 0.01*(V_pred+65)) / (torch.exp(1 - 0.1*(V_pred+65)) - 1))
        
        alphaH = 0.07 * (torch.exp(-(V_pred+65)/20))
        
        betaM = 4.0*(torch.exp(-(V_pred+65)/18))
        
        betaN = 0.125*(torch.exp(-(V_pred+65)/80))
        
        betaH = 1.0/(torch.exp(3.0-0.1*(V_pred+65))+1)

        V_pred_dt = utilities.get_derivative(V_pred, self.t, 1)
        h_pred_dt = utilities.get_derivative(h_pred, self.t, 1)
        m_pred_dt = utilities.get_derivative(m_pred, self.t, 1)
        n_pred_dt = utilities.get_derivative(n_pred, self.t, 1)

        I = 10
        # V_init = -74.924
        # h_init = 0.025
        # m_init = 0.682
        # n_init = 0.129
        V_init = -75
        m_init = 0.025
        h_init = 0.07
        n_init = 0.75
        
        
        equation_loss = \
              (torch.sum((V_pred_dt - \
                        ((gK * (n_pred ** 4) * (eK - (V_pred + 65))) + \
                        (gNa * (m_pred ** 3) * h_pred * (eNa - (V_pred + 65))) + \
                        (gL * (eL - (V_pred + 65))) + \
                         I)) ** 2) + \
               torch.sum((m_pred_dt - \
                        (alphaM * (1 - m_pred) - \
                         betaM * m_pred)) ** 2) + \
               torch.sum((n_pred_dt - \
                        (alphaN * (1 - n_pred) - \
                         betaN * n_pred)) ** 2) + \
               torch.sum((h_pred_dt - \
                        (alphaH * (1 - h_pred) - \
                         betaH * h_pred)) ** 2)).view(1)

        # boundary_loss = \
        #     ((V_pred[0] - V_init) ** 2) + \
        #     ((m_pred[0] - m_init) ** 2) + \
        #     ((n_pred[0] - n_init) ** 2) + \
        #     ((h_pred[0] - h_init) ** 2)
        boundary_loss = \
            ((V_pred[0] - V_init) ** 2) + \
            (((m_pred[0] - m_init)*100) ** 2) + \
            (((n_pred[0] - n_init)*100) ** 2) + \
            (((h_pred[0] - h_init)*100) ** 2) # + \
            # ((V_pred[-1] - V_init) ** 2) + \
            # (((m_pred[-1] - m_init)*100) ** 2) + \
            # (((n_pred[-1] - n_init)*100) ** 2) + \
            # (((h_pred[-1] - h_init)*100) ** 2)
      
               
        return equation_loss, boundary_loss

    
    def train(self, epochs, optimizer='Adam', **kwargs):
        """Train the model."""

        # Set optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        
        elif optimizer=='LBFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)

        # Initialize history arrays
        self.differential_equation_loss_history = np.zeros(epochs)
        self.boundary_condition_loss_history = np.zeros(epochs)
        self.total_loss_history = np.zeros(epochs)

        # Training loop
        for i in range(epochs):
            # Predict displacements
            V, h, n, m = self.get_predictions(self.t)

            # Cost function calculation
            differential_equation_loss, boundary_condition_loss = self.costFunction(V, h, n, m)

            # Total loss
            total_loss = differential_equation_loss + boundary_condition_loss

            # Add energy values to history
            self.differential_equation_loss_history[i] += differential_equation_loss
            self.boundary_condition_loss_history[i] += boundary_condition_loss
            self.total_loss_history[i] += total_loss

            # Print training state
            self.print_training_state(i, epochs)

            # Update parameters (Neural network train)
            self.optimizer.step(self.closure)

    def print_training_state(self, epoch, epochs, print_every=10):
        """Print the loss values of the current epoch in a training loop."""

        if epoch == 0 or epoch == (epochs - 1) or epoch % print_every == 0 or print_every == 'all':
            # Prepare string
            string = "Epoch: {}/{}\t\tDifferential equation loss = {:2f}\t\tBoundary condition loss = {:2f}\t\tTotal loss = {:2f}"

            # Format string and print
            print(string.format(epoch, epochs - 1, self.differential_equation_loss_history[epoch],
                                self.boundary_condition_loss_history[epoch], self.total_loss_history[epoch]))


def run ():
    samples = 100
    end_time = 14.64
    pinnModel = PhysicsInformedHHModel([100, 100, 100, 100], samples, end_time)
    epochs = 70
    learningRate = 1e-2    
    pinnModel.train(epochs, optimizer='LBFGS', lr=learningRate)
    t_test = utilities.generate_grid_1d(end_time, samples)
    V, h, n, m = pinnModel.get_predictions(t_test)
    V = (V*110) - 80    
    V_plt = V.detach().numpy()
    t_plt = t_test.detach().numpy()
    plt.plot(t_plt, V_plt)
    plt.show()
    
torch.manual_seed(1)
run()
    
