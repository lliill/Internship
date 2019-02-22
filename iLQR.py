import autograd.numpy as np
from autograd import grad, jacobian, hessian
from scipy.signal import unit_impulse
basis = [unit_impulse(6,i) for i in range(6)]

class iLQR:
    """iterative Linear Quadratic Regulator."""
    def __init__(self,  x0: "initial position, R^n", 
                 p: "desired position, R^n",
                 m: "dimension of control, R^m", 
                 F: "Euler schema function, R^{n+m} -> R^n",
##                 DF:"first and second differentials of F",
                 L: "loss function, R^{n+m} -> R",
##                 DL:"first and second differentials of L",
                 Lf:"final loss function: R^n -> R",
##                 DLf:"first and second differentials of Lf",
                 T: "time" = 1, 
                 N: "discretization" = 100,
                ):
         
        self.T = T
        self.N = N
        self.X = np.array(N * [x0])
        self.U = np.zeros((N,m))    
        self.p = p
        self.L = L
##        self.DL = DL
        self.Lf = Lf
##        self.DLf = DLf
        self.F = F
##        self.DF = DF
        self.state_dim = x0.size
        self.control_dim = m

    def push(self):
        """Backward pass and Forward pass."""

        n = self.state_dim
        
        gradient_Lf, hessian_Lf = grad(self.Lf)(*self.X[-1]), hessian(self.Lf)(*self.X[-1])

        DVstar_list_inv = [gradient_Lf] #DV*_{n-1}(x_{n-1})
        D2Vstar_list_inv = [hessian_Lf]

        DV_list_inv = []
        D2V_list_inv = []

        #backward pass, begin with DV_n-2
        for t in range(self.N-2, -1, -1): #from N-2 to 0

            gradient_L, hessian_L = grad(self.L)(*self.X[t], *self.U[t]), hessian(self.L)(*self.X[t], *self.U[t])
            jacobian_F, hessian_F = jacobian(self.F)(*self.X[t], *self.U[t]), hessian(self.F)(*self.X[t], *self.U[t])
            DV = gradient_L + DVstar_list_inv[-1] @ jacobian_F
            D2V = np.reshape([ei @ hessian_L @ ej + 
                              DVstar_list_inv[-1] @ (ej @ hessian_F @ ei) + 
                              (jacobian_F @ ej) @ D2Vstar_list_inv[-1] @ (jacobian_F @ ei) for ei in basis for ej in basis], 
                             (self.state_dim + self.control_dim, self.state_dim + self.control_dim))

            DV_list_inv.append(DV)
            D2V_list_inv.append(D2V)

            DVstar = DV[:n] + DV[n:] @ np.linalg.inv(D2V[n:, n:]) @ D2V[n:, :n]
            D2Vstar = D2V[:n, :n] + D2V[:n, n:] @ np.linalg.inv(D2V[n:, n:]) @ D2V[n:, :n]

            DVstar_list_inv.append(DVstar)
            D2Vstar_list_inv.append(D2Vstar)

        DV = DV_list_inv[::-1]
        D2V = D2V_list_inv[::-1]

        X_hat = np.copy(self.X)
        #Forward pass
        for t in range(N-1):
            if t == 0:
                h_u = -np.linalg.inv(D2V[t][4:, 4:]) @ DV[t][4:]
                self.U[t] = self.U[t] + h_u
                X_hat[t+1] = self.F(*X_hat[t], *self.U[t])
            else:
                h_x = X_hat[t] - self.X[t]
                h_u = -np.linalg.inv(D2V[t][4:, 4:]) @ (DV[t][4:] + D2V[t][4:, :4] @ h_x)
                self.U[t] = self.U[t] + h_u
                X_hat[t+1] = self.F(*X_hat[t], *self.U[t])

        self.X = X_hat
