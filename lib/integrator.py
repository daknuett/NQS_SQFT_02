class runge_kutta:
    """
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    def __init__(self, weights, nodes, rk_matrix):
        self.weights = weights
        self.nodes = nodes
        self.rk_matrix = rk_matrix

    def __call__(self, Y, t, dt, F, F_args):
        ki = []
        for i, ci in enumerate(self.nodes):
            ki.append(F(t + ci * dt
                        , Y + self.contract_rk_matrix(ki) * dt
                        , *F_args))

        result = Y + dt * sum(b * k for b,k in zip(self.weights, ki))
        return result

    def contract_rk_matrix(self, ki):
        j = len(ki)
        return sum(self.rk_matrix[j][i] * k for i, k in enumerate(ki))

class RK4(runge_kutta):
    """
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples
    """
    def __init__(self):
        self.weights = [1. / 6, 1. / 3, 1. / 3, 1. / 6]
        self.nodes = [0, 0.5, 0.5, 1]
        self.rk_matrix = [[0.0], [0.5, 0], [0, 0.5, 0], [0, 0, 1, 0]]


class SSPRK3(runge_kutta):
    """
    https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """
    def __init__(self):
        self.weights = [1/6, 1/6, 2/3]
        self.nodes = [0, 1, 1/2]
        self.rk_matrix = [[0.0], [1.0, 0.0], [0.25, 0.25, 0.0]]

class Euler1(runge_kutta):
    def __init__(self):
        self.weights = [1.0]
        self.nodes = [0.0]
        self.rk_matrix = [[0.0]]


class Midpoint(runge_kutta):
    def __init__(self):
        self.weights = [0.0, 1.0]
        self.nodes = [0.0, 0.5]
        self.rk_matrix = [[0.0], [0.5, 0.0]]

class SecondOrderRK(runge_kutta):
    def __init__(self, alpha):
        if alpha <= 0:
            raise ValueError("alpha must be in (0, 1]")
        if alpha > 1:
            raise ValueError("alpha must be in (0, 1]")
        
        self.weights = [1 - 1/(2*alpha), 1 / (2*alpha)]
        self.nodes = [0.0, alpha]
        self.rk_matrix = [[0.0], [alpha, 0.0]]

class Midpoint(SecondOrderRK):
    def __init__(self):
        super().__init__(0.5)

class HeunsMethod(SecondOrderRK):
    def __init__(self):
        super().__init__(1.0)
    
class RalstonsMethod(SecondOrderRK):
    def __init__(self):
        super().__init__(2/3)
