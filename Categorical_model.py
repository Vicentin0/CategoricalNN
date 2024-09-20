import numpy as np

def relu(x):
        return np.maximum(0, x)

def relu_derivative(x):
        return np.where(x > 0, 1, 0)

def softmax(z):
        """
        Compute the softmax of vector z.
        
        Parameters:
        z -- numpy array of shape (n, m), where n is the number of classes and m is the number of examples.

        Returns:
        s -- numpy array of the same shape as z, representing the probability distribution of each class.
        """
        # Subtract the max value from each element for numerical stability
        z = z - np.max(z, axis=0, keepdims=True)
        
        # Compute softmax
        exp_z = np.exp(z)
        s = exp_z / np.sum(exp_z, axis=0, keepdims=True)

        return s

def categorical_cross_entropy_loss(y_pred, y_true):

    m = y_true.shape[0]
    y_pred = np.clip(y_pred.T, 1e-15, 1 - 1e-15)  # Transpose y_pred to match y_true
    loss = -np.sum(y_true * np.log(y_pred)) / m

    return loss

class model:
    """
    Normal nn with 2 hidden layers.
    Uses ReLU and softmax.

    """
    def __init__(self, input_size, hidden_size, output_size) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameters = {}
        self.cache = {}
        self.gradients = {}
        self.A3 = None
        self.trained = False
        self.loss = 0.0

    def initialize_parameters(self):

        hidden_size = self.hidden_size
        input_size = self.input_size
        output_size = self.output_size

        np.random.seed(41)

        W1 = np.random.randn(hidden_size, input_size) * 0.01 
        b1 = np.zeros((hidden_size, 1))
        W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        b2 = np.zeros((hidden_size, 1))
        W3 = np.random.randn(output_size, hidden_size) * 0.01
        b3 = np.zeros((output_size, 1))

        self.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}



    def forward_propagation(self, X):

        parameters = self.parameters

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]

        Z1 = np.dot(W1, X.T) + b1
        A1 = relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = relu(Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = softmax(Z3)  # Use softmax for the output layer
        
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}

        self.A3 = A3
        self.cache = cache

    def update_parameters(self, learning_rate):

        gradients = self.gradients
        parameters = self.parameters

        parameters["W1"] -= learning_rate * gradients["dW1"]
        parameters["W2"] -= learning_rate * gradients["dW2"]
        parameters["W3"] -= learning_rate * gradients["dW3"]
        parameters["b1"] -= learning_rate * gradients["db1"]
        parameters["b2"] -= learning_rate * gradients["db2"]
        parameters["b3"] -= learning_rate * gradients["db3"]

        self.parameters = parameters

    def backward_propagation(self, X, y):

        parameters = self.parameters
        cache = self.cache

        m = X.shape[0]

        A1 = cache["A1"]
        A2 = cache["A2"]
        A3 = cache["A3"]
        Z1 = cache["Z1"]
        Z2 = cache["Z2"]

        W3 = parameters["W3"]
        W2 = parameters["W2"]

        dZ3 = A3 - y.T
        dW3 = np.dot(dZ3, A2.T) / m
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m

        dA2 = np.dot(W3.T, dZ3)
        dZ2 = dA2 * relu_derivative(Z2) 
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(dZ1, X) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m


        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

        self.gradients = gradients

    def train(self, X, y, num_iterations, learning_rate):

        self.initialize_parameters()

        for i in range(num_iterations):

            self.forward_propagation(X)
            loss = categorical_cross_entropy_loss(self.A3,y)

            self.backward_propagation(X, y)
            self.update_parameters(learning_rate)
            
            self.loss = loss

            if i % 1000 == 0:
                print(f"iteration {i}: loss = {loss}")
            
            if loss < 0.01:
                print(f"iteration {i}: loss = {loss}")
                break
        self.trained = True

    def predict(self, X) -> np.array:
    
        if not self.trained:
            print("Model wasn't trained yet.")
            return 
        
        self.forward_propagation(X)
        predictions = np.argmax(self.A3, axis=0)

        return predictions

    
    
     