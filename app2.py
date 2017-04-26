from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        #seed the random number generator
        random.seed(1)

        #we model a single neuron with 3 input connections and 1 output connection
        #we assign random weights to a 3x1 matrix, with values in the range of -1 to 1
        #with a mean of 0
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    #the sigmoid function, which describes an s-curve
    #we pass the weighted sum of the inputs through this function
    #to normalise them between 0 - 1
    def __sigmoid(self,x):
        return 1/(1 + exp(-x))

    def __sigmoid_derivative(self,x):
        return x * (1 - x)

    def train(self,training_set_inputs, training_set_outputs,number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            #pass the training set though the NN
            output = self.predict(training_set_inputs)

            #calculate the error
            error = training_set_outputs - output

            #multiply the error by the input and again by the gradient of the sigmoid curve
            adjustment = dot(training_set_inputs.T,error * self.__sigmoid_derivative(output))

            #adjust the weights
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        #pass inputs through our single neural network
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == '__main__':
    #initialise a single neuron neural network
    neural_network = NeuralNetwork()

    print 'Random starting synoptic weights'
    print neural_network.synaptic_weights

    #the training set, we have 4 examples, each consisting of 3 input and
    #1 output value
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,0,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    #Train the neural network using a training set
    #Do it 10k times and adjust each time
    neural_network.train(training_set_inputs,training_set_outputs,10000)

    print 'New synaptic weights after training'
    print neural_network.synaptic_weights

    #Test the NN with a new situation
    print 'Considering a new situation [1,0,0] --> ?:'
    print neural_network.predict(array([1,0,0]))

    #Test the NN with a new situation
    print 'Considering another new situation [1,1,0] --> ?:'
    print neural_network.predict(array([1,1,0]))

    #Test the NN with a new situation
    print 'Considering another new situation [1,1,0] --> ?:'
    print neural_network.predict(array([0,1,0]))
