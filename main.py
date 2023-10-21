import numpy as np
import random
from math import e

class Network():
    def __init__(self, layers): # Setup network information
        self.input_size = layers["input"]["length"]
        self.hidden = layers["hidden"]
        self.output = layers["output"]["length"]
    
    def reset(self, settings): # Reset the network by giving it random weights and biases
        def set_weights():
            min_val = settings["weights_range"][0] * 1000
            max_val = settings["weights_range"][1] * 1000

            weights = []

            for index, i in enumerate(self.hidden): # For each hidden layer
                layer_weights = []
                for n in range(i["length"]): # For each neuron of the hidden layer
                    neuron_weights = []
                    if index == 0:
                        for n in range(self.input_size): # For each neuron of the previous layer
                            neuron_weights.append(random.randint(min_val, max_val) / 1000)
                    else:
                        for n in range(self.hidden[index - 1]["length"]):
                            neuron_weights.append(random.randint(min_val, max_val) / 1000)
                    
                    layer_weights.append(neuron_weights)
                
                weights.append(layer_weights)
            
            # Setting output neurons weights
            output_weights = []
            for i in range(self.output):
                neuron_weights = []
                for n in range(self.hidden[-1]["length"]):
                    neuron_weights.append(random.randint(min_val, max_val) / 1000)
                
                output_weights.append(neuron_weights)
            
            weights.append(output_weights)

            return weights
        
        def set_bias():
            min_val = settings["bias_range"][0] * 1000
            max_val = settings["bias_range"][1] * 1000
            bias = []

            for i in self.hidden:
                layer_bias = []
                for n in range(i["length"]):
                    layer_bias.append(random.randint(min_val, max_val) / 1000)
                
                bias.append(layer_bias)
            
            output_bias = []
            for i in range(self.output):
                output_bias.append(random.randint(min_val, max_val) / 1000)
            
            bias.append(output_bias)

            return bias

        self.weights = set_weights()
        self.bias = set_bias()

    def predict(self, input): # Given an input, predict the output
        if len(input) != self.input_size:
            return

        weights = self.weights
        bias = self.bias
        values = input

        for i, layer in enumerate(self.hidden):
            new_values = []

            for n in range(layer["length"]):
                sum1 = 0
                for index, i2 in enumerate(weights[i][n]):
                    sum1 += values[index] * i2
                
                sum1 += bias[i][n]
                new_values.append(max(0, sum1))
            
            values = new_values
        
        new_values = []

        for i in range(self.output):
            sum1 = 0

            for index, n in enumerate(weights[-1][i]):
                sum1 += values[index] * n
            
            sum1 += bias[-1][i]
            sum1 = 1 / (1 + (e**-sum1))

            new_values.append(sum1)
        
        values = new_values

        return values

layers = {
    "input": {
        "length": 2
    },
    "output": {
        "length": 2,
        "activation": "relu"
    },
    "hidden": [
        {
            "length": 3,
            "activation": "sigmoid"
        }
    ]
}
reset_settings = {
    "weights_range": [-1, 1],
    "bias_range": [-1, 1]
}

network = Network(layers)

network.reset(reset_settings)
output = network.predict([1, 2])

print(output)