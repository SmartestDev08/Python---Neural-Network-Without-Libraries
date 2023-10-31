import numpy as np
import random
from math import e

def copy_array(array):
    result = []
    for i in array:
        if isinstance(i, list):
            result.append(copy_array(i))
        else:
            result.append(i)
    
    return result

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

    def predict(self, input, train = False): # Given an input, predict the output
        if len(input) != self.input_size:
            return

        weights = self.weights
        bias = self.bias
        values = [input]

        for i, layer in enumerate(self.hidden):
            new_values = []

            for n in range(layer["length"]):
                sum1 = 0
                for index, i2 in enumerate(weights[i][n]):
                    sum1 += values[-1][index] * i2
                
                sum1 += bias[i][n]
                new_values.append(max(0, sum1))
            
            values.append(new_values)
        
        new_values = []

        for i in range(self.output):
            sum1 = 0

            for index, n in enumerate(weights[-1][i]):
                sum1 += values[-1][index] * n
            
            sum1 += bias[-1][i]
            sum1 = 1 / (1 + (e**-sum1))

            new_values.append(sum1)
        
        values.append(new_values)

        if not train:
            return values[-1]
        else:
            return values

    def train(self, input, expectedOutput, learning_rate): # Adjust weights and biases based off an input and the expected output
        
        def getRelation(layerIndex, neuronIndex): # Calculates a neuron's value relation with the cost
            if layerIndex == len(values):
                pred = expectedOutput[neuronIndex]
                C_a = 2 * (values[layerIndex - 1][neuronIndex] - pred)
                a_r = values[layerIndex - 1][neuronIndex] * (1 - values[layerIndex - 1][neuronIndex])
                return C_a * a_r
            
            relation_allR_a = 0

            for i in range(len(values[layerIndex])):
                new_layer = layerIndex + 1
                a_r = getRelation(new_layer, i)
                #print(layerIndex)
                #print(i)
                #print(neuronIndex)
                #print(weights[layerIndex - 1][i])
                r_preva = weights[layerIndex - 1][i][neuronIndex]

                relation_allR_a += a_r * r_preva
            
            return relation_allR_a
        
        values = self.predict(input, True)
        actual_output = values[-1]
        weights = self.weights
        
        total_error = 0
        for i, expected in enumerate(expectedOutput):
            target = actual_output[i]
            total_error += (target - expected) ** 2
        
        total_error = total_error / len(expectedOutput)
        
        # Updating weights which connect with the output neurons
        new_weights = copy_array(weights)

        for index1, neuronWeights in enumerate(weights[-1]):
            for index2, weight in enumerate(neuronWeights):
                output = actual_output[index1]
                target = expectedOutput[index1]

                C_a = 2 * (output - target)
                a_r = output * (1 - output)
                r_w = values[-2][index2]
                
                C_w = C_a * a_r * r_w
                new_weight_value = weight - ( learning_rate * C_w )
                
                new_weights[-1][index1][index2] = new_weight_value
        
        # Updating weights which don't connect with output neurons

        for index1, layerWeights in enumerate(weights[:-1]): # THESE ARE WEIGHTS - INDEX 1 OF WEIGHTS = INDEX 2 OF VALUES
            for index2, neuronWeights in enumerate(layerWeights):
                for index3, weight in enumerate(neuronWeights):                    
                    r_w = values[index1][index3]
                    a_r = values[index1 + 1][index2] * (1 - values[index1 + 1][index2])
                    allR_a = getRelation(index1 + 2, index2) # Recursion

                    C_w = r_w * a_r * allR_a

                    new_weights[index1][index2][index3] = weight - (learning_rate * C_w)
        self.weights = new_weights

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

epochs = 10 ** 5
learning_rate = 0.03
print(output)
for i in range(epochs):
    network.train([1, 2], [1, 0], learning_rate)

output = network.predict([1, 2])
print(output)

# https://www.youtube.com/watch?v=YOlOLxrMUOw&t=521s&ab_channel=DefendIntelligence

#back propagation
#  output neuron 1:
#    target: 0
#    result: 0.8
#    error = (0-0.8)**2 --> -0.8**2
#    error = 0.64
#  error neuron 2: 0.0324

# total error = mean(0.64, 0.0324)
# total error = 0.3362

# weight error = partial derivative(total error) / parial derivative(weight value)

# weight error = ∂total error / ∂output2 *
#                ∂output2 / ∂input2 *
#                ∂input2 / ∂weight8

# ∂total error / ∂output2 = output2 - target2
# ∂output2 / ∂input2 = output2 * (1 - output2)
# ∂input2 / ∂weight8 = output previous neuron

# output2 = 0.82
# target2 = 1
# output previous neuron = 0.61

# weight error = -0.18 * 0.1476 * 0.61
# weight error = -0.016

# new weight value = weight value - learning rate * weight error
# new weight value = 0.52 - 0.1 * -0.016
# new weight value = 0.5216

# SAME THING FOR HIDDEN LAYER'S WEIGHTS

# ∂total / ∂w4 = ∂Etotal / ∂outj2 * ∂outj2 / ∂inj2 * ∂inj2 / ∂w4

# ∂Etotal / ∂outj2 = ∂Eo1 / ∂outj2 + ∂Eo2 / ∂outj2
#   ∂Eo1 / ∂outj2 = ∂Eo1 / ∂ino1 * ∂ino1 / ∂outj2
#      ∂Eo1 / ∂ino1 = ∂Eo1 / ∂outo1 * ∂outo1 / ∂ino1
#      ∂ino1 / ∂outj2 = w5

#   ∂Eo2 / ∂outj2 = ∂Eo2 / ∂ino2 * ∂ino2 / ∂outj2
#      ∂Eo2 / ∂ino2 = ∂Eo2 / ∂outo2 * ∂outo2 / ∂ino2
#      ∂ino2 / ∂outj2 = w8

# ∂Etotal / ∂outj2 = ∂Etotal

# ∂outj2 / ∂inj2 = outj2 * (1 - outj2)
# ∂inj2 / ∂w4 = i2

# new hidden weight value = hidden weight value - learning rate * hidden weight error