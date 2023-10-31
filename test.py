from math import e

def sigmoid_oneIn_oneOut(iterations, rt):
    i = 0.7
    t = 0.5
    w = 0.95

    for n in range(iterations):
        a = i * w
        a = 1 / (1 + (e**-a))

        C_a = 2 * (a - t)
        a_w = a * (1 - a)

        w = w - rt * (C_a * a_w)
        print(f"Value is {a} and weight is {w}")

def sigmoid_twoIn_oneOut(iterations, rt): # C_w CALCULATIONS ARE WRONG - I DIDN'T TAKE INTO ACCOUNT THE ACTIVATION FUNCTION
    i = [0.7, 0.2]
    w = [0.7, -0.9]
    t = 0.5

    for iter in range(iterations):
        a = 0
        for index, n in enumerate(i):
            a += n * w[index]
        
        a = 1 / (1 + (e**-a))

        if iter % (iterations / 100) == 0:
            print(a)
            print(w)

        C_a = 2 * (a - t)
        for index, n in enumerate(w):
            a_w = i[index]
            C_w = C_a * a_w
            w[index] = n - rt * C_w

def sigmoid_oneIn_oneHid_oneOut(iterations, rt):
    i1 = 0.3
    w1 = -0.7
    t = 0.5
    w2 = 0.65

    for iter in range(iterations):
        a1 = i1 * w1
        a1 = 1 / (1 + (e**-a1))

        a2 = a1 * w2
        a2 = 1 / (1 + (e**-a2))

        if iter % (iterations / 100) == 0:
           print(a2)
           print(f"Weight 1: {w1}")
           print(f"Weight 2: {w2}")

        C_a2 = 2 * (a2 - t)
        a2_w2 = a2 * (1 - a2)
        C_w2 = C_a2 * a2_w2
        w2 = w2 - rt * C_w2

        a2_a1 = w2
        a1_w1 = i1

        C_w1 = C_a2 * a2_a1 * a1_w1
        w1 = w1 - rt * C_w1

def sigmoid_twoIn_threeHid_twoOut(iterations, rt):
    expected = [1, 0]
    inputs = [0.3, 0.6]
    w1 = [[.8, -.4], [.15, .6], [-.64, -.8]]
    w2 = [[.32, -.84, -.42], [.1, .62, .92]]


    for iter in range(iterations):
        a1 = []

        for neuronIndex in range(3):
            cur = 0
            for inputIndex, n in enumerate(inputs):
                cur += n * w1[neuronIndex][inputIndex]
            cur = 1 / (1 + (e**-cur))
            a1.append(cur)
        
        a2 = []

        for neuronIndex in range(2):
            cur = 0
            for inputIndex, n in enumerate(a1):
                cur += n * w2[neuronIndex][inputIndex]
            cur = 1 / (1 + (e**-cur))
            a2.append(cur)
        
        if iter % (iterations / 2) == 0:
            print(a1)
            print(a2)
            print(f"Weights 1: {w1}")
            print(f"Weights 2: {w2}")

        output_relations = []

        for outputIndex, weights in enumerate(w2):
            for inputIndex, n in enumerate(weights):
                output = a2[outputIndex]
                wanted = expected[outputIndex]

                C_output = 2 * (output - wanted)
                output_prevOutput = output * (1 - output)
                output_relations.append(output_prevOutput)
                prevOutput_w = a1[inputIndex]

                C_w = C_output * output_prevOutput * prevOutput_w

                w2[outputIndex][inputIndex] = n - rt * C_w
        
        for outputIndex, weights in enumerate(w1):
            for inputIndex, n in enumerate(weights):
                total_Cw = 0
                for i in range(2):
                    output_prevOutput = output_relations[i]
                    prevOutput_HidOut = w2[i][outputIndex]
                    HidOut_prevHidOut = a1[outputIndex] * (1 - a1[outputIndex])
                    prevHidOut_w = inputs[inputIndex]

                    C_w = output_prevOutput * prevOutput_HidOut * HidOut_prevHidOut * prevHidOut_w

                    total_Cw += C_w
                w1[outputIndex][inputIndex] = n - rt * total_Cw


def test_sigmoid_twoIn_threeHid_twoOut(iterations, rt):
    expected = [1, 0]
    inputs = [0.3, 0.6]
    w1 = [[.8, -.4], [.15, .6], [-.64, -.8]]
    w2 = [[.32, -.84, -.42], [.1, .62, .92]]


    for iter in range(iterations):
        a1 = []

        for neuronIndex in range(3):
            cur = 0
            for inputIndex, n in enumerate(inputs):
                cur += n * w1[neuronIndex][inputIndex]
            cur = 1 / (1 + (e**-cur))
            a1.append(cur)
        
        a2 = []

        for neuronIndex in range(2):
            cur = 0
            for inputIndex, n in enumerate(a1):
                cur += n * w2[neuronIndex][inputIndex]
            cur = 1 / (1 + (e**-cur))
            a2.append(cur)
        
        if iter % (iterations / 100) == 0:
            print(a1)
            print(a2)
            print(f"Weights 1: {w1}")
            print(f"Weights 2: {w2}")

        output_relations = []

        for outputIndex, weights in enumerate(w2):
            for inputIndex, n in enumerate(weights):
                output = a2[outputIndex]
                wanted = expected[outputIndex]

                C_output = 2 * (output - wanted)
                output_prevOutput = output * (1 - output)
                output_relations.append(output_prevOutput)
                prevOutput_w = a1[inputIndex]

                C_w = C_output * output_prevOutput * prevOutput_w

                w2[outputIndex][inputIndex] = n - rt * C_w
        
        for outputIndex, weights in enumerate(w1):
            for inputIndex, n in enumerate(weights):
                external_sum = 0
                for i in range(2):
                    output_prevOutput = output_relations[i]
                    prevOutput_HidOut = w2[i][outputIndex]

                    external_sum += output_prevOutput * prevOutput_HidOut
                
                HidOut_prevHidOut = a1[outputIndex] * (1 - a1[outputIndex])
                prevHidOut_w = inputs[inputIndex]

                C_w = external_sum * HidOut_prevHidOut * prevHidOut_w
                
                w1[outputIndex][inputIndex] = n - rt * C_w

#sigmoid_oneIn_oneOut(1000, 0.1)
#sigmoid_twoIn_oneOut(1000, 0.1)
#sigmoid_oneIn_oneHid_oneOut(1000, 0.1)
test_sigmoid_twoIn_threeHid_twoOut(1000000, 0.1)