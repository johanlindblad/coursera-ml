import numpy as np
import matplotlib.pyplot as plot

data = np.loadtxt("ex1data2.txt", delimiter=",")

normalize = True

if normalize:
    # Normalisera enligt (X - snitt) / varians
    averages = np.sum(data, axis=0) / data.shape[0]
    max_values = np.max(data, axis=0)
    min_values = np.min(data, axis=0)

    # Dela med avståndet som är högst av (x_max - snitt) och (snitt - x_min)
    # Det gör att värden aldrig blir över 1
    upper_deltas = max_values - averages
    lower_deltas = averages - min_values
    scaling_factors = np.amax(np.matrix([upper_deltas, lower_deltas]), axis=0)

    data = data - averages
    data = data / scaling_factors

x, y = np.hsplit(data, [-1])
m = x.shape[0]
n = x.shape[1] + 1

# Lägg till konstantfaktorn
x_zero = np.matrix([1] * m).transpose()
x = np.concatenate((x_zero, x), axis=1)

alpha = 0.001
precision = 0.0001

theta = np.array([1] * n)

def hypothesis(theta, x):
    return x.dot(theta)

def cost_function(theta, x, y):
    diffs = hypothesis(theta, x) - np.transpose(y)
    squared = np.multiply(diffs, diffs)
    return 1.0/(m*2) * np.sum(squared)

iterations = 0

J_x = []
J_y = []

while True:
    # h(x_i) - y_i
    differences = hypothesis(theta, x) - np.transpose(y)

    # Stapla diffarna, en för varje variabel
    # Då blir det en rad för varje variables delsummor
    repeated_differences = np.repeat(differences, n, axis=0).T

    times_x = np.multiply(repeated_differences, x)

    # Summera för alla variabler på en gång
    sums = np.sum(times_x, axis=0)
    deltas = sums * alpha/m

    theta = (theta - deltas.A1)
    max_diff = np.amax(abs(deltas))
    iterations += 1

    theta = (theta - deltas.A1)

    if iterations%100 == 0 or max_diff < precision:
        J_y.append(cost_function(theta, x, y))
        J_x.append(iterations)

    if max_diff < precision:
        break

print(theta)

plot.figure(1)
plot.plot(J_x, J_y)

plot.show()
