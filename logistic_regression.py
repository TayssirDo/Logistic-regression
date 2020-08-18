import numpy as np
from scipy.special import expit
from numpy.linalg import norm
from utils.metric import accuracy
from sklearn.model_selection import train_test_split


def split_train_val_sets(X, y, val_size, random_state=10):
    """
    Split data into training and validation sets
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size,
                                                      random_state=random_state, shuffle=False)

    return [X_train, X_val, y_train, y_val]


def initialize_weights(n, random_state=None):
    """
    Initialize n weights_list with random values in range [-1,1]
    """

    if random_state is not None:
        np.random.seed(random_state)

    weights = (np.random.rand(n) - 0.5) * 2
    return weights


def predict(x, weights):
    """
    Return the predictions for all inputs in matrix x
    """

    return expit(x @ weights)


def predictions(x, weights):
    """
    Return the predictions for all inputs in matrix x
    """

    return np.round(expit(x @ weights))


def cross_entropy(weights, x, y, alpha):
    """
    Return the value of cross entropy error function
    """

    m = x.shape[0]  # number of inputs

    h = predict(x, weights)  # predictions

    # compute cost function
    epsilon = 1e-14
    term1 = y * np.log(h + epsilon)
    term2 = (1 - y) * np.log(1 - h + epsilon)
    loss = sum(term1 + term2)

    # regularization
    r = .5 * alpha * (weights @ weights)

    return (-1 / m) * loss + r


def cross_entropy_gradient(weights, x, y, alpha):
    """
    Return the gradient of cross entropy error function
    """

    m = x.shape[0]  # number of inputs

    h = predict(x, weights)  # predictions

    # compute gradient
    grad = x.T @ (h - y)
    r_grad = weights

    return (1 / m) * grad + alpha * r_grad


def step_gold_search(fun, w, direction, s, e, alpha, x, y, tol, fixed_range):
    """
    Find suitable learning_rate using golden section search method
    """

    # initialization
    gg = (np.sqrt(5) - 1) / 2
    a1 = gg * s + (1 - gg) * e
    a2 = gg * e + (1 - gg) * s

    weights = np.copy(w)
    fe = fun(weights + e * direction, x, y, alpha)

    weights = np.copy(w)
    f1 = fun(weights + a1 * direction, x, y, alpha)

    weights = np.copy(w)
    f2 = fun(weights + a2 * direction, x, y, alpha)
    i = 0
    while True and not fixed_range:
        if fe >= f2:
            break

        if fe < f2:
            a1 = a2
            f1 = f2
            a2 = e
            f2 = fe
            e = (e - (1 - gg) * s) / gg

            weights = np.copy(w)
            fe = fun(weights + e * direction, x, y, alpha)
        i += 1

    i = 0
    while (abs(a1 - a2)/a1) > tol:
        if f1 >= f2:
            s = a1
            a1 = a2
            f1 = f2
            a2 = gg * e + (1 - gg) * s

            weights = np.copy(w)
            f2 = fun(weights + a2 * direction, x, y, alpha)
        else:
            e = a2
            a2 = a1
            f2 = f1
            a1 = gg * s + (1 - gg) * e

            weights = np.copy(w)
            f1 = fun(weights + a1 * direction, x, y, alpha)
        i += 1

    if f2 < f1:
        x = a2
    else:
        x = a1

    return x


def gradient_descent(x, t, weights, iter_max, alpha):
    """
    Update weights_list using gradient descent
    """

    tol_grad = 1e-8
    iteration = 0
    e_old = 1e20
    tol_error = 1e-8
    while True:
        grad = cross_entropy_gradient(weights, x, t, alpha)  # gradient

        if norm(grad) <= tol_grad or iteration >= iter_max:  # stopping condition
            break

        direction = -grad  # search direction

        # find step size
        s = 0
        e = 1e-3
        tol = 1e-4
        fixed_range = 0
        step = step_gold_search(cross_entropy, weights, direction, s, e, alpha, x, t, tol, fixed_range)

        # update weights_list
        weights += step * direction

        e = cross_entropy(weights, x, t, alpha)

        if abs(e - e_old) <= tol_error:
            break

        e_old = np.copy(e)
        iteration += 1


def cross_validation(x_train, t_train, x_val, t_val, x_test, t_test, iter_max, seed, alpha_vals, labels):
    """
    Apply cross validation to find the optimal regularization parameter.
    """

    # initialize arrays
    train_error = []
    val_error = []

    i = 0
    # try different values of alpha
    for alpha in alpha_vals:
        print('--> alpha = %f' % alpha)

        # initialize weights_list
        weights = initialize_weights(x_train.shape[1], seed)

        # gradient descent
        gradient_descent(x_train, t_train, weights, iter_max, alpha)

        train_error.append(cross_entropy(weights, x_train, t_train, alpha))
        val_error.append(cross_entropy(weights, x_val, t_val, 0))

        if i == 0:  # initial optimal values
            alpha_optimal = alpha
            weights_optimal = weights

        if len(train_error) != 1:  # skip first alpha
            if val_error[i-1] < val_error[i]:
                break
                #True
            else:
                alpha_optimal = alpha
                weights_optimal = weights
        i += 1

    print('\n --> Training set size: ' + str(x_train.shape))
    print('=== After applying logistic regression: ===')
    print('Error: Optimal alpha = ' + str(alpha_optimal))
    print('Train: Error = ' + str(cross_entropy(weights_optimal, x_train, t_train, alpha_optimal)))

    print('Val: Error = ' + str(cross_entropy(weights_optimal, x_val, t_val, 0)))
    print('Val: Accuracy = ' + str(accuracy(t_val, predictions(x_val, weights_optimal))))

    print('Test: Error = ' + str(cross_entropy(weights_optimal, x_test, t_test, 0)))
    print('Test: Accuracy = ' + str(accuracy(t_test, predictions(x_test, weights_optimal))))

    return [weights_optimal, train_error, val_error]


def fit(x, t, x_test, t_test, iter_max, n_images_list, seed, alpha_vals, labels):
    """ Training logistic regression"""

    print('\n--> alpha in ' + str(alpha_vals))
    print('--> iter_max = %d' % iter_max)
    print('--> seed = %d' % seed)
    print('--> number of images in the training set: ' + str([2 ** i for i in n_images_list]))

    # split data into training and validation sets
    val_size = 0.20  # validation set percentage
    x_train_all, x_val, t_train_all, t_val = split_train_val_sets(x, t, val_size)

    indices0 = np.where(t_train_all == 0)[0]
    indices1 = np.where(t_train_all == 1)[0]
    x_train = np.zeros_like(x_train_all)
    t_train = np.zeros_like(t_train_all)

    classe0 = True
    for i in range([len(indices0) if len(indices0)<len(indices1) else len(indices1)][0]):
        if classe0:
            x_train[i] = x_train_all[indices0[i]]
            t_train[i] = t_train_all[indices0[i]]
            classe0 = False
        else:
            x_train[i] = x_train_all[indices1[i]]
            t_train[i] = t_train_all[indices1[i]]
            classe0 = True

    cross_validation(x_train[:2**n_images_list[0]], t_train[:2**n_images_list[0]], x_val,
                     t_val, x_test, t_test, iter_max, seed, alpha_vals, labels)

