import sys; sys.path.append("src") # Importing relative codes with relative imports ends badly, adding folder manually
from src.NeuralNetwork import NeuralNetwork # Still using relative position notation (src.) to get VSCode highlighting
from DataProcessing import load_data
import numpy as np
import matplotlib.pyplot as plt
seed = 0

### Several shared print blocks which are used in tests
def print_data(t_train, t_test):
    print("--Data--")
    print(f"Learning set: {len(t_train)} cards")
    print(f"Test/valid. set: {len(t_test)} cards")
    print(f"Classes: {len(np.unique(t_test))}")

def print_structure(image_size, NN, ada_method, eta0, costfunc):
    print("--Network structure--")
    print(f"{'Input':20}", f"{np.prod(image_size):6d}", image_size)
    for l in NN.layers:
        print(f"{l.__class__.__name__:20}", f"{np.prod(l.output_shape):6d}", l.output_shape)
    ada_str = "" if ada_method==None else f", with {ada_method} adaptive method"
    print(f"Learning rate: {eta0:.1e}" + ada_str)
    print(f"Cost function: {costfunc}")

def print_score(NN, X_test, t_test, multiclass=False):
    p = NN.test(X_test, classify=0)
    if not multiclass:
        a = np.where(p >= 0.5, 1, 0)
        cost = NN.costfunc(p, t_test)
        acc = sum(np.where(a == t_test, 1, 0))/len(t_test)
    else:
        a = np.argmax(p, axis=1)
        t_ = np.zeros_like(p) # One-hot vector
        for i, ti in enumerate(t_test):
            t_[i, ti] = 1
        cost = NN.costfunc(p, t_)
        acc = sum(np.where(a == t_test, 1, 0))/len(t_test)
    print("Output probability:", p)
    print("Output class:", a)
    print("Target class:", t_test)
    print("Cost:", cost)
    print("Accuracy:", acc)

    return p, cost, acc

def save_plot_with_metadata(NN: NeuralNetwork, filename: str, params, n_train, n_iters):
    """
    Allows us to save plots along with information how it was created,
    results in simple way to recreate plots if needed, without bloated blocks of code
    or manual notes
    Note seed is global parameter in file and not taken as input (see imports section)
    Some freelaying plot is also assumed to exist, hence plt.savefig
    Include python/module versions? Or overkill?
    NB, potential gradient accumulation is not stored, and is reset with every
    time network is setup anew.
    """
    others = np.array([seed, params, n_train, n_iters], dtype=object)

    NN.save_network("fig/setups/NN_"+filename+".npy")
    NN.save_params("fig/setups/WB_"+filename+".npy")
    np.save("fig/setups/Other_"+filename+".npy", others)
    print("Saved other parameters to", "fig/setups/Other_"+filename+".npy")
    plt.savefig("fig/"+filename+".pdf", format="pdf")
    print("Saved figure to", "fig/"+filename+".pdf")

### Simple two card differentiation
def BinaryTest():
    ##### Data #####
    params = [
        {'suit':2, 'num':8}, # 0: Eight of hearts
        {'suit':1, 'num':1}  # 1: Ace of spades
    ]
    n_train_cards = 10 # per class/item in params
    X_train, X_test, t_train, t_test = load_data(params, size=n_train_cards)

    ##### Network setup #####
    costfunc = "MSE"
    eta0 = 1e-3; ada_method=None
    eta0 = 1e-3; ada_method="Adam"

    image_size = X_train[0].shape
    NN = NeuralNetwork(input_nodes=image_size, init_method="Xavier")
    for n_filters in [16, 32, 64]:
        NN.addLayer(type="Convolutional", params=[3, n_filters, 1, 0], act="linear", seed=seed) # [kernel, filters, stride, padding]
        NN.addLayer(type=      "Pooling", params=[2, 'max'], seed=seed) # [kernel, type]
    NN.addLayer(type="Dense", params= 100,  act="Sigmoid", seed=seed)
    NN.addLayer(type="Dense", params=1, act="Sigmoid")
    NN.set_CostFunc(costfunc)
    NN.set_LearningRate(eta0, method=ada_method)

    ##### Training section #####
    # NN.load_params("params/BinaryH8S1_90+Train95Test.npy")

    # X_test, t_test = X_train, t_train

    print_data(t_train, t_test)
    print_structure(image_size, NN, ada_method, eta0, costfunc)
    print("--Initial state--")
    p, cost, acc = print_score(NN, X_test, t_test)
    
    n_iters = 50
    p_list = [p]; acc_list = [acc] # Collect state of system while training
    try:
        for i in range(n_iters-1):
            print(f"--Iteration {i+1}--")
            NN.train(X_train, t_train, epochs=1, seed=seed, silent=False)
            p, cost, acc = print_score(NN, X_test, t_test)
            p_list.append(p); acc_list.append(acc)
    except KeyboardInterrupt:
        # Note: Partial training step is not discarded with interruption,
        # network is not accountable/reproducable if used after interrupt,
        # just here for easy testing
        n_iters = len(p_list)

    # NN.save_params("params/wb.npy")

    ##### Plotting section #####
    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    for i, p in enumerate(np.array(p_list).T):
        color = "red" if t_test[i] == 1 else "blue"
        ax0.plot(np.arange(n_iters), p, color=color)
    ax1.plot(np.arange(n_iters), acc_list)
    plt.tight_layout()
    save_plot_with_metadata(NN, 'test', params, n_train_cards, n_iters)
    plt.show()

    fig, ax = plt.subplots(*(2, 3))
    ax = ax.flatten()
    for i, axi in enumerate(ax):
        a = NN.layers[i].x[..., 0]
        axi.imshow(a)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.tight_layout()
    plt.show()

### All 4 suits
def SuitTest():
    ##### Data #####
    # params = [{'suit':suit, 'num':rank} for suit in range(1,5) for rank in range(1,14)] # All 52 suit+rank combinations
    params = [{'suit':suit} for suit in range(1,5)] # All 4 suits, [Ace, Heart, Club, Diamond] -> [0, 1, 2, 3]
    # n_train_cards = 3
    n_train_cards = 50 # per class in params
    # n_test_cards = 5
    n_test_cards = 13
    X_train, X_test, t_train, t_test = load_data(params, size=(n_train_cards, n_test_cards), fullsize=False)

    ##### Network setup #####
    costfunc = "MSE"
    eta0 = 1e-4; ada_method="Adam"

    image_size = X_train[0].shape
    NN = NeuralNetwork(input_nodes=image_size, init_method="Xavier")
    for n_filters in [16, 32, 64]:
        NN.addLayer(type="Convolutional", params=[3, n_filters, 1, 1], act="linear", seed=seed) # [kernel, filters, stride, padding]
        NN.addLayer(type=      "Pooling", params=[2, 'max'], seed=seed) # [kernel, type]
    NN.addLayer(type="Dense", params= 128, act="linear", seed=seed)
    NN.addLayer(type="Dense", params=   4, act="linear")

    NN.set_CostFunc(costfunc)
    NN.set_LearningRate(eta0, method=ada_method)

    ##### Training section #####
    NN.load_params("params/Suits50+13Cards_73Train54Test.npy")

    X_test, t_test = X_train, t_train

    print_data(t_train, t_test)
    print_structure(image_size, NN, ada_method, eta0, costfunc)
    print("--Initial state--")
    print_score(NN, X_test, t_test, multiclass=True)

    n_iters = 15
    for i in range(n_iters-1):
        print(f"--Iteration {i+1}--")
        NN.train(X_train, t_train, epochs=1, seed=seed, silent=False)
        print_score(NN, X_test, t_test, multiclass=True)

    NN.save_params("params/wb4.npy")

### All 52 cards
def FullTest():
    ##### Data #####
    params = [{'suit':suit, 'num':rank} for suit in range(1,5) for rank in range(1,14)] # All 52 suit+rank combinations
    n_train_cards = 10 # per class in params
    n_test_cards = 10
    X_train, X_test, t_train, t_test = load_data(params, size=(n_train_cards, n_test_cards), fullsize=False)

    ##### Network setup #####
    costfunc = "MSE"
    eta0 = 1e-4; ada_method="Adam"

    image_size = X_train[0].shape
    NN = NeuralNetwork(input_nodes=image_size, init_method="Xavier")
    for n_filters in [16, 32, 64]:
        NN.addLayer(type="Convolutional", params=[3, n_filters, 1, 1], act="linear", seed=seed) # [kernel, filters, stride, padding]
        NN.addLayer(type=      "Pooling", params=[2, 'max'], seed=seed) # [kernel, type]
    NN.addLayer(type="Dense", params= 128, act="linear", seed=seed)
    NN.addLayer(type="Dense", params=  52, act="linear")

    NN.set_CostFunc(costfunc)
    NN.set_LearningRate(eta0, method=ada_method)

    ##### Training section #####
    NN.load_params("params/wb52.npy")

    print_data(t_train, t_test)
    print_structure(image_size, NN, ada_method, eta0, costfunc)
    print("--Initial state--")
    print_score(NN, X_test, t_test, multiclass=True)

    n_iters = 3
    for i in range(n_iters-1):
        print(f"--Iteration {i+1}--")
        NN.train(X_train, t_train, epochs=1, seed=seed, silent=False)
        print_score(NN, X_train, t_train, multiclass=True)

    NN.save_params("params/wb52.npy")

    ##### Plotting #####

    # CONFUSTION MATRIX

    # Vis igjen at alt går mot p=1/52
    # Gjør dette med varianse? Mean vil alltid være 1/52, men variansen (per output på 52 noder) vil synke
    # Også vis antall kategorier som blir brukt for hele test-settet, dvs. a=[7,7,7,7] -> a=[0,1,34,1] for å vise at det spesialiseres
    # cost også over tid?

if __name__ == "__main__":
    # BinaryTest()
    # SuitTest()
    FullTest()