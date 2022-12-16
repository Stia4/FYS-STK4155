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

def save_plot_with_metadata(NN: NeuralNetwork, filename: str, params, n_train, n_test):
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
    others = np.array([seed, params, n_train, n_test], dtype=object)

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

    n_iters = 10
    p_list = [p]; acc_list = [acc] # Collect state of system while training
    for i in range(n_iters):
        print(f"--Iteration {i+1}--")
        NN.train(X_train, t_train, epochs=1, seed=seed, silent=False)
        p, cost, acc = print_score(NN, X_test, t_test)
        p_list.append(p); acc_list.append(acc)

    # NN.save_params("params/wb.npy")

    ##### Plotting section #####
    plt.rcParams.update({'font.size': 20})
    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, figsize=(12, 8))
    for i, p in enumerate(np.array(p_list).T):
        color = "red" if t_test[i] == 1 else "blue"
        ax0.plot(np.arange(n_iters+1), p, color=color)
    ax1.plot(np.arange(n_iters+1), acc_list)
    plt.suptitle("Per-card output probability and accuracy score evolution")
    ax0.set_ylabel("Card output probability")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epochs")
    plt.tight_layout()
    save_plot_with_metadata(NN, 'OutputEvol', params, n_train_cards, 10)
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
    n_train_cards = 20 # per class in params
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
    NN.load_params("params/All52_20C_88Train40+Test.npy")
    # NN.load_params("params/wb52.npy")

    print_data(t_train, t_test)
    print_structure(image_size, NN, ada_method, eta0, costfunc)
    print("--Initial state--")
    p, cost, acc = print_score(NN, X_test, t_test, multiclass=True)

    n_iters = 0
    for i in range(n_iters):
        print(f"--Iteration {i+1}--")
        NN.train(X_train, t_train, epochs=1, seed=seed, silent=False)
        print_score(NN, X_train, t_train, multiclass=True)

    # NN.save_params("params/wb52.npy")

    ##### Plotting #####

    ## CONFUSTION MATRIX
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    suit = {0:'♠', 1:'♥', 2:'♣', 3:'♦'}
    rank = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'10', 10:'J', 11:'Q', 12:'K'}

    a = np.argmax(p, axis=1) # Categorized output
    conf = confusion_matrix(t_test, a)
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize = (8, 8))
    sns.heatmap(conf, annot=False, ax=ax, cmap="viridis", cbar=False)
    ax.set_xticks(np.arange(0, 52, 13) + 0.5)
    ax.set_yticks(np.arange(0, 52, 13) + 0.5)
    xticks = np.floor(ax.get_xticks()).astype(int)
    yticks = np.floor(ax.get_yticks()).astype(int)
    ax.set_xticklabels([suit[x // 13] + rank[x % 13] for x in xticks])
    ax.set_yticklabels([suit[y // 13] + rank[y % 13] for y in yticks])
    ax.set_title("52 card classification confusion matrix")
    ax.set_ylabel("True value")
    ax.set_xlabel("Network output")
    plt.tight_layout()
    save_plot_with_metadata(NN, 'ConfusionMatrix', params, n_train_cards, n_test_cards)
    plt.show()

    ## Feature maps
    g = lambda x: x / np.max(x)
    norm = lambda x: g(x - np.min(x))

    xn, yn = NN.layers[0].a.shape[:-1] # feature map shape
    conc = np.empty((4*xn, 4*yn))
    for i in range(4):
        for j in range(4):
            conc[i*xn:(i+1)*xn, j*yn:(j+1)*yn] = norm(NN.layers[0].a[..., 4*i + j]) # Each picture is normalized individually
    plt.figure(figsize=(8,8))
    plt.imshow(conc)
    plt.axis('off')
    plt.title("Output of initial convolutional layer")
    save_plot_with_metadata(NN, '16Maps', params, n_train_cards, n_test_cards)
    plt.show()

    xn, yn = NN.layers[4].a.shape[:-1] # feature map shape
    conc = np.empty((8*xn, 8*yn))
    for i in range(8):
        for j in range(8):
            conc[i*xn:(i+1)*xn, j*yn:(j+1)*yn] = norm(NN.layers[4].a[..., 8*i + j])
    plt.figure(figsize=(8,8))
    plt.imshow(conc)
    plt.axis('off')
    plt.title("Output of final convolutional layer")
    save_plot_with_metadata(NN, '64Maps', params, n_train_cards, n_test_cards)
    plt.show()

    ## Plot a correct and a wrong cards
    yes = np.where(t_test == a)[0] # All correct cards
    no  = np.where(t_test != a)[0] # All wrong cards
    n = len(yes)-1

    types = {'spades':1, 'hearts':2, 'clubs':3, 'diamonds':4}
    numbers = {'ace':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
            'eight':8, 'nine':9, 'ten':10, 'jack':11, 'queen':12, 'king':13}
    types = {v: k for k, v in types.items()} # Reverse dictionary
    numbers = {v: k for k, v in numbers.items()}

    fig, ax = plt.subplots(1, 2, figsize=(12,8))
    ax[0].imshow(X_test[yes[n]], cmap='Greys_r')
    ax[1].imshow(X_test[ no[n]], cmap='Greys_r')
    ax[0].set_title(f"This is the {numbers[a[yes[n]] % 13 + 1]} of {types[a[yes[n]] // 13 + 1]}")
    ax[1].set_title(f"This is the {numbers[a[ no[n]] % 13 + 1]} of {types[a[ no[n]] // 13 + 1]}")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].set_xticks([]); ax[1].set_yticks([])
    save_plot_with_metadata(NN, 'GoodBadCards', params, n_train_cards, n_test_cards)
    plt.show()

if __name__ == "__main__":
    BinaryTest()
    # SuitTest()
    # FullTest()