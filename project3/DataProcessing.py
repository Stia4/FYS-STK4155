import pandas as pd # Used to read/modify/save csv data
import os
from PIL import Image # Image processing module
from tqdm import tqdm # Progress bars
import numpy as np

# Support function to load data generated in the main section of this program
def load_data(params, size=(1e5, 1e5), fullsize=False):
    """
    Parameters are a list of criteria dictionaries, where each dict corresponds to a class of card.
    The classes get values in increasing order, so a list of three param dicts will give an output target
    array containing values 0, 1, and 2 respectively. Each element is a dictionary, with the following structure:
    params = [
        {'suit':1, 'num':1},
        {'suit':2, 'num':8}
    ]
    Both suit and number (rank) can be specified, or just one of them.

    Size allows the user to set the maximal amount of cards used in training set per category.
    The data contains roughly 170 images per suit+rank combination, and multiplied by 4 or 13 should
    only one category be restricted. If size is larger than number of available cards, the maximum amount
    will be used, therefore default is a large value.

    Fullsize boolean chooses whether the grayscale, reduced size data is used, or if it should return raw
    RGB/full size data instead.
    """
    cards = pd.read_csv('data/cards.csv')
    if not fullsize:
        dir = 'data/'; ext = '.png' # Processed data
    else:
        dir = 'raw_data/'; ext = '.jpg' # Raw data

    if np.ndim(size) == 0: # If only a single value is given, assume it applies to train set
        size = (size, 1e5)

    ## Loading data
    X_train = []; X_test = [] # Start with empty lists of cards
    t_train = []; t_test = [] # and targets
    for i, param in enumerate(params):                                                # For each class of card
        include_idx = np.ones(len(cards))                                             # By default include all items
        for category, criteria in param.items():                                      # Go over all criteria for card class
            include_idx = np.logical_and(np.where(cards[category] == criteria, 1, 0), # Find where cards match criteria
                                         include_idx)                                 # And only store where it also matches previous criteria

        n = 0; m = 0
        idx = np.where(include_idx)[0]                  # Get indexes of where accepted cards are
        for j in idx:                                   # Loop over accepted cards
            card = cards.loc[j]
            img = Image.open(dir+card['path'][:-4]+ext) # Read card at listed position
            if fullsize:
                img = img.resize((180, 180))
            if card['set'] == 0 and n < size[0]:        # If card is part of training data, and max size is not reached for class
                X_train.append(np.array(img) / 255)     # Scale and append card to list
                t_train.append(i)                       # With corresponding target value
                n += 1
            elif card['set'] != 0 and m < size[1]:      # If part of test or validation set
                X_test.append(np.array(img) / 255)      # Append to test set instead
                t_test.append(i)
                m += 1

    X_train = np.array(X_train); X_test = np.array(X_test)
    t_train = np.array(t_train); t_test = np.array(t_test)

    return X_train, X_test, t_train, t_test

if __name__ == '__main__':
    """
    Main section of program:
    Turn 224x224 RGB images into 50x50 grayscale images
    """
    ## Directories where Kaggle data is input and processed data is output
    dir_in = 'raw_data/'
    dir_out = 'data/'

    ##################
    # CSV Formatting #
    ##################

    ## Read in table data
    dat = pd.read_csv(dir_in+'cards.csv')

    ## Card naming: 'number' of 'type', and misc joker
    types = {'spades':1, 'hearts':2, 'clubs':3, 'diamonds':4}
    numbers = {'ace':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
            'eight':8, 'nine':9, 'ten':10, 'jack':11, 'queen':12, 'king':13}
    def name_to_value(name):
        name = name.split() # Split into words
        if len(name) == 1:
            return (0, 0) # If only one word: joker case
        return (types[name[-1]], numbers[name[0]])

    ## Change a bunch of names and values to more practical format, and drop unused features
    dat.rename(columns={'class index':'num', 'filepaths':'path', 'card type':'suit', 'data set':'set'}, inplace=True)
    dat.drop('labels', inplace=True, axis=1) # We get names from paths instead

    dat.drop(index=120, inplace=True) # Remove the single item which is fucked somehow, filepath 'train/ace of clubs/output'
    dat.reset_index(drop=True, inplace=True) # Re-index system, dropping all old indexes

    set_type = {'train':0, 'test':1, 'valid':2}
    for i, val in enumerate(dat.values):
        typ, name, file = val[1].split('/') # Split path into set type (folder), card name (subfolder), and filename (001.jpg)
        suit, num = name_to_value(name)     # Split card name into suit and number, i.e. 'nine of spades' -> (1, 9)
        dat.loc[i,  'num'] = num            # Number of card 1-13 (0 = joker)
        dat.loc[i, 'suit'] = suit           # Type of card 1-4
        dat.loc[i,  'set'] = set_type[typ]  # Set, 0: train, 1: test, 2: validation

    ##################
    # Image Resizing #
    ##################

    print("Resizing and converting images...")
    for i, path in tqdm(enumerate(dat['path'])):
        image = Image.open(dir_in+path)

        image = image.resize((50, 50)) # Reduce size
        image = image.convert('L')     # Turn to grayscale
        new_path = path[:-3]+'png'     # Convert to png

        dir_path = dir_out+path[::-1].partition('/')[-1][::-1] # Path without filename
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image.save(dir_out+new_path)
        dat.loc[i, 'path'] = new_path

    ## Save table of card data
    dat.to_csv(dir_out+'cards.csv', index=False)