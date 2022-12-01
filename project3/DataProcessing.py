import pandas as pd # Used to read/modify/save csv data
import os
from PIL import Image # Image processing module
from tqdm import tqdm # Progress bars

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