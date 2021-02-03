import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
def generateNegativeDescriptionFile():
    with open('neg.txt', 'w') as f:
        for filename in os.listdir('negatives'):
            f.write('negative/' + filename + '\n')