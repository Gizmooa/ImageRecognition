import enum

class State(enum.Enum):
    INITIALIZING = 0
    SEARCHING = 1
    MINING = 2
    DROPPING = 3
    #BACKTRACKING = 4