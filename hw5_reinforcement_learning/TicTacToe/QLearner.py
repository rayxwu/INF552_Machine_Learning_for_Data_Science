from Board import Board
import numpy as np
import collections

class QLearner:
    """  Your task is to implement `move()` and `learn()` methods, and 
         determine the number of games `GAME_NUM` needed to train the qlearner
         You can add whatever helper methods within this class.
    """


    # ======================================================================
    # ** set the number of games you want to train for your qlearner here **
    # ======================================================================
    GAME_NUM = 10000


    def __init__(self):
        """ Do whatever you like here. e.g. initialize learning rate
        """
        # =========================================================
        #  
        # 
        # ** Your code here **
        #
        # 
        # =========================================================
        self.gamma = 0.9
        self.alpha = 0.2
        self.epsion = 0.3
        self.train_num = 2*QLearner.GAME_NUM
        self.record_states = []
        self.states_value = collections.defaultdict(int)  # state -> value
        np.random.seed(92727)

    def move(self, board):
        """ given the board, make the 'best' move 
            currently, qlearner behaves just like a random player  
            see `play()` method in TicTacToe.py 
        Parameters: board 
        """
        if board.game_over():
            return
        # =========================================================
        # ** Replace Your code here  **
        # find all legal moves
        self.record_states.append(board.encode_state())

        candidates_action = []
        candidates_action_values = []

        def next_state(s,i,j):
            s = ' '.join(s).split(' ')
            s[i*3+j] = str(self.side) 
            return ''.join(s)

        rand = np.random.rand()

        for i in range(0, 3):
            for j in range(0, 3):
                if board.is_valid_move(i, j):
                    candidates_action.append((i,j))
                    if self.train_num>0 and rand<self.epsion:
                        candidates_action_values.append(0)
                    else:
                        candidates_action_values.append(self.states_value[next_state(board.encode_state(),i,j)])

        # randomly select one and apply it
        #idx = np.random.randint(len(candidates))
        #move = candidates[idx]
        if self.side == 1:
            get_idx_value = max
        elif self.side == 2:
            get_idx_value = min
        idx_value = get_idx_value(candidates_action_values)
        if candidates_action_values.count(idx_value)>1:
            best_options = [i for i in range(len(candidates_action)) if candidates_action_values[i]==idx_value]
            move = candidates_action[np.random.choice(best_options)]
        else:
            move = candidates_action[candidates_action_values.index(idx_value)]

        # =========================================================

        new_state,game_result = board.move(move[0],move[1],self.side)

        self.record_states.append(board.encode_state())
        
        return new_state,game_result

    def learn(self, board):
        """ when the game ends, this method will be called to learn from the previous game i.e. update QValues
            see `play()` method in TicTacToe.py 
        Parameters: board
        """
        if self.side != 1:
            self.record_states = ['000000000']+self.record_states
        # =========================================================
        if board.game_result == 1:
            self.states_value[self.record_states[-1]] = \
                    (1 - self.alpha) * self.states_value[self.record_states[-1]] + \
                    self.alpha * 1
        elif board.game_result == 2:
            self.states_value[self.record_states[-1]] = \
                    (1 - self.alpha) * self.states_value[self.record_states[-1]] + \
                    self.alpha * (-1)
        else:
            self.train_num -= 1
            return
        for i in range(len(self.record_states)-2,-1,-1):
            self.states_value[self.record_states[i]] = \
                (1 - self.alpha) * self.states_value[self.record_states[i]] + \
                self.alpha * self.gamma * self.states_value[self.record_states[i+1]] 

        # =========================================================
        self.record_states = []
        self.train_num -= 1
        # if self.train_num == 0:
        #     s = ['0']*9
        #     for i in range(9):
        #         s[i]='1'
        #         print(self.states_value[''.join(s)])
        #         s[i]='0'


    # do not change this function
    def set_side(self, side):
        self.side = side
