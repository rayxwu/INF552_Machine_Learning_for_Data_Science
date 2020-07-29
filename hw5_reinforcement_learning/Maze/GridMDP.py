import sys
import numpy as np
class Grid():
    def __init__(self, grid):
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        self.terminals = []
        for x in range(self.cols):
            for y in range(self.rows):
                states.add((x, y))
                reward[(x, y)] = grid[x][y]
                if grid[x][y] == 99:
                    self.terminals.append((x,y))
        self.states = states
        self.reward = reward
        self.actlist = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in self.actlist:
                transitions[s][a] = self.P(s, a)
        self.transitions = transitions

    def P(self, state, action):
        if state in self.terminals:
            return [(0.0,state)]
        other_actions = self.actlist.copy()
        other_actions.remove(action)
        return [(0.7, self.go(state, action)),
                (0.1, self.go(state, other_actions[0])),
                (0.1, self.go(state, other_actions[1])),
                (0.1, self.go(state, other_actions[2]))]

    def go(self, state, action):
        n_state = (state[0]+action[0],state[1]+action[1])
        return n_state if n_state in self.states else state

    def value_iteration(self, epsilon=0.1, gamma=0.9):
        U1 = {s: 0 for s in self.states}
        while True:
            U = U1.copy()
            delta = 0
            for s in self.states:
                U1[s] = self.reward[s] + gamma * max(sum(p * U[s_] for (p, s_) in self.transitions[s][a])
                                        for a in self.actlist)
                delta = max(delta, abs(U1[s] - U[s]))
            if delta <= epsilon * (1 - gamma) / gamma:
                return U

    def to_arrows(self, U):
        chars = {(1, 0): 'v', (0, 1): '>', (-1, 0): '^', (0, -1): '<'}
        pi = [['.'] * self.cols for _ in range(self.rows)]
        for s in self.states:
            if self.reward[s] == 99:
                pi[s[0]][s[1]] = '.'
            elif self.reward[s] == -101:
                pi[s[0]][s[1]] = 'x'
            else:
                pi[s[0]][s[1]] = chars[max(self.actlist,key=lambda x: sum(p * U[s_] for (p,s_) in self.transitions[s][x]))]
        return pi

def get_data(path):
    with open(path, 'r') as fp:
        data = fp.readlines()
    size = int(data[0].replace(' ', ''))
    maze = np.zeros((size, size))
    maze = maze - 1
    for d in data[2:-1]:
        y = int(d.split(',')[0].replace(' ', ''))
        x = int(d.split(',')[1].replace(' ', ''))
        maze[x][y] = -101
    y = int(data[-1].split(',')[0].replace(' ', ''))
    x = int(data[-1].split(',')[1].replace(' ', ''))
    maze[x][y] = 99
    return maze

if __name__ == '__main__':
    assert len(sys.argv) >= 3
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    grid = Grid(get_data(input_path))
    U = grid.value_iteration(gamma=0.9,epsilon=0.1)
    out = grid.to_arrows(U)
    out = list(map(lambda x:''.join(x),out))
    out = '\n'.join(out)
    with open(output_path,'w') as fp:
        fp.write(out)
