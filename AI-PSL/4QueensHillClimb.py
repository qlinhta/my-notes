import numpy as np
import random
import math
import time as tm


class NQueens:
    '''Creates NQueens board
    '''

    def __init__(self, num_cols, board=None, seed=0):
        self._rand_obj = np.random.RandomState()
        self._rand_obj.seed(seed)
        self._board_size = num_cols
        self._board = board
        self.seed = seed
        if board is None:
            self._new_board()

    def _new_board(self):
        self._board = [None] * self._board_size
        for i in range(self._board_size):
            self._board[i] = self._rand_obj.randint(0, self._board_size - 1)

    def size(self):
        return self._board_size

    def place_queen(self, row, col):
        self._board[col] = row

    def remove_queen(self, col):
        self._board[col] = None

    def reset(self):  # O(n)
        self._new_board()

    def get_board(self):
        return self._board

    def draw(self):
        temp_str = ""
        temp_board = self._board[:]
        for i in range(self._board_size):
            find_indices = []
            cumul_ind = 0
            try:
                for j in range(0, temp_board.count(i)):
                    find_indices.append(temp_board.index(i))
                    temp_board[temp_board.index(i)] = None
                find_indices.sort()
                temp_str += ("-\t" * (find_indices[0])) + "Q" + "\t"
                cumul_ind += find_indices[0]
                for j in range(1, len(find_indices)):
                    cumul_ind += find_indices[j]
                    temp_str += ("-\t" * (find_indices[j] - find_indices[j - 1] - 1)) + \
                                "Q" + "\t"
                temp_str += ("-\t" * (self._board_size - cumul_ind - 1))
            except:
                temp_str = temp_str + ("-\t" * self._board_size)
            temp_str += "\n"
        print(temp_str)


class _local_node:
    def __init__(self, state, h_val):
        self.state = state
        self.h = h_val


class NQueensAgent:
    def __init__(self, initial_state, seed):
        self._curr_node = _local_node(initial_state, self.h(initial_state))
        self._rand_obj = np.random.RandomState()
        self._rand_obj.seed(seed)
        self.seed = seed

    def get_state(self):
        return self._curr_node.state

    def get_h(self):
        return self._curr_node.h

    def h(self, state):
        total_h = 0
        for col in range(state.size()):
            row = state.get_board()[col]
            # We only need to check the columns in front of the current Queen
            for i in range(col + 1, state.size()):
                if state.get_board()[i] == row:
                    total_h += 1
                # Handle upper diagonal
                elif state.get_board()[i] == (row - (col - i)):
                    total_h += 1
                # Handle lower diagonal
                elif state.get_board()[i] == (row + (col - i)):
                    total_h += 1
        return total_h

    def expand_children(self, state):
        child_list = []
        num_cols = state.size()
        lowest_h = -1
        col_list = list(range(num_cols))
        self._rand_obj.shuffle(col_list)

        for i in col_list:
            for j in range(1, num_cols):
                temp_state = NQueens(num_cols, state._board[:], state.seed)
                old_q_p = temp_state.get_board()[i]
                temp_state.place_queen((old_q_p + j) % num_cols, i)
                temp_node = _local_node(temp_state, self.h(temp_state))
                if lowest_h < 0 or temp_node.h <= lowest_h:
                    lowest_h = temp_node.h
                    if temp_node.h < lowest_h or self._rand_obj.randint(0, 1) == 0:
                        child_list.insert(0, temp_node)
                    else:
                        child_list.append(temp_node)
                else:
                    child_list.append(temp_node)
        return child_list

    def _local_hill_climb(self):
        prev_node = self._curr_node
        curr_node = self._curr_node
        children = self.expand_children(self._curr_node.state)
        while curr_node.h > 0:
            if prev_node != curr_node:
                children = self.expand_children(curr_node.state)
            neighbor_node = children[0]
            if (neighbor_node is None) or (neighbor_node.h >= curr_node.h):
                break
            curr_node = neighbor_node
            curr_node.state.draw()
        self._curr_node = curr_node

    def _sim_anneal(self):
        prev_node = self._curr_node
        curr_node = self._curr_node
        children = self.expand_children(curr_node.state)
        t = 100
        while True:
            t -= 1
            if curr_node.h == 0 or t == 0:
                break
            if prev_node != curr_node:
                children = self.expand_children(curr_node.state)
            next_node = self._rand_obj.choice(children)
            E = next_node.h - curr_node.h
            if E < 0:
                curr_node = next_node
                curr_node.state.draw()
                t = 100
            else:
                sel_prob = math.exp(-E / t)
                choose_next = list(self._rand_obj.multinomial(1, [sel_prob, 1 - sel_prob]))
                if choose_next == [0, 1]:
                    curr_node = next_node
                    curr_node.state.draw()

        self._curr_node = curr_node

    def find_best_answer(self):
        self._local_hill_climb()
        # self._sim_anneal()


def main():
    seed = 10
    game_size = 4
    test_game = NQueens(game_size, None, seed)
    print("----Before searching for solution----")
    test_game.draw()
    print("--------")

    test_agent = NQueensAgent(test_game, seed)
    test_time = tm.time()
    test_agent.find_best_answer()
    test_time = tm.time() - test_time
    print("----After searching for solution----")
    test_agent.get_state().draw()
    print(test_agent.get_h())
    print(test_time)


if __name__ == "__main__":
    main()
