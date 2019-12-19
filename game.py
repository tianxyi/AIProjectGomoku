
from __future__ import print_function
import numpy as np

BLANK = 0

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.col = int(kwargs.get('col', 8))
        self.row = int(kwargs.get('row', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type #self.current_player = self.players[start_player]
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):# start 0 is x, start  1 is o
        if self.col < self.n_in_row or self.row < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player ,player x first or player o then
        # keep available moves in a list
        self.availables = list(range(self.row * self.col))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        0 1 2 
        3 4 5 
        6 7 8
        move 3 is like (3/3, 3%3) (1,0)
        """

        row = move // self.col
        col = move % self.col
        return [row, col]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        row = location[0]
        col = location[1]
        move = row * self.row + col
        if move not in range(self.row * self.col): #不在格子内
            return -1
        return move

    def current_state(self):

        #state = np.array([[BLANK]*self.col]*self.row)
        square_state = np.zeros((4, self.col, self.row))
        if self.states: #states not zero
            moves, players = np.array(list(zip(*self.states.items())))#states key value 
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.col,
                            move_curr % self.row] = 1.0
            square_state[1][move_oppo // self.col,
                            move_oppo % self.row] = 1.0
                        
            # indicate the last move location
            square_state[2][self.last_move // self.col,
                            self.last_move % self.row] = 1.0
                        
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate who play first
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player  #self.current_player = self.players[start_player]
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1] #
            else self.players[1]
        )
        self.last_move = move # 0 12345678

    def has_a_winner(self):# 
        col = self.col
        row = self.row
        states = self.states
        n = self.n_in_row

        moved = list(set(range(col * row)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // col
            w = m % col
            player = states[m]

            if (w in range(col - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(row - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * col, col))) == 1):
                return True, player

            if (w in range(col - n + 1) and h in range(row - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (col + 1), col + 1))) == 1):
                return True, player

            if (w in range(n - 1, col) and h in range(row - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (col - 1), col - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner#you winner game end
        elif not len(self.availables):# len(0)
            return True, -1#wu winner tie game end
        return False, -1#wu winner game not end

    def get_current_player(self):
        return self.current_player#player1 player2


class Game(object):


    def __init__(self, board, **kwargs):
        self.board = board
###

    def start_play(self, player1, player2, start_player=0, is_shown=1):

        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players #[1,2]
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            end, winner = self.board.game_end() #0 1
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner
###
    def start_self_play(self, player, is_shown=0, temp=1e-3):

        self.board.init_board()
        p1, p2 = self.board.players#[1,2]
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
          #  if is_shown:
          #      self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:#tie or have winner
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:#have winner, trans the winner to 1, loser to -1
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                
                return winner, zip(states, mcts_probs, winners_z)#
