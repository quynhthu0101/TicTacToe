import random
from collections import namedtuple

import numpy as np

GameState = namedtuple('GameState', 'to_move, utility, board, moves')

# ______________________________________________________________________________
# MinMax Search


def minmax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states."""

    """
    Hàm này dự đoán nước đi tối ưu cho máy tính bằng cách thực hiện tìm kiếm theo chiều sâu 
    trên cây game tree, sử dụng thuật toán Minimax.
    """

    # xác định người chơi hiện tại của trò chơi bằng cách sử dụng phương thức to_move(state) của đối tượng Game ở đây là ttt
    player = game.to_move(state)

    def max_value(state):
        # Nếu trạng thái hiện tại là trạng thái kết thúc, nó trả về giá trị của trạng thái đó.
        if game.terminal_test(state):
            return game.utility(state, player)
        #  Nếu không khởi tạo giá trị v là âm vô cùng 
        v = -np.inf
        #  lặp qua mỗi nước đi có thể từ trạng thái hiện tại. 
        # Đối với mỗi nước đi, nó tính giá trị MIN-VALUE của trạng thái kế tiếp khi thực hiện nước đi đó và cập nhật giá trị v nếu cần.
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    #  tìm giá trị tối thiểu có thể đạt được từ một trạng thái cụ thể.
    def min_value(state):
        # Nếu trạng thái hiện tại là trạng thái kết thúc, nó trả về giá trị của trạng thái đó.
        if game.terminal_test(state):
            return game.utility(state, player)
        #  Nếu không khởi tạo giá trị v là dương vô cùng 
        v = np.inf
        # lặp qua mỗi nước đi có thể từ trạng thái hiện tại. 
        # Đối với mỗi nước đi, nó tính giá trị MAX-VALUE của trạng thái kế tiếp khi thực hiện nước đi đó và cập nhật giá trị v nếu cần.
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))
    # chọn ra nước đi tối ưu cho máy tính trong trạng thái hiện tại 
    # bằng cách sử dụng hàm max để chọn nước đi có giá trị tốt nhất trong danh sách các nước đi có thể thực hiện, 
    # dựa trên giá trị tối thiểu của trạng thái kế tiếp khi thực hiện nước đi a.
# ______________________________________________________________________________


def alpha_beta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    # xác định người chơi hiện tại của trò chơi bằng cách sử dụng phương thức to_move(state) của đối tượng Game ở đây là ttt
    player = game.to_move(state)

    # Functions used by alpha_beta
    # Nó tìm kiếm giá trị lớn nhất mà người chơi có thể đạt được từ trạng thái hiện tại.
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            # Gọi đệ quy hàm min_value để tính giá trị nhỏ nhất có thể đạt được sau nước đi đó của O.
            v = max(v, min_value(game.result(state, a), alpha, beta))
            #  Nếu v lớn hơn hoặc bằng beta, trả về v ngay lập tức (cắt tỉa).
            if v >= beta:   
                return v
            alpha = max(alpha, v)
        return v

    # Hàm này đại diện cho lượt của đối thủ. 
    # Nó tìm kiếm giá trị nhỏ nhất mà đối thủ có thể đạt được từ trạng thái hiện tại.
    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            # Gọi đệ quy hàm max_value để tính giá trị lớn nhất có thể đạt được sau nước đi đó của X.
            v = min(v, max_value(game.result(state, a), alpha, beta))
            #  Nếu v nhỏ hơn hoặc bằng alpha, trả về v ngay lập tức (cắt tỉa).
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:

    # best_score được khởi tạo bằng -np.inf để đảm bảo mọi giá trị khác đều lớn hơn.
    best_score = -np.inf
    beta = np.inf
    best_action = None
    # Duyệt qua tất cả các nước đi có thể xảy ra từ trạng thái hiện tại.
    for a in game.actions(state):
        # gọi hàm min_value để tính giá trị nhỏ nhất có thể đạt được của nước đi đó.
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v  
            best_action = a     #lưu lại nước đi đó
    return best_action


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None
    
    """
        Trước tiên, hàm kiểm tra danh sách các nước đi hợp lệ mà máy tính có thể thực hiện trong trạng thái hiện tại của trò chơi, 
    bằng cách gọi phương thức actions(state) của đối tượng Game ở đây là tictactoe.
        Nếu danh sách nước đi không rỗng (tức là còn nước đi hợp lệ để thực hiện), 
    hàm sẽ chọn một nước đi ngẫu nhiên từ danh sách này bằng cách sử dụng hàm random.choice(actions) từ thư viện random.
        nước đi được chọn ngẫu nhiên sẽ là nước đi được trả về bởi hàm.
    Nếu không có nước đi hợp lệ nào, hàm sẽ trả về None.
    """

def alpha_beta_player(game, state):
    return alpha_beta_search(state, game)


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""

    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    # Trạng thái hiện tại của trò chơi, Nước đi được thực hiện, được biểu diễn dưới dạng một cặp (x, y) tọa độ trên bảng cờ.
    def result(self, state, move): 
        # kiểm tra tính hợp lệ của nước đi. Nếu nước đi không hợp lệ (không nằm trong danh sách các nước đi có thể của trạng thái hiện tại), 
        # nó sẽ trả về trạng thái hiện tại mà không có tác động nào.
        if move not in state.moves:
            return state  # Illegal move has no effect
        
        """
        Nếu nước đi là hợp lệ, hàm sẽ tạo một bản sao của bảng hiện tại (biến board) để thay đổi trạng thái của trò chơi. 
        Bản sao này sẽ giúp tránh thay đổi trạng thái ban đầu của trò chơi.
        Nước đi được thêm vào bản sao của bảng cờ (board). 
        Sau đó, nước đi này sẽ được loại bỏ khỏi danh sách các nước đi hợp lệ (moves) để không được sử dụng lại.
        """
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)

        """
        trả về một trạng thái mới của trò chơi, được tạo từ các thông tin sau:
        to_move: Trong trò chơi Tic Tac Toe, lượt của người chơi luôn thay đổi giữa 'X' và 'O', nên nếu là lượt của 'X', lượt tiếp theo sẽ là của 'O', và ngược lại.
        utility: Giá trị utility của trạng thái mới, được tính toán bằng cách sử dụng phương thức compute_utility() để xác định kết quả của nước đi hiện tại.
        board: Bảng cờ mới sau khi thực hiện nước đi.
        moves: Danh sách các nước đi hợp lệ còn lại
        """
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        #  Gọi hàm Kiểm tra xem có bất kỳ hàng, cột hoặc đường chéo chính, phụ nào có đủ 3 con X / O của người chơi liên tiếp không.
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return +1 if player == 'X' else -1
        else:
            return 0

    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player."""
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k
