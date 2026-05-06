from gridboard import GridBoard, add_tuple, rand_pair


class Gridworld:
    def __init__(self, size=4, mode="static"):
        if size >= 4:
            self.board = GridBoard(size=size)
        else:
            self.board = GridBoard(size=4)

        self.board.add_piece("Player", "P", (0, 0))
        self.board.add_piece("Goal", "+", (1, 0))
        self.board.add_piece("Pit", "-", (2, 0))
        self.board.add_piece("Wall", "W", (3, 0))

        if mode == "static":
            self.init_grid_static()
        elif mode == "player":
            self.init_grid_player()
        else:
            self.init_grid_random()

    def init_grid_static(self):
        self.board.components["Player"].pos = (0, 3)
        self.board.components["Goal"].pos = (0, 0)
        self.board.components["Pit"].pos = (0, 1)
        self.board.components["Wall"].pos = (1, 1)

    def validate_board(self):
        player = self.board.components["Player"]
        goal = self.board.components["Goal"]
        wall = self.board.components["Wall"]
        pit = self.board.components["Pit"]

        all_positions = [player.pos, goal.pos, wall.pos, pit.pos]
        if len(all_positions) > len(set(all_positions)):
            return False

        corners = [(0, 0), (0, self.board.size), (self.board.size, 0), (self.board.size, self.board.size)]
        if player.pos in corners or goal.pos in corners:
            moves = [(0, 1), (1, 0), (-1, 0), (0, -1)]
            val_move_pl = [self.validate_move("Player", addpos) for addpos in moves]
            val_move_go = [self.validate_move("Goal", addpos) for addpos in moves]
            if 0 not in val_move_pl or 0 not in val_move_go:
                return False

        return True

    def init_grid_player(self):
        self.init_grid_static()
        self.board.components["Player"].pos = rand_pair(0, self.board.size)
        if not self.validate_board():
            self.init_grid_player()

    def init_grid_random(self):
        self.board.components["Player"].pos = rand_pair(0, self.board.size)
        self.board.components["Goal"].pos = rand_pair(0, self.board.size)
        self.board.components["Pit"].pos = rand_pair(0, self.board.size)
        self.board.components["Wall"].pos = rand_pair(0, self.board.size)
        if not self.validate_board():
            self.init_grid_random()

    def validate_move(self, piece, addpos=(0, 0)):
        outcome = 0
        pit = self.board.components["Pit"].pos
        wall = self.board.components["Wall"].pos
        new_pos = add_tuple(self.board.components[piece].pos, addpos)
        if new_pos == wall:
            outcome = 1
        elif max(new_pos) > (self.board.size - 1):
            outcome = 1
        elif min(new_pos) < 0:
            outcome = 1
        elif new_pos == pit:
            outcome = 2
        return outcome

    def makeMove(self, action):
        def check_move(addpos):
            if self.validate_move("Player", addpos) in [0, 2]:
                new_pos = add_tuple(self.board.components["Player"].pos, addpos)
                self.board.move_piece("Player", new_pos)

        if action == "u":
            check_move((-1, 0))
        elif action == "d":
            check_move((1, 0))
        elif action == "l":
            check_move((0, -1))
        elif action == "r":
            check_move((0, 1))

    def reward(self):
        if self.board.components["Player"].pos == self.board.components["Pit"].pos:
            return -1
        if self.board.components["Player"].pos == self.board.components["Goal"].pos:
            return 1
        return 0

    def display(self):
        return self.board.render()
