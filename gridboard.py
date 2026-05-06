import numpy as np


def rand_pair(start, end):
    return np.random.randint(start, end), np.random.randint(start, end)


class BoardPiece:
    def __init__(self, name, code, pos):
        self.name = name
        self.code = code
        self.pos = pos


class BoardMask:
    def __init__(self, name, mask, code):
        self.name = name
        self.mask = mask
        self.code = code

    def get_positions(self):
        return np.nonzero(self.mask)


def zip_positions_2d(positions):
    x_vals, y_vals = positions
    return list(zip(x_vals, y_vals))


class GridBoard:
    def __init__(self, size=4):
        self.size = size
        self.components = {}
        self.masks = {}

    def add_piece(self, name, code, pos=(0, 0)):
        self.components[name] = BoardPiece(name, code, pos)

    def add_mask(self, name, mask, code):
        self.masks[name] = BoardMask(name, mask, code)

    def move_piece(self, name, pos):
        move = True
        for _, mask in self.masks.items():
            if pos in zip_positions_2d(mask.get_positions()):
                move = False
        if move:
            self.components[name].pos = pos

    def render(self):
        dtype = "<U2"
        display = np.zeros((self.size, self.size), dtype=dtype)
        display[:] = " "
        for _, piece in self.components.items():
            display[piece.pos] = piece.code
        for _, mask in self.masks.items():
            display[mask.get_positions()] = mask.code
        return display

    def render_np(self):
        num_pieces = len(self.components) + len(self.masks)
        display = np.zeros((num_pieces, self.size, self.size), dtype=np.uint8)
        layer = 0
        for _, piece in self.components.items():
            pos = (layer,) + piece.pos
            display[pos] = 1
            layer += 1
        for _, mask in self.masks.items():
            x_vals, y_vals = mask.get_positions()
            z_vals = np.repeat(layer, len(x_vals))
            display[(z_vals, x_vals, y_vals)] = 1
            layer += 1
        return display


def add_tuple(a, b):
    return tuple([sum(x) for x in zip(a, b)])
