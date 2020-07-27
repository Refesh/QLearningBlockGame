import numpy as np


class Character:
    def __init__(self, GRID_SIZE):
        self.GRID_SIZE = GRID_SIZE
        self.x = np.random.randint(0, GRID_SIZE)
        self.y = np.random.randint(0, GRID_SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def quadrant(self):
        half = (self.GRID_SIZE - 1) // 2
        if self.x > half:
            if self.y > half:
                return 4
            else:
                return 2
        elif self.y > half:
            return 3
        return 1

    def action(self, choice):
        if choice == 0:
            self.move(1, 1)
        elif choice == 1:
            self.move(-1, 1)
        elif choice == 2:
            self.move(1, -1)
        elif choice == 3:
            self.move(-1, -1)

    def move(self, x=0, y=0):
        self.x += x
        self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x >= self.GRID_SIZE:
            self.x = self.GRID_SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y >= self.GRID_SIZE:
            self.y = self.GRID_SIZE - 1

    def __copy__(self):
        copy = Character()
        copy.x = self.x
        copy.y = self.y
        return copy

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y