
import time
import numpy  as np
from BaseAI_3 import BaseAI

inf = float('inf')
max_depth = 100

class PlayerAI(BaseAI):

    def __init__(self):
        self.start_time = 0.0
        self.time_out   = False

    def getMove(self, grid):
        """
        :param grid: Input Node
        :return:     Best Move Node
        """
        self.start_time = time.clock()
        self.time_out   = False
        move = None
        # Iterative Deepening Search
        for d in range(max_depth):
            maxChild, _ = self.maximize(grid, -inf, inf, 0, d)
            if self.time_out:
                break
            elif maxChild is not None:
                move = maxChild
        return move

    def timeout(self):
        """
        :return: Terminate Time Test
        """
        if time.clock() - self.start_time >= 0.2:
            self.time_out = True
            return True
        else:
            return False

    def maximize(self, grid, alpha, beta, depth, max_depth):
        """
        :param grid:       Current move node
        :param alpha:      Largest value for Max across seen children (initial = - inf)
        :param beta:       Lowest  value for Min across seen children (initial = + inf)
        :param depth:      Current depth
        :param max_depth:  Iterative Deepening Depth
        :return:           <State, Utility>
        """
        if self.timeout():
            return None, -inf
        if depth > max_depth:
            return None, self.heuristic(grid)

        maxChild, maxUtility = None, -inf

        for child, child_grid in grid.getAvailableMoves():

            utility = self.expected(child_grid, alpha, beta, depth, max_depth)

            if utility > maxUtility:
                maxChild, maxUtility = child, utility

            if maxUtility >= beta:                        # Alpha-Beta Pruning
                break

            if maxUtility > alpha:                        # MAX: update alpha
                alpha = maxUtility

        return maxChild, maxUtility

    def expected(self, grid, alpha, beta, depth, max_depth):
        """
        Get Expected Utility Values for Current State
        """
        e1 = self.minimize(grid, alpha, beta, depth+1, max_depth, 2)
        e2 = self.minimize(grid, alpha, beta, depth+1, max_depth, 4)
        return 0.9 * e1 + 0.1 * e2

    def minimize(self, grid, alpha, beta, depth, max_depth, tile):
        """
        :param grid:       Current move node
        :param alpha:      Largest value for Max across seen children (initial = - inf)
        :param beta:       Lowest  value for Min across seen children (initial = + inf)
        :param depth:      Current depth
        :param max_depth:  Iterative Deepening Depth
        :param tile:       MIN's best action conditional on Tile Value
        :return:           <Utility>
        """
        if self.timeout():
            return inf
        if depth > max_depth:
            return self.heuristic(grid)

        minUtility = inf

        for cell in grid.getAvailableCells():
            child_grid = grid.clone()
            child_grid.setCellValue(cell, tile)

            _, utility = self.maximize(child_grid, alpha, beta, depth+1, max_depth)

            if utility < minUtility:
                minUtility = utility

            if minUtility <= alpha:                # Alpha-Beta Pruning
                break

            if minUtility < beta:                  # MIN: update beta
                beta = minUtility

        return minUtility

    def heuristic(self, grid):
        """
        Define the heuristic function (based on 6 heuristics)
        """
        w_max = 1.0                  # Weight: log2(max value of tiles)
        w_avg = 1.0                  # Weight: log2(avg value of tiles)
        w_emp = 2.0                  # Weight: the number of free tiles
        w_smt = 1.0                  # Weight: the absolute differences of nonzero tiles
        w_mnt = 1.0                  # Weight: the monotonicity by row and column
        w_crn = 2.0                  # Weight: max tile in the corner
        size = grid.size

        mat = np.asarray(grid.map)
        emp = (mat == 0).sum()
        max = np.log2(mat.max())
        avg = np.log2(mat.sum()/(size**2 - emp))

        # Calculate Smoothness
        m = mat.astype('float')
        m[m == 0] = np.nan                                                          # Exclude free (0) tiles
        smt = (np.diff(m, axis=0) == 0).sum() + (np.diff(m, axis=1) == 0).sum()     # Count equal neighbors by row/col
        d1, d2 = abs(np.diff(mat, axis=0)), abs(np.diff(mat, axis=1))               # Minus absolute differences
        d1[d1 == 0] = 1
        d2[d2 == 0] = 1
        smt = smt - np.log2(d1).sum() - np.log2(d2).sum()                           # H: Minimizes the difference

        # Calculate Monotonicity
        m1, m2 = np.diff(mat, axis=0), np.diff(mat, axis=1)                         # Difference by row/col
        m1[m1 > 0], m1[m1 < 0] = 1, -1
        m2[m2 > 0], m2[m2 < 0] = 1, -1
        mnt = abs(m1.sum(axis=0)).sum() + abs(m2.sum(axis=1)).sum()                 # H: Maximizes the monotonicity

        # Test Corner = Max
        crn = 0
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for c in corners:
            if mat.max() == grid.getCellValue(c):
                crn = 1

        return w_max * max + w_avg * avg + w_emp * emp + w_smt * smt + w_mnt * mnt + w_crn * crn