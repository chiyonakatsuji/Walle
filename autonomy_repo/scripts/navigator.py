#!/usr/bin/env python3
"""HW2 navigation: A*, cubic-spline planning, and HW1 feedback controllers."""
import numpy as np
import rclpy
from scipy.interpolate import splev, splrep

from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from utils import plot_line_segments

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        for i in range(len(x)):
            if x[i] < self.statespace_lo[i] or x[i] > self.statespace_hi[i]:
                return False
        return self.occupancy.is_free(x)
        #raise NotImplementedError("is_free not implemented")
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        #raise NotImplementedError("distance not implemented")
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use 
               (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx==0 and dy==0:
                    continue
                candidate = self.snap_to_grid((x[0] + dx * self.resolution, 
                                              x[1] + dy*self.resolution))
                if self.is_free(candidate):
                    neighbors.append(candidate)
        #raise NotImplementedError("get_neighbors not implemented")
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    '''def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)'''

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while len(self.open_set) > 0:
            x_current = self.find_best_est_cost_through()
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                     continue 
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh] = x_current
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh, self.x_goal)
        return False
        #raise NotImplementedError("solve not implemented")
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))

def compute_smooth_plan(path, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
    # Ensure path is a numpy array
    # path = np.asarray(AStar.path)

    # Compute and set the following variables:
    #   1. ts: 
    #      Compute an array of time stamps for each planned waypoint assuming some constant 
    #      velocity between waypoints. 
    #
    #   2. path_x_spline, path_y_spline:
    #      Fit cubic splines to the x and y coordinates of the path separately
    #      with respect to the computed time stamp array.
    #      Hint: Use scipy.interpolate.splrep
    
    ##### YOUR CODE STARTS HERE #####
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    ts = np.concatenate(([0.0], np.cumsum(segment_lengths / v_desired)))
    path = np.array(path)
    path_x_spline = splrep(ts, path[:, 0], k=3, s=spline_alpha)
    path_y_spline = splrep(ts, path[:, 1], k=3, s=spline_alpha)
    ###### YOUR CODE END HERE ######
    
    return TrajectoryPlan(
        path=path,
        path_x_spline=path_x_spline,
        path_y_spline=path_y_spline,
        duration=ts[-1],
    )

class Navigator(BaseNavigator):
    def __init__(self):
        super().__init__()
        self.kp = 2.0
        self.kpx = self.kpy = self.kdx = self.kdy = 2.0
        self.V_PREV_THRESH = 1e-4
        self.v_desired = 0.15
        self.spline_alpha = 0.005
        self.reset_tracking_history()

    def reset_tracking_history(self):
        self.V_prev = 0.0
        self.om_prev = 0.0
        self.t_prev = 0.0

    def compute_heading_control(self, state: TurtleBotState,
                                goal: TurtleBotState) -> TurtleBotControl:
        control = TurtleBotControl()
        control.omega = float(self.kp * wrap_angle(goal.theta - state.theta))
        return control

    def compute_trajectory_tracking_control(self, state: TurtleBotState,
                                            plan: TrajectoryPlan,
                                            t: float) -> TurtleBotControl:
        # The base class supplies elapsed plan time and manages mode transitions.
        # Clamp only the spline evaluation time, never extrapolate past its domain.
        t_eval = float(np.clip(t, 0.0, plan.duration))
        if t < self.t_prev:
            self.reset_tracking_history()
        dt = max(0.0, float(t) - self.t_prev)
        x_d, xd_d, xdd_d = [float(splev(t_eval, plan.path_x_spline, der=i))
                            for i in range(3)]
        y_d, yd_d, ydd_d = [float(splev(t_eval, plan.path_y_spline, der=i))
                            for i in range(3)]
        speed = max(self.V_prev, self.V_PREV_THRESH)
        c, s = np.cos(state.theta), np.sin(state.theta)
        xd, yd = speed * c, speed * s
        u1 = xdd_d + self.kpx * (x_d - state.x) + self.kdx * (xd_d - xd)
        u2 = ydd_d + self.kpy * (y_d - state.y) + self.kdy * (yd_d - yd)
        acceleration = u1 * c + u2 * s
        omega = (-u1 * s + u2 * c) / speed
        velocity = speed + dt * acceleration
        control = TurtleBotControl(v=float(velocity), omega=float(omega))
        # BaseController clips these commands before publishing. Store that
        # applied speed, as in HW1, rather than an unrealizable raw command.
        self.V_prev = float(np.clip(velocity, -self.v_max, self.v_max))
        self.om_prev = float(np.clip(omega, -self.om_max, self.om_max))
        self.t_prev = float(t)
        return control

    def compute_trajectory_plan(self, state, goal, occupancy,
                                resolution, horizon):
        if resolution <= 0 or horizon <= 0:
            return None
        start = np.array([state.x, state.y], dtype=float)
        target = np.array([goal.x, goal.y], dtype=float)
        # Interpret horizon as the full width of a square centered at the robot.
        lower, upper = start - horizon / 2, start + horizon / 2
        # StochOccupancyGrid2D clips its probability window at map edges;
        # explicit bounds keep empty out-of-map windows out of the graph.
        if hasattr(occupancy, 'origin_xy'):
            lower = np.maximum(lower, occupancy.origin_xy)
            upper = np.minimum(upper, occupancy.origin_xy
                               + (occupancy.size_xy - 1) * occupancy.resolution)
        planner = AStar(tuple(lower), tuple(upper), tuple(start), tuple(target),
                        occupancy, resolution=resolution)
        if not planner.solve() or len(planner.path) < 4:
            return None
        self.reset_tracking_history()
        return compute_smooth_plan(planner.path, self.v_desired, self.spline_alpha)


def main():
    rclpy.init()
    node = Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
