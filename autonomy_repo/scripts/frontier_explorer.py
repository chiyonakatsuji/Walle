#!/usr/bin/env python3
"""
Frontier exploration node for the AA174A TurtleBot stack.

Data flow:
  /map   (nav_msgs/OccupancyGrid)    --> map_callback         --> self.occupancy
  /state (asl_tb3_msgs/TurtleBotState) --> state_callback     --> self.state
  /nav_success (std_msgs/Bool)       --> nav_success_callback --> choose next frontier
  /detector_bool (std_msgs/Bool)     --> detector_callback    --> stop sign pause
  /cmd_nav (asl_tb3_msgs/TurtleBotState) <-- goal for the navigator

The navigator itself holds still for 5 s on a stop sign (it also listens to
/detector_bool). The explorer just pauses its own decision making meanwhile.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.signal import convolve2d

from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D   # VERIFY: same import as the base navigator


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")

        # ---------------- Tunable parameters ----------------
        self.declare_parameter("window_size", 13)            # side length of heuristic window (cells)
        self.declare_parameter("unknown_frac_min", 0.2)      # rule 1
        self.declare_parameter("unoccupied_frac_min", 0.3)   # rule 3
        self.declare_parameter("occupied_thresh", 0.5)       # prob >= this counts as occupied
        self.declare_parameter("blacklist_radius", 0.3)      # [m] skip frontiers near failed goals
        self.declare_parameter("min_goal_dist", 0.15)        # [m] skip frontiers we are already at
        self.declare_parameter("stop_duration", 5.0)         # [s] how long to stop at a stop sign
        self.declare_parameter("stop_cooldown", 8.0)         # [s] ignore the detector after resuming
        self.declare_parameter("retry_delay", 1.0)           # [s] wait after a failed plan before the next goal
        self.declare_parameter("max_blacklist_clears", 3)    # give up after clearing the blacklist this many times

        # ---------------- State ----------------
        self.occupancy = None
        self.state = None
        self.started = False
        self.finished = False
        self.failed_goals = []      # list of np.array([x, y]) that the planner could not reach
        self.current_goal = None
        self.blacklist_clears = 0
        self.retry_timer = None

        # Stop sign state (Task 4.1)
        self.paused = False
        self.deferred_result = None     # navigator result that arrived during a pause
        self.last_resume_time = None
        self.pause_timer = None

        # ---------------- ROS interfaces ----------------
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.create_subscription(Bool, "/nav_success", self.nav_success_callback, 10)
        self.create_subscription(Bool, "/detector_bool", self.detector_callback, 10)

        # Checks once per second whether we can start (map + pose both received).
        self.start_timer = self.create_timer(1.0, self.try_start)

        self.get_logger().info("Frontier explorer ready, waiting for /map and /state ...")

    # ================================================================== #
    # Callbacks
    # ================================================================== #
    def state_callback(self, msg: TurtleBotState):
        self.state = msg

    def map_callback(self, msg: OccupancyGrid):
        # VERIFY: argument names match how the base navigator constructs this object.
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=5,
            probs=msg.data,
        )

    def try_start(self):
        if self.started:
            return
        if self.occupancy is None or self.state is None:
            return
        self.started = True
        self.start_timer.cancel()
        self.get_logger().info("Map and pose received, starting exploration.")
        self.send_next_goal()

    def nav_success_callback(self, msg: Bool):
        # While paused, remember the result and handle it after the pause.
        if self.paused:
            self.deferred_result = msg
            return
        if not self.started or self.finished:
            return

        if msg.data:
            self.get_logger().info("Reached frontier goal.")
            self.blacklist_clears = 0
            self.send_next_goal()
            return

        # Planning failed: remember this goal so we don't pick it again, then wait
        # a moment (map and robot pose change) instead of instantly trying the next one.
        if self.current_goal is not None:
            self.failed_goals.append(self.current_goal.copy())
            self.get_logger().warn(
                f"Planning failed for ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f}), "
                f"blacklisting it ({len(self.failed_goals)} total)."
            )
        self.schedule_retry(float(self.get_parameter("retry_delay").value))

    def schedule_retry(self, delay):
        """Call send_next_goal once after `delay` seconds (at most one pending)."""
        if self.retry_timer is not None:
            return

        def _retry():
            self.retry_timer.cancel()
            self.destroy_timer(self.retry_timer)
            self.retry_timer = None
            if self.finished:
                return
            if self.paused:
                self.schedule_retry(1.0)     # try again after the stop sign pause
                return
            self.send_next_goal()

        self.retry_timer = self.create_timer(delay, _retry)

    # ================================================================== #
    # Stop sign handling (Task 4.1)
    # ================================================================== #
    def detector_callback(self, msg: Bool):
        if not msg.data or self.paused or not self.started or self.finished:
            return

        # Cooldown so we don't re-trigger on the same sign right after resuming.
        if self.last_resume_time is not None:
            cooldown = float(self.get_parameter("stop_cooldown").value)
            since = (self.get_clock().now() - self.last_resume_time).nanoseconds * 1e-9
            if since < cooldown:
                return

        duration = float(self.get_parameter("stop_duration").value)
        self.paused = True
        self.deferred_result = None
        self.get_logger().info(f"Stop sign detected: pausing exploration for {duration:.0f} s.")
        self.pause_timer = self.create_timer(duration, self.resume)

    def resume(self):
        # The timer is periodic, so cancel it here to make it one-shot.
        if self.pause_timer is not None:
            self.pause_timer.cancel()
            self.destroy_timer(self.pause_timer)
            self.pause_timer = None

        self.paused = False
        self.last_resume_time = self.get_clock().now()
        self.get_logger().info("Resuming exploration.")

        # The navigator keeps driving to its current goal on its own. Only step in
        # if it finished (or failed) a goal while we were paused.
        if self.deferred_result is not None:
            msg = self.deferred_result
            self.deferred_result = None
            self.nav_success_callback(msg)

    # ================================================================== #
    # Exploration logic (ported from HW4 Problem 2)
    # ================================================================== #
    def explore(self):
        """Return frontier states, shape (N, 2), in (x, y) world coordinates."""
        occupancy = self.occupancy
        window_size = int(self.get_parameter("window_size").value)
        unknown_frac_min = float(self.get_parameter("unknown_frac_min").value)
        unoccupied_frac_min = float(self.get_parameter("unoccupied_frac_min").value)
        occ_thresh = float(self.get_parameter("occupied_thresh").value)

        probs = np.asarray(occupancy.probs, dtype=float)

        # ROS OccupancyGrid uses -1 (unknown) and 0..100 (known). The notebook used 0..1.
        # Rescale known cells to 0..1 so the same thresholds work in both places.
        if probs.max() > 1.0:
            known = probs >= 0
            probs[known] = probs[known] / 100.0

        unknown = (probs == -1).astype(float)
        occupied = (probs >= occ_thresh).astype(float)
        unoccupied = ((probs >= 0) & (probs < occ_thresh)).astype(float)

        kernel = np.ones((window_size, window_size))
        unknown_count = convolve2d(unknown, kernel, mode="same")
        occupied_count = convolve2d(occupied, kernel, mode="same")
        unoccupied_count = convolve2d(unoccupied, kernel, mode="same")

        unknown_frac = unknown_count / window_size**2
        unoccupied_frac = unoccupied_count / window_size**2

        frontier_mask = (
            (unknown_frac >= unknown_frac_min)
            & (occupied_count == 0)
            & (unoccupied_frac >= unoccupied_frac_min)
            & (unoccupied == 1)          # the cell itself must be known free
        )

        frontier_grid = np.argwhere(frontier_mask)[:, ::-1]   # (row, col) -> (x, y)
        if len(frontier_grid) == 0:
            return np.zeros((0, 2))

        frontier_states = occupancy.grid2state(frontier_grid)
        return np.asarray(frontier_states).reshape(-1, 2)

    def filter_frontiers(self, frontier_states, current_xy):
        """Drop frontiers near blacklisted goals or right on top of the robot."""
        keep = np.ones(len(frontier_states), dtype=bool)

        blacklist_radius = float(self.get_parameter("blacklist_radius").value)
        for bad in self.failed_goals:
            keep &= np.linalg.norm(frontier_states - bad, axis=1) > blacklist_radius

        min_goal_dist = float(self.get_parameter("min_goal_dist").value)
        keep &= np.linalg.norm(frontier_states - current_xy, axis=1) > min_goal_dist

        return frontier_states[keep]

    def send_next_goal(self):
        if self.occupancy is None or self.state is None or self.finished:
            return

        current_xy = np.array([self.state.x, self.state.y])

        raw_frontiers = self.explore()
        frontier_states = self.filter_frontiers(raw_frontiers, current_xy)
        self.get_logger().info(
            f"Frontiers: {len(raw_frontiers)} found, {len(frontier_states)} usable, "
            f"{len(self.failed_goals)} blacklisted goals."
        )

        if len(frontier_states) == 0:
            max_clears = int(self.get_parameter("max_blacklist_clears").value)
            if self.blacklist_clears < max_clears:
                # Nothing usable right now. The map may still be growing, or goals may
                # have been blacklisted after a failed plan. Forget the blacklist and
                # look again shortly instead of ending the run.
                self.blacklist_clears += 1
                self.failed_goals = []
                self.get_logger().warn(
                    f"No usable frontiers ({len(raw_frontiers)} found). Re-checking in 3 s "
                    f"({self.blacklist_clears}/{max_clears})."
                )
                self.schedule_retry(3.0)
                return
            self.finished = True
            self.current_goal = None
            self.get_logger().info("Finished exploring")
            return

        distances = np.linalg.norm(frontier_states - current_xy, axis=1)
        goal_xy = frontier_states[np.argmin(distances)]
        self.current_goal = goal_xy

        goal = TurtleBotState()
        goal.x = float(goal_xy[0])
        goal.y = float(goal_xy[1])
        # Face the direction we are driving, so the camera/lidar points toward new space.
        goal.theta = float(np.arctan2(goal_xy[1] - current_xy[1], goal_xy[0] - current_xy[0]))
        self.cmd_nav_pub.publish(goal)

        self.get_logger().info(
            f"New frontier goal: ({goal.x:.2f}, {goal.y:.2f}), "
            f"{distances.min():.2f} m away, {len(frontier_states)} candidates."
        )


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()