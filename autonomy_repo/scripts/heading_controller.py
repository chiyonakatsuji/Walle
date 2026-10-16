#!/usr/bin/env python3

import numpy as np
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class HeadingController(BaseHeadingController):
    """Rotate toward the goal heading using wrapped proportional error."""

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("kp", 2.0)
        #self.kp = 2.0
    
    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value

    def compute_control_with_goal(
    self, state: TurtleBotState, goal: TurtleBotState
    ) -> TurtleBotControl:
        err = wrap_angle(goal.theta - state.theta)
        control = TurtleBotControl()
        control.omega = float(np.multiply(self.kp, err))
        return control


if __name__ == "__main__":
    rclpy.init()
    node = HeadingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

