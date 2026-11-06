#!/usr/bin/env python3
import rclpy
from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl
from rclpy.parameter import Parameter

class PerceptionController(BaseController):
    def __init__(self):
        super().__init__("perception_controller")
        self.declare_parameter("active", True)
        self.stop_start = None
    @property
    def active(self) -> bool:
        return self.get_parameter("active").value
    def compute_control(self) -> TurtleBotControl:
        control = TurtleBotControl()
        elapsed = 0.0
        if self.active:
            self.stop_start = None
            control.omega = 0.5
        else:
            now = self.get_clock().now()

            if self.stop_start is None:
                self.stop_start = now

            elapsed = (now - self.stop_start).nanoseconds / 1e9

        if elapsed >= 5.0:
            self.set_parameters([Parameter("active", value=True)])
            self.stop_start = None  
            control.omega = 0.5

        return control

if __name__ == "__main__":
    rclpy.init()
    node = PerceptionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
