#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import Int64, Bool, String
from geometry_msgs.msg import Twist

class Control(Node):
    def __init__(self) -> None:
				# initialize base class (must happen before everything else)
        super().__init__("control")  # initialize base class
				
        # a heartbeat counter
        self.hb_counter = 0

        # create publisher with
        #   self.create_publisher(<msg type>, <topic>, <qos>)
        self.control_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        

        # create a timer
		#   self.create_timer(<second>, <callback>)
        self.hb_timer = self.create_timer(0.2, self.ctrl_callback)

		# create subscription with
		#   self.create_subscription(<msg type>, <topic>, <callback>, <qos>)
        self.kill_sub = self.create_subscription(Bool, "/kill", self.kill_callback, 10)

    def ctrl_callback(self) -> None:
        """ control callback triggered by the timer """
        # construct control message
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = 0.4
        self.get_logger().info("sending constant control...")
        self.control_pub.publish(msg)

    def kill_callback(self, msg: Bool) -> None:
        """kill callback triggered by subscription """
        if msg.data:
            self.get_logger().fatal("Control stopped")
            self.hb_timer.cancel()
            self.control_pub.publish(Twist())


if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = Control()  # instantiate the control node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context