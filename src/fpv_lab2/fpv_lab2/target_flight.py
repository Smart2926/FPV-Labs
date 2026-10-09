import math
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from ardupilot_msgs.msg import Status
from ardupilot_msgs.srv import ModeSwitch, ArmMotors, Takeoff
from pymavlink import mavutil


class TargetFlight(Node):
    TAKEOFF_HEIGHT = 2.0
    DX = 5.2
    DY = 1.1
    TOLERANCE = 0.2
    MAX_SPEED = 0.8
    MIN_SPEED = 0.08
    KP = 0.6

    def __init__(self):
        super().__init__("target_flight")

        self.status = None
        self.odometry = None
        self.last_odom_time = None
        self.start = None
        self.target = None

        self.create_subscription(
            Status, "/ap/v1/status", self.status_callback, 10
        )
        self.create_subscription(
            Odometry, "/odometry", self.odometry_callback, 10
        )

        self.prearm = self.create_client(
            Trigger, "/ap/v1/prearm_check"
        )
        self.mode = self.create_client(
            ModeSwitch, "/ap/v1/mode_switch"
        )
        self.arm = self.create_client(
            ArmMotors, "/ap/v1/arm_motors"
        )
        self.takeoff = self.create_client(
            Takeoff, "/ap/v1/experimental/takeoff"
        )

        self.mav = None

    def status_callback(self, msg):
        self.status = msg

    def odometry_callback(self, msg):
        self.odometry = msg
        self.last_odom_time = time.monotonic()

    def log(self, message):
        self.get_logger().info(message)

    def wait_for(self, condition, timeout, description):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if condition():
                return
        raise RuntimeError(f"Timeout: {description}")

    def call_service(self, client, request, description):
        self.wait_for(
            client.service_is_ready, 30.0, description + " service"
        )
        future = client.call_async(request)
        self.wait_for(
            lambda: future.done(), 15.0, description
        )
        if future.exception() is not None:
            raise RuntimeError(str(future.exception()))
        return future.result()

    def position(self):
        p = self.odometry.pose.pose.position
        return p.x, p.y, p.z

    def yaw(self):
        q = self.odometry.pose.pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def odometry_fresh(self):
        return (
            self.last_odom_time is not None
            and time.monotonic() - self.last_odom_time < 2.0
        )

    def send_velocity(self, vx, vy):
        yaw = self.yaw()

        forward = vx * math.cos(yaw) + vy * math.sin(yaw)
        right = vx * math.sin(yaw) - vy * math.cos(yaw)

        self.mav.mav.set_position_target_local_ned_send(
            0,
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0x0DC7,
            0.0, 0.0, 0.0,
            forward, right, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0
        )

    def stop_horizontal(self):
        for _ in range(5):
            self.send_velocity(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.1)

    def switch_mode(self, mode_number):
        request = ModeSwitch.Request()
        request.mode = mode_number
        response = self.call_service(
            self.mode, request, "Mode switch"
        )
        if not response.status or response.curr_mode != mode_number:
            raise RuntimeError("Mode switch rejected")

    def run(self):
        self.log("Waiting for ArduPilot and odometry")

        self.wait_for(
            lambda: self.status is not None
            and self.odometry_fresh(),
            60.0,
            "ArduPilot status and odometry"
        )

        self.start = self.position()
        x0, y0, z0 = self.start
        self.target = (x0 - self.DX, y0 - self.DY)

        self.log(
            f"Initial position: x={x0:.3f}, y={y0:.3f}, z={z0:.3f}"
        )
        self.log(
            f"Target position: x={self.target[0]:.3f}, "
            f"y={self.target[1]:.3f}"
        )

        self.log("Connecting to MAVLink UDP 14550")
        self.mav = mavutil.mavlink_connection(
            "udpin:127.0.0.1:14550"
        )
        heartbeat = self.mav.wait_heartbeat(timeout=15)
        if heartbeat is None:
            raise RuntimeError("MAVLink heartbeat not received")
        self.log("MAVLink connected")

        self.log("Running pre-arm check")
        response = self.call_service(
            self.prearm, Trigger.Request(), "Pre-arm check"
        )
        if not response.success:
            raise RuntimeError(
                f"Pre-arm check failed: {response.message}"
            )
        self.log("Pre-arm check passed")

        self.switch_mode(4)
        self.log("GUIDED mode enabled")

        request = ArmMotors.Request()
        request.arm = True
        response = self.call_service(
            self.arm, request, "Arming"
        )
        if not response.result:
            raise RuntimeError("Arming rejected")

        self.log("Motors armed")

        request = Takeoff.Request()
        request.alt = self.TAKEOFF_HEIGHT
        response = self.call_service(
            self.takeoff, request, "Takeoff"
        )
        if not response.status:
            raise RuntimeError("Takeoff rejected")

        self.log("Takeoff accepted; waiting for altitude")

        self.wait_for(
            lambda: self.odometry_fresh()
            and self.position()[2] - z0 >= self.TAKEOFF_HEIGHT - 0.15,
            90.0,
            "Takeoff altitude"
        )

        self.log("Takeoff altitude reached")
        self.log("Starting horizontal navigation")

        deadline = time.monotonic() + 120.0
        reached_since = None
        last_log = 0.0

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            if not self.odometry_fresh():
                raise RuntimeError("Odometry data lost")
            if self.status is None or not self.status.armed:
                raise RuntimeError("Drone unexpectedly disarmed")

            x, y, z = self.position()
            ex = self.target[0] - x
            ey = self.target[1] - y
            distance = math.hypot(ex, ey)

            if distance <= self.TOLERANCE:
                self.send_velocity(0.0, 0.0)
                if reached_since is None:
                    reached_since = time.monotonic()
                if time.monotonic() - reached_since >= 2.0:
                    break
            else:
                reached_since = None
                speed = min(
                    self.MAX_SPEED,
                    max(self.MIN_SPEED, self.KP * distance)
                )
                vx = speed * ex / distance
                vy = speed * ey / distance
                self.send_velocity(vx, vy)

            if time.monotonic() - last_log >= 2.0:
                self.log(
                    f"Position: x={x:.2f}, y={y:.2f}, "
                    f"distance={distance:.2f} m"
                )
                last_log = time.monotonic()
        else:
            raise RuntimeError("Navigation timeout")

        self.stop_horizontal()
        x, y, z = self.position()
        error = math.hypot(
            self.target[0] - x,
            self.target[1] - y
        )

        self.log("Target reached")
        self.log(
            f"Position before landing: x={x:.3f}, y={y:.3f}"
        )
        self.log(f"Position error: {error:.3f} m")
        self.log("Horizontal movement stopped")

        self.switch_mode(9)
        self.log("LAND mode enabled")

        self.wait_for(
            lambda: self.status is not None
            and not self.status.flying,
            90.0,
            "Landing"
        )

        self.log("Landing completed")
        self.log("Flight completed successfully")


def main(args=None):
    rclpy.init(args=args)
    node = TargetFlight()

    try:
        node.run()
    except (Exception, KeyboardInterrupt) as error:
        node.get_logger().error(f"Flight interrupted: {error}")

        if node.mav is not None and node.odometry_fresh():
            try:
                node.stop_horizontal()
            except Exception:
                pass

        if node.status is not None and node.status.armed:
            try:
                node.switch_mode(9)
                node.log("Emergency LAND mode requested")
            except Exception:
                node.get_logger().error(
                    "Automatic LAND request failed"
                )
    finally:
        if node.mav is not None:
            node.mav.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
