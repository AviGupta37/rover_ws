import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class Object3D(Node):

    def __init__(self):
        super().__init__('object_3d')

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        self.depth_image = None
        self.target_class = "bottle"

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx_intr = None
        self.cy_intr = None

        # Subscriptions
        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

    # ---------------- CAMERA INFO ----------------
    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx_intr = msg.k[2]
        self.cy_intr = msg.k[5]

    # ---------------- DEPTH ----------------
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # ---------------- IMAGE ----------------
    def image_callback(self, msg):

        if self.depth_image is None or self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        results = self.model(frame, conf=0.4)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]

                if label == self.target_class:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw bounding box
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # Center pixel
                    u = (x1 + x2) // 2
                    v = (y1 + y2) // 2

                    depth = self.depth_image[v, u] * 0.001  # mm → m

                    if depth > 0:

                        # Convert to 3D
                        X = (u - self.cx_intr) * depth / self.fx
                        Y = (v - self.cy_intr) * depth / self.fy
                        Z = depth

                        # Log
                        self.get_logger().info(
                            f"🎯 3D Position → X:{X:.2f}, Y:{Y:.2f}, Z:{Z:.2f}"
                        )

                        # Show on screen
                        cv2.putText(frame,
                                    f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}",
                                    (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0,0,255),
                                    2)

                        # Draw center point
                        cv2.circle(frame, (u, v), 5, (255,0,0), -1)

        cv2.imshow("Detection", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = Object3D()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
