import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import logging
from typing import Optional, Dict, Any
import asyncio
import time

from physical_object import PhysicalObject
from camera import CameraCapture


logger = logging.getLogger(__name__)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

# Silence Ultralytics logger
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class TableSpace:
    """
    This class handles video input, performs perspective transformation, tracks objects,
    and manages obstacle detection in a defined scene (table space).

    Detects and tracks different types of objects:
    - Robotic objects (with uuid)
    - Pushable objects (without uuid)
    - Landmark objects (used for reasoning and as map) 
    """

    def __init__(self,
                 top_down_capture: CameraCapture,
                 scene_config_path,
                 video : str = None, 
                 ):

        self.top_down_capture = top_down_capture
        with open(scene_config_path, 'r') as scene_config_yaml:
            config = yaml.safe_load(scene_config_yaml)

        # Extract source coordinates for perspective transform
        src_points = np.array([
            config["source_coords"]['top_l'],
            config["source_coords"]['top_r'],
            config["source_coords"]['bot_r'],
            config["source_coords"]['bot_l']
        ], dtype=np.float32)

        # Extract destination coordinates for perspective transform
        dst_points = np.array([
            config["destination_coords"]['top_l'],
            config["destination_coords"]['top_r'],
            config["destination_coords"]['bot_r'],
            config["destination_coords"]['bot_l']
        ], dtype=np.float32)

        # Obtain mask and background images for the scene
        filepath_mask = config['table_mask']
        filepath_bg = config['table_background']

        # table_mask should be 255 in areas that we want to keep (i.e. table area)
        self.table_mask = cv2.bitwise_not(cv2.imread(filepath_mask, cv2.IMREAD_GRAYSCALE))
        self.table_mask = cv2.resize(self.table_mask, (640, 360), interpolation=cv2.INTER_AREA)

        # If true, we will flip image 180 degrees
        self.should_flip = False if 'flip' not in config else config['flip']

        # Compute the homography matrix (used for perspective transformation)
        self.homography_matrix, _ = cv2.findHomography(src_points, dst_points)

        # Store plain background (use for bg subtration)
        self.background = self.transform_table(cv2.imread(filepath_bg))
        self.background = cv2.resize(self.background, (640, 360), interpolation=cv2.INTER_AREA)

        # Store transformed masked area (we will superimpose obstacles on this)
        # empty_table_map will be 0 in free areas, 255 elsewhere
        self.empty_table_map = cv2.threshold(
            self.transform_table(self.table_mask), 127, 255, cv2.THRESH_BINARY_INV)[1]

        # for debugging
        self.custom_preview = None

        self.yolo_object_detection_model = config['obj_detection_model']

        self.yolo_person_detection_model = config['person_detection_model']
        self.last_updated = 0

        self.yolo8n = YOLO(self.yolo_person_detection_model)
        self.yolo8n_names = self.yolo8n.names

        # Load the YOLO11 model
        self.yolo_obb = YOLO(self.yolo_object_detection_model)
        self.yolo_class_mapping = config['yolo_class_mapping']

        # update detected objects (as sets for current-frame representation)
        self.detected_robotic_objects = set()
        self.detected_pushable_objects = set()
        self.detected_landmark_objects = set()
        # initialize list of known objects
        self.known_objects: list[PhysicalObject] = []
        # self.gemini_detected_objects: list[Dict[str, Any]] = []

        # read known objects once
        for object_name in config["active_objects"]:
            object_info = config["active_objects"][object_name]
            ob = PhysicalObject(
                name = object_name,
                uuid = object_info.get('uuid', ''),
                device_name = object_info.get('device_name', ''),
                heading_offset = object_info.get('heading_offset', 0.0))

            self.known_objects.append(ob)
        for object_name in config["non_active_objects"]:
            ob = PhysicalObject(name = object_name)
            self.known_objects.append(ob)



        # initialize frame counter (for control loop)
        self.frame_cnt = 0

        # for debugging?
        self.obstacle_maps: dict = {}
        self.object_targets: dict = {}
        self.user: Optional[Dict[str, Any]] = None  # Initialize user as None
        self.object_actions: dict = {} # tracks actions to be executed, modified by control loop

        # Cache the latest transformed frame to avoid double transformation
        self.latest_transformed_frame = None

        # Person detection throttling (seconds)
        self.person_detection_interval = 2.0
        self._last_person_detection_ts = 0.0

        self.background_tasks = set()


    async def loop(self):
        """
        Continuously observes the table state by capturing frames and tracking objects.
        """
        # last_person_detection_time = 0
        await self.top_down_capture.ready()
        while True:
            task = asyncio.create_task(self.track_in_frame())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            await asyncio.sleep(0.1)

    async def track_in_frame(self):
        frame = self.transform_frame()

        # Cache the transformed frame for display to avoid double transformation
        self.latest_transformed_frame = frame.copy()
        # Throttle person detection to reduce load
        now_ts = time.time()
        if (now_ts - self._last_person_detection_ts) >= self.person_detection_interval:
            self.track_single_person(frame)
            self._last_person_detection_ts = now_ts

        self.track_known_objects(frame)
        self.construct_map(frame)

    def transform_table(self, frame, use_mask=False):
        image_masked = cv2.bitwise_and(frame, frame, mask=self.table_mask) if use_mask else frame
        image_warped = cv2.warpPerspective(
            image_masked, self.homography_matrix, (frame.shape[1], frame.shape[0]))
        # image_resized = cv2.resize(image_warped, (640, 360), interpolation=cv2.INTER_AREA)
        return image_warped

    def transform_frame(self, use_mask=False):
        frame = self.top_down_capture.get_latest_frame_rgb()
        self.frame_cnt += 1
        if self.should_flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return self.transform_table(frame)

    def construct_map(self, frame=None, use_bg_subtraction=False):
        """
        This constructs:
            1. A "full map" constructed either by background subtraction or object
                bounding boxes.
            2. A dictionary of "obstacle maps" that only show the presence of a single
                object. (These can be stacked togther later, in case the agent is only
                interested in knowing about certain object obstacles.)
        """

        if use_bg_subtraction:
            back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=125)
            back_sub.apply(self.background)

            # Create rough mask (obstacle pixels are 255)
            obstacles_raw = back_sub.apply(frame)

            # Remove noise (erode black areas and dilate white areas)
            kernel_denoise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            obstacles_clean = cv2.morphologyEx(obstacles_raw, cv2.MORPH_OPEN, kernel_denoise, iterations=3)
            obstacles_clean = cv2.threshold(obstacles_clean, 0, 255, cv2.THRESH_BINARY)[1]

            # Store map
            self.full_map = cv2.bitwise_or(self.empty_table_map, obstacles_clean)

            for ob in self.known_objects:
                object_mask = np.zeros_like(self.full_map)
                cv2.fillPoly(object_mask, [np.array(ob.bbox, dtype=np.int32)], 255)
                masked_map = cv2.bitwise_and(self.full_map, object_mask)
                self.obstacle_maps[ob.name] = masked_map
        else:
             self.full_map = self.empty_table_map.copy()
             for ob in self.known_objects:
                object_mask = np.zeros_like(self.full_map)
                cv2.fillPoly(object_mask, [np.array(ob.bbox, dtype=np.int32)], 255)
                self.full_map = cv2.bitwise_or(self.full_map, object_mask)
                self.obstacle_maps[ob.name] = object_mask

    def track_known_objects(self, frame):

        results = self.yolo_obb.track(frame, persist=True, verbose=False)[0]
        if len(results.obb) == 0:
            # No objects this frame: clear detected sets
            self.detected_robotic_objects.clear()
            self.detected_pushable_objects.clear()
            self.detected_landmark_objects.clear()
            return

        boxes = results.obb.xyxyxyxy.cpu().numpy()
        xywhr = results.obb.xywhr.cpu().numpy()
        classes = results.obb.cls.cpu()
        confs = results.obb.conf.cpu()
        ids = results.obb.id.int().cpu().tolist() if results.obb.id is not None else []

        # Prepare per-frame seen sets for pruning
        seen_robotic = set()
        seen_pushable = set()
        seen_landmark = set()

        for i in range(len(ids)):
            class_id = int(classes[i])
            confidence = confs[i]
            # if class_id > 9:
            #     continue
            if confidence < 0.8:
                continue

            # find object that matches the ID
            detected_object = None
            # self.known_objects stores all objects that are on the table
            for ob in self.known_objects:
                if ob.yolo_id == ids[i]:
     
                    detected_object = ob
                    break

            # If no object matches the ID then we search our list.
            # If the name matches, then we assign an id to that object
            if detected_object is None:
                name = self.yolo_class_mapping.get(class_id, f"unknown_{class_id}")
                for ob in self.known_objects:
                    if ob.name == name:
                        detected_object = ob
                        detected_object.yolo_id = ids[i]
                        break

            if detected_object is not None:
                # print("detected object: ", detected_object.name)
                # print("detected object robotic: ", detected_object.is_active)

                bbox = boxes[i].copy()
                detected_object.angle_rad = xywhr[i][4]
                detected_object.bbox = bbox
                detected_object.confidence = confidence  # Store confidence for visualization

                ax_a = bbox[1] - bbox[0]
                ax_b = bbox[2] - bbox[1]
                major_axis = ax_a if np.linalg.norm(ax_a) > np.linalg.norm(ax_b) else ax_b
                major_axis /= np.linalg.norm(major_axis)
                if detected_object.should_flip_major_axis:
                    major_axis *= -1
                angle = np.arctan2(major_axis[1], major_axis[0])

                if detected_object.major_axis is None:
                    detected_object.major_axis = major_axis
                else:
                    major_axis_prev = detected_object.major_axis
                    angle_prev = np.arctan2(major_axis_prev[1], major_axis_prev[0])

                    angle_diff = angle - angle_prev
                    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]

                    if np.abs(angle_diff) > 1.5:
                        detected_object.should_flip_major_axis ^= True
                        major_axis *= -1

                    new_major_axis = detected_object.major_axis * 0.6 + major_axis * 0.4
                    new_major_axis /= np.linalg.norm(new_major_axis)
                    detected_object.major_axis = new_major_axis

                detected_object.bbox = boxes[i].copy()

                # Maintain detected lists
                if detected_object.is_active:
                    # Robotic objects set
                    self.detected_robotic_objects.add(detected_object)
                    seen_robotic.add(detected_object)
                else:
                    # Non-robotic: decide between pushable vs landmark
                    if detected_object in self.known_objects:
                        self.detected_pushable_objects.add(detected_object)
                        seen_pushable.add(detected_object)
                    else:
                        self.detected_landmark_objects.add(detected_object)
                        seen_landmark.add(detected_object)

        # Prune entries not seen in this frame via set intersection
        if seen_robotic or seen_pushable or seen_landmark:
            self.detected_robotic_objects &= seen_robotic
            self.detected_pushable_objects &= seen_pushable
            self.detected_landmark_objects &= seen_landmark
        else:
            # If we saw nothing above thresholds, clear sets
            self.detected_robotic_objects.clear()
            self.detected_pushable_objects.clear()
            self.detected_landmark_objects.clear()

        return classes

    def track_single_person(self, frame) -> None:
        """Track a single person (highest confidence) in the frame using YOLOv8n"""

        try:
            results = self.yolo8n.track(frame.copy(), persist=True, conf=0.3, imgsz=640, classes=[0], verbose=False)  # class 0 is person
            highest_confidence = 0
            best_box = None
            best_track_id = None

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
                # print("person results: ", boxes)
                for i, (box, confidence) in enumerate(zip(boxes, confidences)):
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_box = box
                        best_track_id = int(track_ids[i]) if track_ids is not None else None
            # Only update user object if we have a new detection
            if best_box is not None:
                if self.user is None:
                    self.user = PhysicalObject(
                        name="user",
                        uuid=None,
                    )
                # Update user properties
                x1, y1, x2, y2 = best_box
                self.user.bbox = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ])
                self.user.confidence = highest_confidence
                self.user.yolo_id = best_track_id
                self.user.label = "person"

                logger.debug(f"Detected single person with confidence {highest_confidence:.2f}")
            else:
                # No person detected but keep previous user state
                logger.debug("No new person detection")

        except Exception as e:
            logger.error(f"Error in single person detection: {e}")

    def construct_obstacle_map(self, r_expand=30, use_known_objects=True, omit=None):

        # Option 1: When mapping obstacles, only consider objects that are known to the system
        if use_known_objects:
            map = self.empty_table_map.copy()
            omit_list = omit or []
            for ob in self.known_objects:
                if ob in omit_list:
                    continue
                map = cv2.bitwise_or(map, self.obstacle_maps[ob.name])

        # Option 2: When mapping objects, use everything detected by the background subtractor
        else:
            map = self.full_map

        self.custom_preview = map.copy()

        # Expand obstacles (Use object radius)
        kernel_size = (r_expand*2 + 1, r_expand*2 + 1)
        kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        map = cv2.dilate(map, kernel_expand, iterations=1)

        return map

    def get_latest_transformed_frame(self):
        """
        Get the latest transformed frame that was used for object detection.
        This avoids double transformation - the frame is already transformed
        and ready for display with bounding boxes.
        """
        return self.latest_transformed_frame


async def main():
    from camera import init_camera
    top_down_capture, _= init_camera()
    table = TableSpace(top_down_capture, "config/scene_config.yaml")
    tasks = [
        top_down_capture.loop(),
        table.loop(),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())

