import asyncio
import cv2
import yaml
import numpy as np
from tablespace import TableSpace
import time


class ControlLoop:
    table: TableSpace
    tasks: list[asyncio.Task]

    def __init__(self, table, preview=False,
                 goal_filepath="", reason_and_injection_only=False,
                 record_video=False, session_dir=None,
                 alignment_threshold: float = 0.5, display=None):
        self.table = table
        self.preview = preview
        self.goal_filepath = goal_filepath
        self.click_coord = None
        self.display = display  # Reference to CameraDisplay for manual user position

        # Initialize position tracking for user
        self.detected_user_position = None
        self.draw_bounding_boxes = True
        self.gemini_objects = []

        # Control detection frequency
        self.last_person_detection_time = 0
        self.person_detection_interval = 1.5
        self.reason_and_injection_only = reason_and_injection_only
        self.enable_person_detection = False
        self.if_act = False
        self.sky_touched = False
        # Video recording settings
        self.record_video = record_video
        self.video_writer = None
        self.video_filename = None
        self.video_fps = 30.0
        self.video_frame_size = None
        self.session_dir = session_dir

        # Internal background tasks set (for symmetry with other modules)
        self.background_tasks: set[asyncio.Task] = set()

        # Alignment gating: default deny until first alignment update
        self._alignment_score: float | None = None
        self.alignment_threshold: float = float(alignment_threshold)
        self._pending_action_plan: dict | None = None

    def set_alignment_score(self, score: float) -> None:
        """
        Update latest alignment score from the agent loop.
        """
        self._alignment_score = score

    def get_effective_user_position(self) -> np.ndarray | None:
        """
        Get the effective user position, prioritizing manual position over YOLO detection.
        Returns position in table coordinates, or None if no user position is available.
        """
        # Priority 1: Manual user position from display
        if self.display is not None:
            manual_pos = self.display.get_manual_user_position()
            if manual_pos is not None:
                print(f"## Using manual user position: {manual_pos}")
                return manual_pos
        
        # Priority 2: YOLO-detected user position
        if self.table.user is not None and hasattr(self.table.user, 'bbox') and self.table.user.bbox is not None:
            # Calculate centroid from YOLO bounding box
            return np.mean(self.table.user.bbox, axis=0)
        
        return None

    def get_effective_user_bbox(self) -> np.ndarray | None:
        """
        Get the effective user bounding box, prioritizing manual position over YOLO detection.
        Returns bbox in table coordinates, or None if no user position is available.
        """
        # Priority 1: Manual user bounding box from display
        if self.display is not None:
            manual_pos = self.display.get_manual_user_position()
            if manual_pos is not None:
                return self.display.manual_user_bbox
        
        # Priority 2: YOLO-detected user bounding box
        if self.table.user is not None and hasattr(self.table.user, 'bbox') and self.table.user.bbox is not None:
            return self.table.user.bbox
        
        return None

    # may need more check
    def submit_action_plan(self, plan: dict) -> None:
        """
        Receive the latest action plan from the agent loop.
        Stored until alignment passes and planning consumes it.
        """
        self._pending_action_plan = plan

    # may need more check
    def _consume_pending_plan_into_actions(self) -> None:
        """
        Translate a pending action plan into concrete table.object_actions once
        alignment gating allows. Clears the pending plan after consumption.
        """
        if self._pending_action_plan is None:
            return

        plan = self._pending_action_plan
        object_to_move = plan.get('object_to_move')
        robotic_pusher = plan.get('robotic_object_to_push')
        movement_destination = plan.get('movement_destination')
        action = plan.get('action')

        if object_to_move is None or movement_destination is None or action is None:
            self._pending_action_plan = None
            return

        # Determine actor and push target if applicable
        is_push = isinstance(action, str) and action.startswith('push_')
        actor_name = None
        to_push_ob = None
        if is_push:
            # Actor is the robotic pusher; the thing being pushed is object_to_move
            if robotic_pusher is None or str(robotic_pusher).lower() == 'none':
                # Cannot execute a push without a pusher
                self._pending_action_plan = None
                return
            actor_name = robotic_pusher
            push_matches = [ob for ob in self.table.known_objects if ob.name == object_to_move]
            if not push_matches:
                self._pending_action_plan = None
                return
            to_push_ob = push_matches[0]
        else:
            actor_name = object_to_move

        print(f'From Agent Loop: actor_name: {actor_name}, movement_destination: {movement_destination}, action: {action}')

        # Resolve destination coordinate
        # TODO: check with old repo
        target_coord = None
        if isinstance(movement_destination, str) and movement_destination.lower() == 'user':
            user_bbox = self.get_effective_user_bbox()
            if user_bbox is not None:
                target_left = np.array([
                    int(user_bbox[3][0]+55),
                    int(user_bbox[3][1]+30)
                ])
                target_right = np.array([
                    int(user_bbox[2][0]-55),
                    int(user_bbox[2][1]+30)
                ])
                agent_matches = [ob for ob in self.table.known_objects if ob.name == actor_name]
                if agent_matches:
                    agent_ob = agent_matches[0]
                    dist_to_right = np.linalg.norm(target_right - agent_ob.centroid)
                    dist_to_left = np.linalg.norm(target_left - agent_ob.centroid)
                    target_coord = target_right if dist_to_right < dist_to_left else target_left
                else:
                    target_coord = target_left
        else:
            # treat as object name
            dest_matches = [ob for ob in self.table.known_objects if ob.name == movement_destination]
            if dest_matches:
                target_coord = dest_matches[0].centroid

        if target_coord is None:
            # Nothing actionable
            self._pending_action_plan = None
            return

        # Normalize/translate actions to those implemented in plan_and_set_waypoints
        normalized_action = action
        coord = target_coord

        # Translate push_away into a push_towards towards a point away from the destination
        if action == 'push_away' and to_push_ob is not None:
            away_vec = to_push_ob.centroid - target_coord
            norm = np.linalg.norm(away_vec)
            if norm > 1e-6:
                away_dir = away_vec / norm
                coord = to_push_ob.centroid + away_dir * 120.0
            normalized_action = 'push_towards'

        action_info = {'action': normalized_action, 'coordinate': coord}
        if normalized_action == 'push_towards' and to_push_ob is not None:
            action_info['to_push'] = to_push_ob

        self.table.object_actions[actor_name] = action_info
        self._pending_action_plan = None

    async def loop(self):
        """
        Main async loop, aligned with other modules (e.g., TableSpace, Agent).
        Spawns internal control tasks and keeps them running.
        """
        self.tasks = [
            asyncio.create_task(self.plan_and_set_waypoints()),
            asyncio.create_task(self.connect_to_active_objects()),
            asyncio.create_task(self.control_active_objects())
        ]
        # old
        # if self.goal_filepath:
        #     self.tasks.append(asyncio.create_task(self.set_goals_from_file(self.goal_filepath)))

        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            # Propagate cancellation to child tasks
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
        finally:
            # Clean up resources
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"Control loop video saved to: {self.video_filename}")
                self.video_writer = None

    def run_standalone(self):
        """
        Starts the control loop, running independently from the rest of the system. (Used
        for debuggging, or could run in separate process if needed.)
        """
        asyncio.run(self.run())


    def handle_click_event(self, event, x, y, flags, param):
        # print('getting mouse event')
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_coord = [x, y]
            print(f'Clicked at {self.click_coord}')

            for ob in self.table.known_objects:
                ob.behavior_index = 0

            self.if_act = True


    async def plan_and_set_waypoints(self, period=0.2):
        
        # Initialize frame counter
        frame_cnt = -1

        while True:

            # Attempt to re-run the planner every [period] seconds (default 5 Hz)
            await asyncio.sleep(period)

            # Gate planning based on alignment score
            if self._alignment_score is None or self._alignment_score < self.alignment_threshold:
                continue

            # Consume any pending plan into object_actions once alignment passes
            self._consume_pending_plan_into_actions()


            # Avoid path-planning again if observation has not changed
            if self.table.frame_cnt == frame_cnt:
                continue

            # Keep track of frame number so we don't unnecessarily re-plan
            frame_cnt = self.table.frame_cnt

            for ob_name, action_info in list(self.table.object_actions.items()):
                matches = [ob for ob in self.table.known_objects if ob.name == ob_name]
                if not matches:
                    print(f"Warning: action refers to unknown object '{ob_name}'. Skipping.")
                    # Optionally drop stale action to avoid repeated warnings
                    # del self.table.object_actions[ob_name]
                    continue
                ob = matches[0]
                action = action_info.get('action', None)

                if action == 'move_towards':
                    coord = action_info.get('coordinate', None)
                    if coord is not None:
                        ob.behaviors[0] = 'move_to_waypoint'
                        print("control loop calling plan_path_towards")
                        ob.plan_path_towards(self.table, coord)

                elif action == 'move_away':
                    coord = action_info.get('coordinate', None)
                    if coord is not None:
                        ob.behaviors[0] = 'move_to_waypoint'
                        ob.plan_path_away(self.table, coord)

                elif action == 'point_towards':
                    coord = action_info.get('coordinate', None)
                    if coord is not None:
                        # get vector pointing to coord
                        object_to_coord = coord - ob.centroid
                        ob.behaviors[0] = 'rotate_to_angle'
                        ob.target_heading = object_to_coord / np.linalg.norm(object_to_coord)

                elif action == 'move_towards_and_point_away':
                    coord = action_info.get('coordinate', None)
                    if coord is not None:
                        # get vector pointing away from coord
                        coord_to_object = ob.centroid - coord
                        ob.behaviors[0] = 'move_to_waypoint'
                        ob.behaviors[1] = 'rotate_to_angle'
                        ob.plan_path_towards(self.table, coord)
                        ob.target_heading = coord_to_object / np.linalg.norm(coord_to_object)

                elif action == 'point_away':
                    coord = action_info.get('coordinate', None)
                    if coord is not None:
                        # get vector pointing away from coord
                        coord_to_object = ob.centroid - coord
                        ob.behaviors[0] = 'rotate_to_angle'
                        ob.target_heading = coord_to_object / np.linalg.norm(coord_to_object)

                elif action == 'push_towards':

                    if ob.behavior_index < 2:
                        if ob.behaviors[0] != '': continue

                        coord = action_info.get('coordinate', None)
                        ob_to_push = action_info.get('to_push', None)

                        # first, plan path from pushed object to coordinate
                        ob_to_push.plan_path_towards(self.table, coord, omit=[ob])

                        # did we find a waypoint? if not, exit
                        if len(ob_to_push.waypoints) < 1: continue


                        to_waypoint = np.array(ob_to_push.waypoints[0]) - ob_to_push.centroid
                        start_coord = \
                            (to_waypoint / np.linalg.norm(to_waypoint) * -20) + ob_to_push.centroid
                        # start_coord = start_coord + np.array([0,-40])
                        # start_coord = \
                        #     (to_waypoint / np.linalg.norm(to_waypoint) * -70) + ob_to_push.centroid
                        # start_coord = start_coord + np.array([0,0])

                        # plan a path to the starting point
                        ob.plan_path_towards(self.table, start_coord, omit=[ob_to_push])
                        ob.behaviors[0] = 'move_to_waypoint'


                        # rotate to face the first waypoint
                        ob.target_heading = to_waypoint / np.linalg.norm(to_waypoint)
                        ob.behaviors[1] = 'rotate_to_angle'
                        # ob.target_heading = coord_to_object

                    else:
                        ob.plan_path_towards(self.table, coord, omit=[ob_to_push])
                        ob.behaviors[2] = 'move_to_waypoint'

                else:
                    if len(ob.waypoints) > 0:
                        print(f"[ControlLoop] Clearing waypoints for {ob.name} due to unmatched action '{action}'")
                    ob.clear_waypoints()


    async def connect_to_active_objects(self, period=2):
        """
        Periodically try to connect to active objects in the scene.
        """
        while True:
            await asyncio.sleep(period)

            for ob in self.table.known_objects:
                if ob.ble is None: continue

                # Connect to object if not currently connected
                if not ob.ble.is_connected:
                    await ob.ble.connect()


    # Send motor commands to objects
    async def control_active_objects(self):
        """"
        Send motor commands to active objects.
        """


        last_time = time.time()

        while True:

            # Yield to other coroutines
            await asyncio.sleep(0)

            # current_time = time.time()
            # print(f"Elapsed time: {current_time - last_time}")
            # last_time = current_time

            for ob in self.table.known_objects:
                if ob.ble is None: continue
                if not ob.ble.is_connected: continue

                if ob.orientation_is_uncertain:

                    await ob.calibrate_orientation()
                else:
                    await ob.update_motors()

    
    def touch_the_sky(self, goal_data):
        print(f"Looking extra fly! Here are the goals: {goal_data}")

        for ob_name in goal_data:
            push_name = goal_data[ob_name].get('object_to_push', None)
            push_ob = [ob for ob in self.table.known_objects if ob.name == push_name][0]

            target_coord = np.array([389, 138])
                        
            action_info = {
                'action': 'push_towards',
                'coordinate': target_coord,
                'to_push' : push_ob
            }

            self.table.object_actions[ob_name] = action_info

            print(f"Setting goal for {ob_name}: {action_info}")



    async def set_goals_from_file(self, filepath):
        

        while True:

            await asyncio.sleep(5)

            if (self.sky_touched):
                continue
            
            self.sky_touched = True

            goals_data = {
                'pencil_box': {
                    'action': 'push_towards',
                    'target': 'user',
                    'object_to_push': 'white_adapter'
                }
            }

            print('touching the sky...')
            self.touch_the_sky(goals_data)

            continue
            
            print('reading from file...')
            # self.if_act = True
            if self.if_act:
                
                self.if_act = False

                # Reset behavior counter
                for ob in self.table.known_objects:
                    ob.behavior_index = 0

                print("Setting goals from file")

                with open(filepath, 'r') as yaml_file:
                    goals = yaml.safe_load(yaml_file)

                if goals is None: continue

                for ob_name in goals:
                    action = goals[ob_name].get('action', None)
                    target_name = goals[ob_name].get('target', None)
                    push_name = goals[ob_name].get('object_to_push', None)

                    if action is None or target_name is None: continue


                    agent = [ob for ob in self.table.known_objects if ob.name == ob_name][0]


                    if action == 'move_towards':
                        if target_name == 'mouse':
                            if self.click_coord is None: continue
                            click_target = np.array(self.click_coord)
                            target_coord = click_target

                        elif target_name == 'user':
                            user_bbox = self.get_effective_user_bbox()
                            if user_bbox is not None:

                                # project user onto table (two points near left and right side of user)
                                target_left = np.array([
                                    int(user_bbox[3][0]+55),
                                    int(user_bbox[3][1]+30)
                                ])

                                target_right = np.array([
                                    int(user_bbox[2][0]-55),
                                    int(user_bbox[2][1]+30)
                                ])
                                
                                # choose closest target
                                dist_to_right = np.linalg.norm(target_right - agent.centroid)
                                dist_to_left = np.linalg.norm(target_left - agent.centroid)
                                if dist_to_right < dist_to_left:
                                    target_coord = target_right
                                else:
                                    target_coord = target_left
                            else:
                                continue
                        else:
                            # Find target object by name, skip if not found
                            matching_targets = [ob for ob in self.table.known_objects if ob.name == target_name]
                            if not matching_targets:
                                print(f"Warning: Target object '{target_name}' not found. Skipping this goal.")
                                continue
                            
                            target_ob = matching_targets[0]
                            target_coord = target_ob.centroid
                       
                        action_info = {
                            'action': 'move_towards',
                            'coordinate': target_coord
                        }
                        self.table.object_actions[ob_name] = action_info

                    elif action == 'move_towards_and_point_away':
                        if target_name == 'mouse':
                            target_coord = self.click_coord
                        elif target_name == 'user':
                            user_pos = self.get_effective_user_position()
                            user_bbox = self.get_effective_user_bbox()
                            if user_pos is not None and user_bbox is not None:
                                target_coord = np.array([
                                    int(user_pos[0]),
                                    int(user_bbox[2][1] + 5)
                                ])

                        else:
                            # Find target object by name, skip if not found
                            matching_targets = [ob for ob in self.table.known_objects if ob.name == target_name]
                            if not matching_targets:
                                print(f"Warning: Target object '{target_name}' not found. Skipping this goal.")
                                continue
                            
                            target_ob = matching_targets[0]
                            target_coord = target_ob.centroid
                        
                        action_info = {
                            'action': 'move_towards_and_point_away',
                            'coordinate': target_coord
                        }
                        self.table.object_actions[ob_name] = action_info


                    elif action == 'move_away':

                        if target_name == 'mouse':
                            target_coord = self.click_coord

                            action_info = {
                                'action': 'move_away',
                                'coordinate': target_coord
                            }
                            self.table.object_actions[ob_name] = action_info

                        else:
                            target_coord = np.array([agent.centroid[0], agent.centroid[1]+100])
                            action_info = {
                                'action': 'move_towards',
                                'coordinate': target_coord
                            }
                            self.table.object_actions[ob_name] = action_info

                    elif action == 'point_towards':
                        if target_name == 'user':
                            user_pos = self.get_effective_user_position()
                            if user_pos is not None:
                                target_coord = user_pos
                            else:
                                continue
                        elif target_name == 'mouse':
                            target_coord = np.array(self.click_coord)
                        else:
                            # Find target object by name, skip if not found
                            matching_targets = [ob for ob in self.table.known_objects if ob.name == target_name]
                            if not matching_targets:
                                print(f"Warning: Target object '{target_name}' not found. Skipping this goal.")
                                continue
                            
                            target_ob = matching_targets[0]
                            target_coord = target_ob.centroid
                        action_info = {
                            'action': 'point_towards',
                            'coordinate': target_coord
                        }
                        self.table.object_actions[ob_name] = action_info

                    elif action == 'point_away':
                        if target_name == 'user':
                            user_pos = self.get_effective_user_position()
                            if user_pos is not None:
                                target_coord = user_pos
                            else:
                                continue
                        else:
                            # Find target object by name, skip if not found
                            matching_targets = [ob for ob in self.table.known_objects if ob.name == target_name]
                            if not matching_targets:
                                print(f"Warning: Target object '{target_name}' not found. Skipping this goal.")
                                continue
                            
                            target_ob = matching_targets[0]
                            target_coord = target_ob.centroid
                        action_info = {
                            'action': 'point_away',
                            'coordinate': target_coord
                        }
                        self.table.object_actions[ob_name] = action_info

                    elif action == 'push_towards':
                        if push_name is None: continue

                        if push_name == 'white_usb_adapter':
                            push_name = 'white_adapter'

                        push_ob = [ob for ob in self.table.known_objects if ob.name == push_name][0]

                        if target_name == 'mouse':
                            target_coord = np.array([340, 140])
                            # target_coord = np.array(self.click_coord)

                        elif target_name == 'user':
                            target_coord = np.array([340, 140])
                        
                        # else:
                        #     target_ob = [ob for ob in self.table.known_objects if ob.name == target_name][0]
                        #     target_coord = target_ob.centroid
                        
                        action_info = {
                            'action': 'push_towards',
                            'coordinate': target_coord,
                            'to_push' : push_ob
                        }

                        self.table.object_actions[ob_name] = action_info
                        print(action_info)
