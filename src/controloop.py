import asyncio
import cv2
import numpy as np
from tablespace import TableSpace
import time


class ControlLoop:
    table: TableSpace
    tasks: list[asyncio.Task]

    instruction_queue : asyncio.Queue
    current_instruction : dict
    # instructions : set[dict[str,str]]

    def __init__(self, table, instruction_queue):
        self.table = table
        self.instruction_queue = instruction_queue
        self.current_instruction = {}

        self.display_click_coord = None


    # def get_effective_user_position(self) -> np.ndarray | None:
    #     """
    #     Get the effective user position, prioritizing manual position over YOLO detection.
    #     Returns position in table coordinates, or None if no user position is available.
    #     """
    #     # Priority 1: Manual user position from display
    #     if self.display is not None:
    #         manual_pos = self.display.get_manual_user_position()
    #         if manual_pos is not None:
    #             print(f"## Using manual user position: {manual_pos}")
    #             return manual_pos
        
    #     # Priority 2: YOLO-detected user position
    #     if self.table.user is not None and hasattr(self.table.user, 'bbox') and self.table.user.bbox is not None:
    #         # Calculate centroid from YOLO bounding box
    #         return np.mean(self.table.user.bbox, axis=0)
        
    #     return None

    # def get_effective_user_bbox(self) -> np.ndarray | None:
    #     """
    #     Get the effective user bounding box, prioritizing manual position over YOLO detection.
    #     Returns bbox in table coordinates, or None if no user position is available.
    #     """
    #     # Priority 1: Manual user bounding box from display
    #     if self.display is not None:
    #         manual_pos = self.display.get_manual_user_position()
    #         if manual_pos is not None:
    #             return self.display.manual_user_bbox
        
    #     # Priority 2: YOLO-detected user bounding box
    #     if self.table.user is not None and hasattr(self.table.user, 'bbox') and self.table.user.bbox is not None:
    #         return self.table.user.bbox
        
    #     return None

    async def loop(self):
        """
        Main async loop, aligned with other modules (e.g., TableSpace, Agent).
        Spawns internal control tasks and keeps them running.
        """
        self.tasks = [
            asyncio.create_task(self.fetch_latest_instruction()),
            asyncio.create_task(self.plan_and_set_waypoints()),
            asyncio.create_task(self.connect_to_active_objects()),
            asyncio.create_task(self.control_active_objects())
        ]

        await asyncio.gather(*self.tasks)
    
    async def fetch_latest_instruction(self, peroid=0.2):
        while True:
            await asyncio.sleep(peroid)
            instruction = await self.instruction_queue.get()
            print(f'Received instruction: {instruction}')

            # If there are still instructions in the queue, clear the queue
            while self.instruction_queue.qsize() > 0:
                try: self.instruction_queue.get_nowait()
                except: pass
            
            # TODO: Validate instruction
            self.current_instruction = instruction

    async def plan_and_set_waypoints(self, period=.2):
        
        while True:

            # Attempt to re-run the planner every [period] seconds (default 5 Hz)
            await asyncio.sleep(period)

            active_object_name = self.current_instruction.get('active_object')
            target_name = self.current_instruction.get('target')
            action = self.current_instruction.get('action')

            if active_object_name is None: continue

            matches = [ob for ob in self.table.known_objects if ob.name == active_object_name]
            if not matches: continue
            active_object = matches[0]

            if target_name == 'user':

                if self.table.user is None: continue
                
                if self.display_click_coord is not None:
                    target_coordinate = self.display_click_coord

                else:

                    user_left_x = self.table.user.bbox[0][0]
                    user_right_x = self.table.user.bbox[1][0]
                    table_edge = self.table.bottom_y - 40

                    user_left = np.array([user_left_x, table_edge])
                    user_right = np.array([user_right_x, table_edge])

                    dist_to_left = np.linalg.norm(user_left - active_object.centroid)
                    dist_to_right = np.linalg.norm(user_right - active_object.centroid)

                    target_coordinate = user_left if dist_to_left < dist_to_right else user_right

            elif 'coordinate' in target_name:
                print(target_name)
                coords_str = target_name.split('_')
                target_coordinate = [int(coords_str[1]), int(coords_str[2])]
                print(target_coordinate)

            else:
                # Find target object by name, skip if not found
                matching_targets = [ob for ob in self.table.known_objects if ob.name == target_name]
                if not matching_targets:
                    print(f"Warning: Target object '{target_name}' not found. Skipping this goal.")
                    continue
                
                target_object = matching_targets[0]
                target_coordinate = target_object.centroid

            if action == 'move_towards':
                active_object.behavior_index = 0
                active_object.behaviors[0] = 'move_to_waypoint'
                active_object.plan_path_towards(self.table, target_coordinate)

            elif action == 'move_away':
                active_object.behavior_index = 0
                active_object.behaviors[0] = 'move_to_waypoint'
                active_object.plan_path_away(self.table, target_coordinate)

            # Clear the current instruction
            self.current_instruction = {}




            


            # for ob_name, action_info in list(self.table.object_actions.items()):
            #     matches = [ob for ob in self.table.known_objects if ob.name == ob_name]
            #     if not matches:
            #         print(f"Warning: action refers to unknown object '{ob_name}'. Skipping.")
            #         # Optionally drop stale action to avoid repeated warnings
            #         # del self.table.object_actions[ob_name]
            #         continue
            #     ob = matches[0]
            #     action = action_info.get('action', None)

            #     if action == 'move_towards':
            #         coord = action_info.get('coordinate', None)
            #         if coord is not None:
            #             ob.behaviors[0] = 'move_to_waypoint'
            #             print("control loop calling plan_path_towards")
            #             ob.plan_path_towards(self.table, coord)

            #     elif action == 'move_away':
            #         coord = action_info.get('coordinate', None)
            #         if coord is not None:
            #             ob.behaviors[0] = 'move_to_waypoint'
            #             ob.plan_path_away(self.table, coord)

            #     else:
            #         if len(ob.waypoints) > 0:
            #             print(f"[ControlLoop] Clearing waypoints for {ob.name} due to unmatched action '{action}'")
            #         ob.clear_waypoints()


    async def connect_to_active_objects(self, period=2):
        """Periodically try to connect to active objects in the scene."""
        while True:
            await asyncio.sleep(period)

            for ob in self.table.known_objects:
                if ob.ble is None: continue

                # Connect to object if not currently connected
                if not ob.ble.is_connected:
                    await ob.ble.connect()


    # Send motor commands to objects
    async def control_active_objects(self):
        """"Send motor commands to active objects."""

        last_time = time.time()

        while True:

            # Yield to other coroutines
            await asyncio.sleep(0)

            for ob in self.table.known_objects:
                if ob.ble is None: continue
                if not ob.ble.is_connected: continue

                if ob.orientation_is_uncertain:

                    await ob.calibrate_orientation()
                else:
                    await ob.update_motors()

    

    # async def set_goals_from_file(self, filepath):
        

    #     while True:

    #         await asyncio.sleep(5)
            
    #         print('reading from file...')
    #         # self.if_act = True
    #         if self.if_act:
                
    #             self.if_act = False

    #             # Reset behavior counter
    #             for ob in self.table.known_objects:
    #                 ob.behavior_index = 0

    #             print("Setting goals from file")

    #             with open(filepath, 'r') as yaml_file:
    #                 goals = yaml.safe_load(yaml_file)

    #             if goals is None: continue

    #             for ob_name in goals:
    #                 action = goals[ob_name].get('action', None)
    #                 target_name = goals[ob_name].get('target', None)
    #                 push_name = goals[ob_name].get('object_to_push', None)

    #                 if action is None or target_name is None: continue


    #                 agent = [ob for ob in self.table.known_objects if ob.name == ob_name][0]


    #                 if action == 'move_towards':
    #                     if target_name == 'mouse':
    #                         if self.click_coord is None: continue
    #                         click_target = np.array(self.click_coord)
    #                         target_coord = click_target

    #                     elif target_name == 'user':
    #                         user_bbox = self.get_effective_user_bbox()
    #                         if user_bbox is not None:

    #                             # project user onto table (two points near left and right side of user)
    #                             target_left = np.array([
    #                                 int(user_bbox[3][0]+55),
    #                                 int(user_bbox[3][1]+30)
    #                             ])

    #                             target_right = np.array([
    #                                 int(user_bbox[2][0]-55),
    #                                 int(user_bbox[2][1]+30)
    #                             ])
                                
    #                             # choose closest target
    #                             dist_to_right = np.linalg.norm(target_right - agent.centroid)
    #                             dist_to_left = np.linalg.norm(target_left - agent.centroid)
    #                             if dist_to_right < dist_to_left:
    #                                 target_coord = target_right
    #                             else:
    #                                 target_coord = target_left
    #                         else:
    #                             continue
    #                     else:
    #                         # Find target object by name, skip if not found
    #                         matching_targets = [ob for ob in self.table.known_objects if ob.name == target_name]
    #                         if not matching_targets:
    #                             print(f"Warning: Target object '{target_name}' not found. Skipping this goal.")
    #                             continue
                            
    #                         target_ob = matching_targets[0]
    #                         target_coord = target_ob.centroid
                       
    #                     action_info = {
    #                         'action': 'move_towards',
    #                         'coordinate': target_coord
    #                     }
    #                     self.table.object_actions[ob_name] = action_info

    #                 elif action == 'move_towards_and_point_away':
    #                     if target_name == 'mouse':
    #                         target_coord = self.click_coord
    #                     elif target_name == 'user':
    #                         user_pos = self.get_effective_user_position()
    #                         user_bbox = self.get_effective_user_bbox()
    #                         if user_pos is not None and user_bbox is not None:
    #                             target_coord = np.array([
    #                                 int(user_pos[0]),
    #                                 int(user_bbox[2][1] + 5)
    #                             ])

    #                     else:
    #                         # Find target object by name, skip if not found
    #                         matching_targets = [ob for ob in self.table.known_objects if ob.name == target_name]
    #                         if not matching_targets:
    #                             print(f"Warning: Target object '{target_name}' not found. Skipping this goal.")
    #                             continue
                            
    #                         target_ob = matching_targets[0]
    #                         target_coord = target_ob.centroid
                        
    #                     action_info = {
    #                         'action': 'move_towards_and_point_away',
    #                         'coordinate': target_coord
    #                     }
    #                     self.table.object_actions[ob_name] = action_info


    #                 elif action == 'move_away':

    #                     if target_name == 'mouse':
    #                         target_coord = self.click_coord

    #                         action_info = {
    #                             'action': 'move_away',
    #                             'coordinate': target_coord
    #                         }
    #                         self.table.object_actions[ob_name] = action_info

    #                     else:
    #                         target_coord = np.array([agent.centroid[0], agent.centroid[1]+100])
    #                         action_info = {
    #                             'action': 'move_towards',
    #                             'coordinate': target_coord
    #                         }
    #                         self.table.object_actions[ob_name] = action_info

    #                 elif action == 'point_towards':
    #                     if target_name == 'user':
    #                         user_pos = self.get_effective_user_position()
    #                         if user_pos is not None:
    #                             target_coord = user_pos
    #                         else:
    #                             continue
    #                     elif target_name == 'mouse':
    #                         target_coord = np.array(self.click_coord)
    #                     else:
    #                         # Find target object by name, skip if not found
    #                         matching_targets = [ob for ob in self.table.known_objects if ob.name == target_name]
    #                         if not matching_targets:
    #                             print(f"Warning: Target object '{target_name}' not found. Skipping this goal.")
    #                             continue
                            
    #                         target_ob = matching_targets[0]
    #                         target_coord = target_ob.centroid
    #                     action_info = {
    #                         'action': 'point_towards',
    #                         'coordinate': target_coord
    #                     }
    #                     self.table.object_actions[ob_name] = action_info

    #                 elif action == 'point_away':
    #                     if target_name == 'user':
    #                         user_pos = self.get_effective_user_position()
    #                         if user_pos is not None:
    #                             target_coord = user_pos
    #                         else:
    #                             continue
    #                     else:
    #                         # Find target object by name, skip if not found
    #                         matching_targets = [ob for ob in self.table.known_objects if ob.name == target_name]
    #                         if not matching_targets:
    #                             print(f"Warning: Target object '{target_name}' not found. Skipping this goal.")
    #                             continue
                            
    #                         target_ob = matching_targets[0]
    #                         target_coord = target_ob.centroid
    #                     action_info = {
    #                         'action': 'point_away',
    #                         'coordinate': target_coord
    #                     }
    #                     self.table.object_actions[ob_name] = action_info

    #                 elif action == 'push_towards':
    #                     if push_name is None: continue

    #                     if push_name == 'white_usb_adapter':
    #                         push_name = 'white_adapter'

    #                     push_ob = [ob for ob in self.table.known_objects if ob.name == push_name][0]

    #                     if target_name == 'mouse':
    #                         target_coord = np.array([340, 140])
    #                         # target_coord = np.array(self.click_coord)

    #                     elif target_name == 'user':
    #                         target_coord = np.array([340, 140])
                        
    #                     # else:
    #                     #     target_ob = [ob for ob in self.table.known_objects if ob.name == target_name][0]
    #                     #     target_coord = target_ob.centroid
                        
    #                     action_info = {
    #                         'action': 'push_towards',
    #                         'coordinate': target_coord,
    #                         'to_push' : push_ob
    #                     }

    #                     self.table.object_actions[ob_name] = action_info
    #                     print(action_info)



async def send_instructions(instruction_queue : asyncio.Queue):
    print('Sending instruction!')
    await instruction_queue.put(
        {
            'active_object' : 'stapler',
            'target' : 'user',
            'action' : 'move_towards'
        }
    )
    asyncio.sleep(5)

async def run_standalone():
    from camera import init_camera
    from display import CameraDisplay

    top_down_capture, env_capture = init_camera(0, 0)
    object_detection = TableSpace(top_down_capture, "config/scene_config.yaml")

    instruction_queue = asyncio.LifoQueue()

    control_loop = ControlLoop(object_detection, instruction_queue)

    display = CameraDisplay(top_down_capture, env_capture, object_detection, None, enable_dev_window=True)  # Agent will be set later


    tasks = [
        top_down_capture.loop(),
        object_detection.loop(),
        control_loop.loop(),
        display.loop(),
        send_instructions(instruction_queue)
    ]

    await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(run_standalone())