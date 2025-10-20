import numpy as np
import time
import cv2
import routefinding
from bleak import BleakClient, BleakScanner
import asyncio

MOTOR_CHARACTERISTIC_UUID = "bc464776-c843-48e9-8320-ac820ecd96b8"

class PhysicalObject:
    name: str

    def __init__(self, name: str, uuid:str = '', device_name:str = '', heading_offset:float = 0):
        """
        Initialize a physical object.
        Args:
            name (str): Label for the object (e.g. "coffee_mug") 
            uuid (str): UUID for connecting to device
        """
        self.name = name                # object label
        self.bbox = np.zeros((4,2))     # oriented bounding box (four corners)
        self.major_axis = None
        self.should_flip_major_axis = False

        self.orientation_is_uncertain = True

        self.last_updated: float = time.time()     # time of last OBB update
        self.yolo_id: int = None
        self.waypoints: list[np.ndarray] = []
        self.target_heading = None
        self.uuid = uuid
        self.device_name = device_name
        self.is_active = False
        if uuid or device_name:
            self.is_active = True
        self.last_centroid = None
        self.heading_error = 0

        self.behaviors = ['', '', '']
        self.behavior_index = 0

        # Calculate unique motion primitives (used for routefinding)
        self.primitives = routefinding.generate_simple_primitives(step_size=10)

        # Only initialize BLE if a UUID is provided (active objects)
        self.ble = DifferentialDriveController(uuid, device_name) if self.is_active else None

        # Rotation matrix that maps the major axis the the "forward heading"
        # theta = -1 * np.pi / 2
        self.axis_to_heading = np.array([
            [np.cos(heading_offset), -np.sin(heading_offset)],
            [np.sin(heading_offset),  np.cos(heading_offset)]
        ])

    @property
    def centroid(self):
        return np.mean(self.bbox, axis=0)

    @property
    def orientation_radians(self, from_corner:int=3, to_corner:int=0):
        obb_vector = self.bbox[to_corner] - self.bbox[from_corner]
        angle = np.arctan2(obb_vector[1], obb_vector[0])
        return angle if angle >= 0 else angle + 2*np.pi

    @property
    def heading(self):
        return (self.axis_to_heading @ self.major_axis.reshape((2, 1))).reshape((2,))
        # return self.axis_to_heading * self.major_axis
        return self.major_axis

    async def move_fwd_back(self):
        await self.ble.send_motor_command([150,0,150,0])
        await asyncio.sleep(0.2)
        await self.ble.send_motor_command([0,0,0,0])
        await asyncio.sleep(1.0)
        await self.ble.send_motor_command([0,150,0,150])
        await asyncio.sleep(0.2)
        await self.ble.send_motor_command([0,0,0,0])



    async def calibrate_orientation(self):

        print(f'Calibrating {self.name}')

        # record inital centroid
        centroid_inital = self.centroid

        # move fwd
        print('Moving Forward...')
        await self.ble.send_motor_command([150, 0, 150, 0])
        await asyncio.sleep(0.05)
        await self.ble.send_motor_command([0, 0, 0, 0])

        # stop and wait, then record new orientation
        await asyncio.sleep(1)
        centroid_after_fwd = self.centroid

        # move back
        print('Moving Backward...')
        await self.ble.send_motor_command([0, 150, 0, 150])
        await asyncio.sleep(0.05)
        await self.ble.send_motor_command([0, 0, 0, 0])

        # stop and wait, then record orientation again
        await asyncio.sleep(1)
        centroid_after_bck = self.centroid

        print(centroid_inital, centroid_after_fwd, centroid_after_bck)

        self.orientation_is_uncertain = False


    async def update_motors(self):

        if self.behavior_index >= len(self.behaviors):
            await self.turn_off_motors()
            return

        match self.behaviors[self.behavior_index]:

            case 'calibrate':
                await self.calibrate_orientation()
    
            case 'rotate_to_angle':
                await self.rotate_to_angle()
    
            case 'move_to_waypoint':
                await self.move_to_waypoint()

            case _:
                await self.turn_off_motors()
    
    async def turn_off_motors(self):
        await self.ble.send_motor_command([0, 0, 0, 0])

    async def rotate_to_angle(self, tolerance=0.4, use_major_axis = False):

        heading = self.major_axis if use_major_axis else self.heading

        # If we do not know the heading of the agent, then there's nothing we can do
        if heading is None or self.target_heading is None:
            return

        # Get heading error (angle from current axb. to target axis)
        error = signed_heading_error(heading, self.target_heading)

        # get error as a percent of heading tolerance
        error_pct = np.abs(error) / tolerance

        # If error is 100% or less of tolerance, then motor power should be zero. If
        # error is error_pct_max (or more) of tolerance, then motor power is motor_max. For
        # all values in between, interpolate.
        motor_min = 40
        motor_max = 45
        error_pct_max = 3.0
        motor_power = int(np.interp(
            error_pct, [1.0, error_pct_max], [motor_min, motor_max]))
        
        if (error < tolerance * -0.8):
            # Agent needs to turn left. Right wheel should turn forward, and left wheel
            # should turn backward.
            await self.ble.send_motor_command([motor_power, 0, 0, motor_power])


        elif (error > tolerance * 0.8):
            # Agent needs to turn right. Right wheel should turn backward, and left wheel
            # should turn forward.
            # await self.agent.drive(right=motor_power, left=motor_power*-1)
            await self.ble.send_motor_command([0, motor_power, motor_power, 0])

        else:
            # We are well within tolerance, stop motors.
            # await self.agent.drive(right=0, left=0)
            await self.ble.send_motor_command([0, 0, 0, 0])

        # If we are within tolerance, set status flag to success
        if np.abs(error) < tolerance:
            await self.ble.send_motor_command([0, 0, 0, 0])
            self.behavior_index += 1
            return
    

    async def move_to_waypoint(self):
        
        # If no waypoints are set, stop motors and return immediately
        if not self.waypoints:
            await self.turn_off_motors()
            return

        heading_error_threshold = 0.5
        distance_error_threshold = 20

        while True:
            # get current waypoint from path
            waypoint = self.waypoints[0]

            position_error = waypoint - self.centroid
            distance_to_waypoint = np.linalg.norm(position_error)

            # Are we already at the waypoint?
            if distance_to_waypoint <= distance_error_threshold:
                if len(self.waypoints) > 1:
                    # We have arrived at a waypoint, but there are still more to go
                    self.waypoints = self.waypoints[1:]
                else:
                    # We have arrived at the final waypoint
                    await self.turn_off_motors()
                    self.behavior_index += 1
                    return

            # We are not yet at the waypoint
            else:
                break


        # Use distance to waypoint to calculate target heading
        target_heading = np.arctan2(position_error[1], position_error[0])


        current_orientation = np.arctan2(self.heading[1], self.heading[0])

        heading_error = target_heading - current_orientation
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        self.heading_error = heading_error

        abs_error = np.abs(self.heading_error)
        
        motor_full = 60
        motor_off = 50

        if self.heading_error < 0:
            # Object needs to turn left. Keep the right motor at full power,
            # and gradually decrease the left motor as error increases.
            motor_r = motor_full

            error_normalized = np.min([abs_error / heading_error_threshold, 1.0])
            motor_range = motor_full - motor_off
            motor_l = motor_full - (error_normalized * motor_range)
            if motor_l < motor_off + 1: motor_l = 0

            await self.ble.send_motor_command([int(motor_r), 0, int(motor_l), 0])

        elif self.heading_error > 0:
            # Object needs to turn RIGHT. Keep the LEFT motor at full power,
            # and gradually decrease the RIGHT motor as error increases.
            motor_l = motor_full

            error_normalized = np.min([abs_error / heading_error_threshold, 1.0])
            motor_range = motor_full - motor_off
            motor_r = motor_full - (error_normalized * motor_range)
            if motor_r < motor_off + 1: motor_r = 0

            await self.ble.send_motor_command([int(motor_r), 0, int(motor_l), 0])

        else:
            await self.ble.send_motor_command([motor_full,0,motor_full,0])

        self.last_centroid = self.centroid
    

    def plan_path_towards(self, table, destination_coord, omit=[]):


        map = table.construct_obstacle_map(r_expand=30, use_known_objects=True, omit=[self]+omit)

        start_pos = self.centroid
        start_orientation = round(self.orientation_radians * 8 / (2*np.pi)) % 8

        goal_threshold = 10
        goal_fn = routefinding.make_goal_NEAR(
            destination_coord[0],
            destination_coord[1],
            goal_threshold
        )
        
        heuristic_fn = routefinding.make_heuristic_NEAR(
            destination_coord[0],
            destination_coord[1]
        )
        print("physical object's plan_path_towards is called, calling plan_and_refine_path")
        path = self.plan_and_refine_path(map, start_pos, start_orientation, goal_fn, heuristic_fn)

        if path and len(path) > 1:
            self.waypoints = path[1:]
            print("### plan path towards self.waypoints", self.waypoints)


    def plan_path_away(self, table, repelling_coord):

        map = table.construct_obstacle_map(r_expand=5, use_known_objects=True, omit=[self])

        start_pos = self.centroid
        start_orientation = round(self.orientation_radians * 8 / (2*np.pi)) % 8

        goal_threshold = 200
        goal_fn = routefinding.make_goal_AWAY(
            repelling_coord[0],
            repelling_coord[1],
            goal_threshold
        )
        
        heuristic_fn = routefinding.make_heuristic_AWAY(
            repelling_coord[0],
            repelling_coord[1]
        )

        path = self.plan_and_refine_path(map, start_pos, start_orientation, goal_fn, heuristic_fn)

        if path and len(path) > 1:
            self.waypoints = path[1:]
        
    
    def plan_and_refine_path(self, map, start_pos, start_orientation, goal, heuristic):
        print(f"Planning path from {start_pos} to {goal}")
        path = routefinding.astar_with_motion_primitives(
            int(start_pos[0]), int(start_pos[1]), start_orientation,
            goal,
            heuristic,
            map,
            self.primitives
        )

        if not path: 
            print("No path found")
            #
            return None

        path_refined = routefinding.line_of_sight_optimization(path, map)

        return path_refined


    def clear_waypoints(self):
        """
        Clear any planned waypoints and reset behavior state so the object
        stops following a path and idles until given a new action.
        """
        self.waypoints = []
        self.target_heading = None
        self.behaviors = ['', '', '']
        self.behavior_index = 0

class DifferentialDriveController:
    def __init__(self, device_address=None, device_name=None):
        self.device_address = device_address
        self.device_name = device_name
        self.client = None
        self.motor_values = [0, 0, 0, 0]
        self.is_connected = False

    async def scan_for_device(self, target_name=None):
        if not target_name: return None

        print(f'Searching for BLE device with name {target_name}')
        devices = await BleakScanner.discover()
        for device in devices:
            if device.name == target_name:
                print(f'Found device! UUID: {device.address}')
                return device.address

        print(f'Could not find device with name {target_name}')
        return None

    async def connect(self):

        # If a UUID is provided, try to connect via UUID.


        # If a UUID is provided, try to connect via UUID. Return if successful.
        if self.device_address:
            if await self.connect_to_address(self.device_address): return True

        # If UUID connection fails (or is not provided), search for candidate
        # devices, and try to connect via device_name.
        if self.device_name:
            self.device_address = await self.scan_for_device(self.device_name)
            if not self.device_address: return False
            if await self.connect_to_address(self.device_address): return True

        return False
        
    async def connect_to_address(self, address):
        try:
            print(f'Attempting to connect to device {address}')
            self.client = BleakClient(address)
            await self.client.connect()
            self.is_connected = self.client.is_connected
            
            if self.is_connected:
                print(f'SUCCESS! Connected to device {address}')
                return True
            else:
                print(f'FAILURE. Could not connect to device {address}')
                return False
            
        except Exception as e:
            print(f"ERROR connecting to device: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the BLE device."""
        if self.client and self.client.is_connected:

            await self.client.disconnect()
            print("Disconnected from device.")
        self.is_connected = False

    async def send_motor_command(self, motor_values):
        """Send the current motor commands to the device."""
        if not self.is_connected or self.client is None or not self.client.is_connected:
            # raise RuntimeError("BLE client not connected; cannot send motor command")
            print("BLE client not connected; cannot send motor command")
            
        try:
            command = bytearray([
                motor_values[0],
                motor_values[1],
                motor_values[2],
                motor_values[3]
            ])
            
            await self.client.write_gatt_char(MOTOR_CHARACTERISTIC_UUID, command)
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            # # Escalate to terminate the program if desired (unhandled in task)
            # raise RuntimeError(f"Error sending command: {e}") from e
        

Point2D = np.ndarray[2, np.int_]
Vector2D = np.ndarray[2, np.int_]

def signed_heading_error(v_from: Vector2D, v_to: Vector2D) -> float:
    return np.arctan2(np.cross(v_from, v_to), np.dot(v_from, v_to))
