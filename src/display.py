"""
Display module for stitching and showing camera feeds.
"""

import asyncio

import cv2
import numpy as np

from camera import CameraCapture
from controloop import ControlLoop


class CameraDisplay:
    """Handles stitching and displaying multiple camera feeds."""

    def __init__(
        self,
        top_down_capture: CameraCapture,
        env_capture: CameraCapture,
        motion_control_loop : ControlLoop,
        table_space=None,
        agent=None,
        display_width: int = 3456,
        display_height: int = 2234,
        window_name: str = "Object Agents",
        sleep: float = 0.05,
        enable_dev_window: bool = False,
        additional_height: int = 0,
    ):
        """
        Initialize the camera display.

        Args:
            top_down_capture: Top-down camera capture instance
            env_capture: Environment camera capture instance
            table_space: TableSpace instance for object detection visualization
            agent: Agent instance for displaying agent outputs
            display_width: Width of the combined display window
            display_height: Height of the display window
            window_name: Name of the display window
            enable_dev_window: Whether to show additional developer window with top-down view
            additional_height: Additional height in pixels to add as black space at the bottom
        """
        self.top_down_capture = top_down_capture
        self.env_capture = env_capture
        self.table_space = table_space
        self.agent = agent
        self.display_width = display_width
        self.display_height = display_height
        self.window_name = window_name
        self.sleep = sleep
        self.enable_dev_window = enable_dev_window
        self.dev_window_name = "Developer - Top Down View"
        self.additional_height = additional_height

        # Fixed layout dimensions - manually adjust these values as needed
        self.camera_width = 640 #640
        self.camera_height = 360 #480
        self.right_panel_width = 1000

        # Click-based user position system
        self.manual_user_position = None  # Position in table coordinates
        self.manual_user_bbox = None  # Bounding box for visualization
        self.manual_selection_mode = False  # Whether manual selection mode is active
        
        # Button for toggling manual mode in developer window
        self.button_rect = None  # Will be set when developer window is created
        self.button_height = 40
        self.button_width = 200
        self._button_debug_printed = False
        self._button_positioned = False  # Track if button position is set
        
        self.y_perceive = 40
        self.y_reason = 500 #500
        self.y_act = 780

        self.perceive_rect_h = 400
        self.reason_rect_h = 200
        self.act_rect_h = 200
        
        # Calculate panel height to accommodate all sections
        # Panel height should be at least: y_act + act_rect_h
        self.panel_height = max(self.camera_height * 2, self.y_act + self.act_rect_h)
        
        # Initialize the display window (resizable for projector setups)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
        # Initialize developer window if enabled (fixed size, no resizing)
        if self.enable_dev_window:
            cv2.namedWindow(self.dev_window_name, cv2.WINDOW_AUTOSIZE)  # Fixed size based on image
        
        # Mouse callback will be set up in the loop after the first frame is shown
        self._callback_set = False
        self._dev_callback_set = False
        
        # Pre-create agent panel background with static elements
        self._init_agent_panel_background()

        self.motion_control_loop = motion_control_loop

    def _init_agent_panel_background(self):
        """Pre-create the agent panel background with static rectangles and headers."""
        self.agent_panel_background = np.zeros((self.panel_height, self.right_panel_width, 3), dtype=np.uint8)
        
        # Draw black background
        self.agent_panel_background[:] = (0, 0, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 4
        header_color = (0, 0, 0)
        
        # PERCEIVE section rectangles and header
        cv2.rectangle(self.agent_panel_background, (0, self.y_perceive-40), (150, self.y_perceive + 10), (255, 255, 255), -1)
        cv2.putText(self.agent_panel_background, "Perceive", (10, self.y_perceive), font, font_scale, header_color, thickness)
        cv2.rectangle(self.agent_panel_background, (0, self.y_perceive+5), (self.right_panel_width-10, self.y_perceive + self.perceive_rect_h), (255, 255, 255), -1)
        
        # REASON section rectangles and header
        cv2.rectangle(self.agent_panel_background, (0, self.y_reason-40), (140, self.y_reason + 10), (255, 255, 255), -1)
        cv2.putText(self.agent_panel_background, "Reason", (10, self.y_reason), font, font_scale, header_color, thickness)
        cv2.rectangle(self.agent_panel_background, (0, self.y_reason+5), (self.right_panel_width-10, self.y_reason + self.reason_rect_h), (255, 255, 255), -1)
        
        # ACT section rectangles and header
        cv2.rectangle(self.agent_panel_background, (0, self.y_act-40), (80, self.y_act + 10), (255, 255, 255), -1)
        cv2.putText(self.agent_panel_background, "Act", (10, self.y_act), font, font_scale, header_color, thickness)
        cv2.rectangle(self.agent_panel_background, (0, self.y_act+5), (self.right_panel_width-10, self.y_act + self.act_rect_h), (255, 255, 255), -1)

    def _dev_mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks in developer window to set manual user position or toggle mode."""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"🖱️  Click at ({x}, {y})")
            
            # Check if click is on the toggle button
            if self._is_click_on_button(x, y):
                print("🖱️  Button clicked! Toggling mode...")
                self._toggle_manual_mode()
            elif self.manual_selection_mode:

                self.motion_control_loop.display_click_coord = [x, y]
                self.manual_user_bbox = [
                    [x, y],
                    [x+10, y],
                    [x+10, y+10],
                    [x, y+10],
                ]

                # if self.instruction_queue is not None:
                #     self.instruction_queue.put_nowait({
                #         'active_object': 'stapler',
                #         'target': f'coordinate_{x}_{y}',
                #         'action': 'move_towards',
                #     })

                # # Convert developer window coordinates to table coordinates
                # table_position = self._dev_coords_to_table_coords(x, y)
                # if table_position is not None:
                #     self.manual_user_position = table_position
                #     # Create a simple bounding box around the clicked point for visualization
                #     self.manual_user_bbox = self._create_manual_user_bbox(table_position)
                #     print(f"✅ Manual user position set at table coordinates: ({table_position[0]:.1f}, {table_position[1]:.1f})")

            else:
                print("ℹ️  Manual selection mode disabled. Click the toggle button to enable.")

    def _dev_coords_to_table_coords(self, dev_x: int, dev_y: int) -> np.ndarray | None:
        """Convert developer window coordinates to table coordinates."""
        if self.table_space is None:
            return None
        
        # Get the latest transformed frame to check its dimensions
        transformed_frame = self.table_space.get_latest_transformed_frame()
        if transformed_frame is None:
            return None
            
        # Direct 1:1 mapping since developer window shows frame at original size
        original_height, original_width = transformed_frame.shape[:2]
        
        # Ensure coordinates are within frame bounds
        if 0 <= dev_x < original_width and 0 <= dev_y < original_height:
            return np.array([dev_x, dev_y], dtype=float)
        else:
            return None

    def _create_manual_user_bbox(self, center_position: np.ndarray) -> np.ndarray:
        """Create a bounding box around the manual user position for visualization."""
        # Create a simple rectangular bounding box around the clicked point
        bbox_size = 40  # Size of the bounding box
        x, y = center_position
        
        return np.array([
            [x - bbox_size/2, y - bbox_size/2],  # top-left
            [x + bbox_size/2, y - bbox_size/2],  # top-right
            [x + bbox_size/2, y + bbox_size/2],  # bottom-right
            [x - bbox_size/2, y + bbox_size/2]   # bottom-left
        ])

    def get_manual_user_position(self) -> np.ndarray | None:
        """Get the manual user position in table coordinates."""
        return self.manual_user_position

    def clear_manual_user_position(self):
        """Clear the manual user position."""
        self.manual_user_position = None
        self.manual_user_bbox = None
        print("🗑️  Manual user position cleared")

    def _toggle_manual_mode(self):
        """Toggle manual selection mode on/off."""
        if self.manual_selection_mode:
            self.manual_selection_mode = False
            self.motion_control_loop.display_click_coord = None
            print("🚪 Manual user selection mode DISABLED - returning to YOLO detection only")
        else:
            self.manual_selection_mode = True
            print("🎯 Manual user selection mode ENABLED - click in developer window to set user position")

    def _is_click_on_button(self, x: int, y: int) -> bool:
        """Check if click coordinates are within the toggle button."""
        if self.button_rect is None:
            return False
        
        bx, by, bw, bh = self.button_rect
        return bx <= x <= bx + bw and by <= y <= by + bh

    def _update_button_position(self, frame_width: int, frame_height: int):
        """Update button position based on frame size."""
        # Position button in top-right corner with some margin
        margin = 10
        button_x = frame_width - self.button_width - margin
        button_y = 330  # bottom of frame
        
        # Ensure button fits within frame
        if button_x < 0:
            button_x = margin
        if button_y + self.button_height > frame_height:
            button_y = frame_height - self.button_height - margin
        
        self.button_rect = (button_x, button_y, self.button_width, self.button_height)
        
        # Debug: Print button position (only once)
        if not self._button_debug_printed:
            print(f"🖱️  Button positioned at: x={button_x}, y={button_y}, w={self.button_width}, h={self.button_height}")
            print(f"🖱️  Frame size: {frame_width}x{frame_height}")
            self._button_debug_printed = True

    def _resize_frame_fixed(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fixed camera dimensions."""
        return cv2.resize(frame, (self.camera_width, self.camera_height))

    def _create_placeholder_frame(self, text: str = "No Frame") -> np.ndarray:
        """Create a placeholder frame when camera data is not available."""
        placeholder = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

        # Add text to the placeholder
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2

        # Get text size to center it within the placeholder itself
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        ph_h, ph_w = placeholder.shape[:2]
        text_x = (ph_w - text_size[0]) // 2
        text_y = (ph_h + text_size[1]) // 2

        cv2.putText(
            placeholder, text, (text_x, text_y), font, font_scale, color, thickness
        )

        return placeholder

    def _compose_left_column(
        self, top_down_frame: np.ndarray | None, env_frame: np.ndarray | None
    ) -> np.ndarray:
        """Stack two frames vertically on the left with fixed dimensions."""
        # Prepare frames or placeholders
        if top_down_frame is None:
            top_down_frame = self._create_placeholder_frame("Top-Down Camera")
        else:
            top_down_frame = self._resize_frame_fixed(top_down_frame)
            # top_down_frame = top_down_frame
        
        if env_frame is None:
            env_frame = self._create_placeholder_frame("Environment Camera")
        else:
            env_frame = self._resize_frame_fixed(env_frame)
            # top_down_frame = env_frame

        # Add camera labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 0)
        thickness = 2
        
        # Top-down label (clean - no mode indicator)
        cv2.putText(top_down_frame, "Planning view", (10, 30), font, font_scale, color, thickness)
        cv2.putText(env_frame, "Agent view", (10, 30), font, font_scale, color, thickness)

        # Stack vertically
        left_column = np.vstack((top_down_frame, env_frame))
        return left_column

    def _wrap_text(self, text: str, font, font_scale: float, thickness: int, max_width: int) -> list[str]:
        """Wrap text to fit within max_width pixels."""
        if not text:
            return [""]
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [""]

    def _draw_wrapped_text(self, panel, text: str, x: int, y: int, font, font_scale: float, color, thickness: int, max_width: int, line_height: int, max_height: int = None) -> int:
        """Draw wrapped text and return the y position after the last line.
        
        Args:
            max_height: Optional maximum height boundary. If provided, text will be clipped
                       when it would exceed this y-coordinate.
        """
        lines = self._wrap_text(text, font, font_scale, thickness, max_width)
        current_y = y
        
        for i, line in enumerate(lines):
            # Check if drawing this line would exceed the maximum height boundary
            if max_height is not None and current_y + line_height > max_height:
                # Add ellipsis to indicate text was truncated, but only if there are more lines
                remaining_lines = len(lines) - i
                if remaining_lines > 0 and current_y <= max_height:
                    ellipsis_text = "..."
                    cv2.putText(panel, ellipsis_text, (x, current_y), font, font_scale, color, thickness)
                break
                
            cv2.putText(panel, line, (x, current_y), font, font_scale, color, thickness)
            current_y += line_height
        
        return current_y

    def _create_agent_panel(self, panel_width: int, panel_height: int) -> np.ndarray:
        """Create the agent output panel on the right with fixed header positions."""
        if self.agent is None:
            # optimizing display without agent 
            # # No agent available, show placeholder
            # panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            # panel[:] = (255, 255, 255)
            # cv2.putText(panel, "Agent not available", (10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # return panel
            outputs = {
                        'perceive': {'activity': 
                                        'Context-aware computing focuses on systems that adapt their be- 298 havior based on information about the user’s situation [12]. Early 299 work by Dey et al. [11] established frameworks for context recogni- 300 tion that enable applications to respond appropriately to environ- 301 mental changes. More recent approaches leverage machine learning 302 to understand complex user contexts and predict appropriate sys- 303 tem responses [24].With the current LLMs’ capabilities, systems 304 also use LLMs’ knowledge to gain context-awareness for describ- 305 ing live scenes fo visually-impaired users [9], control embedded 306 interfaces through various. environments [27], and so on Our work 307 extends this line by applying contextual reasoning specifically to 308 physical objects. While prior context-aware systems typically adapt 309 digital interfaces or smart environments [27, 44], Object Agents 310 apply contextual intelligence to the objects themselves, enabling 311 granular, object-specific responses to user needs.', 
                                      'summary': 'The concept of agency—the ability to act autonomously to achieve 333 goals—has been explored in many interactive systems. q', 
                                      'narration': ["Mixed-initiative 334 interfaces [29] share control between users and automated pro- 335 cesses, while intelligent agents [74] act on behalf of users to ac- 336 complish tasks in many digital task domains, such as web naviga- 337 tion [79], slide generation [19], software engineering [70], and so 338 on.", 
                                                    "Closer to the physical world are works doing activity monitor- 339 ing during everyday procedural tasks. In this context, agents can 340 observe user activities, and decide on when to intervene to help [3]."
                                                    "Besides interacting with the user, generative agents supported by 342 an LLM backbone can also interact among themselves, and be used 343 to create a social simulacra [49]."], 'timestamp': 0},
                        'reason': {'goal': 
                                        'Initializing...egh,cuoisghbciulgqbwliugbadeiubciudjskbjcmbdwa,chjxhmzNbcjsavchievcecxdscsecdzxdbiuwacvbeiuwavcbliuewqbcliu', 
                                        'timestamp': 0},
                        'act': {
                            'move goal': 'Initializing...bd kdjsmnzcbjdnbc,jkaddheiugcliueragcubs,cmxnzb mxzncxzndsjhcb,sEOAbcoueabcouweblciowbqaccjcbseic.bhe;oiawhbciobcudibgcluizdbscliubdac.kijzbnc,dea,i',
                            'justification': 'Justification justification is blahblahblah justification blahblahblah justification blah and more justification:generative agents supported by 342 an LLM backbone can also interact among themselves, and be used 343 to create a social simulacra',
                            'alignment_score': None,
                            'alignment_passed': False,
                            'timestamp': 0,
                        }
                    }
        else:
            # Get latest agent outputs
            outputs = self.agent.latest_outputs

        # Start with pre-created background (copy to avoid modifying original)
        panel = self.agent_panel_background.copy()
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        normal_thickness = 2
        text_color = (0, 0, 0)
        header_color = (0, 0, 0)

        # Text wrapping parameters
        text_x_position = 40  # Where text actually starts
        text_margin = 60  # Right margin for safety
        max_text_width = panel_width - text_x_position - text_margin
        line_height = 22
        small_line_height = 30

        
        # PERCEIVE details (background rectangles and headers already drawn)
        y_current = self.y_perceive + line_height + 5
        perceive_max_y = self.y_perceive + self.perceive_rect_h # Bottom of perceive rectangle
        
        perceive_summary = outputs['perceive'].get('summary', '')
        narration_segments = outputs['perceive'].get('narration', [])
        
        # Check if we're in initializing state (no real data yet)
        is_initializing = (perceive_summary in ['N/A', 'Initializing...'] and len(narration_segments) == 0)
        
        if is_initializing:
            # Show initializing message
            initializing_text = "Initializing..."
            y_current = self._draw_wrapped_text(panel, initializing_text, text_x_position, y_current + 10, font, 0.8, (128, 128, 128), normal_thickness, max_text_width, small_line_height, perceive_max_y)
        else:
            # Show actual content
            if perceive_summary and perceive_summary not in ['N/A', 'Initializing...']:
                summary_text = f"{perceive_summary}"
                y_current = self._draw_wrapped_text(panel, summary_text, text_x_position, y_current + 10, font, 0.8, text_color, normal_thickness, max_text_width, small_line_height, perceive_max_y)
                y_current += 5  # Small gap

            if narration_segments and len(narration_segments) > 0:
                # Display each narration segment on a new line
                for i, segment in enumerate(narration_segments):
                    narration_text = f"{segment}".strip()
                    # prefix = "Narration:" if i == 0 else ""  # Only show label on first line
                    if i == 0:
                        y_current = self._draw_wrapped_text(panel, narration_text, text_x_position + 10, y_current + 10, font, 0.7, text_color, normal_thickness, max_text_width, small_line_height, perceive_max_y)
                    else:
                        y_current = self._draw_wrapped_text(panel, narration_text, text_x_position + 10, y_current + 5, font, 0.7, text_color, normal_thickness, max_text_width, small_line_height-2, perceive_max_y)

        # REASON details (background rectangles and headers already drawn)
        y_current = self.y_reason + line_height + 5
        reason_max_y = self.y_reason + self.reason_rect_h  # Bottom of reason rectangle
        reason_goal = outputs['reason'].get('goal', '')
        
        if reason_goal in ['N/A', 'Initializing...']:
            # Show initializing message
            initializing_text = "Initializing..."
            y_current = self._draw_wrapped_text(panel, initializing_text, text_x_position, y_current + 10, font, 0.8, (128, 128, 128), normal_thickness, max_text_width, small_line_height, reason_max_y)
        elif reason_goal and len(reason_goal) > 50:  # If goal is long, wrap it too
            y_current = self._draw_wrapped_text(panel, reason_goal, text_x_position, y_current +10, font, 0.8, text_color, normal_thickness, max_text_width, small_line_height, reason_max_y)

        # ACT section - redraw header with dynamic color based on alignment status
        act_passed = outputs['act'].get('alignment_passed', False)
        act_header_color = (0, 100, 0) if act_passed else header_color
        # act_thickness = 5 if act_passed else normal_thickness
        
        # # Redraw just the header text with dynamic color (background rectangles already drawn)
        # cv2.putText(panel, "Act", (10, self.y_act), font, font_scale, act_header_color, act_thickness)
        
        # ACT details below header
        y_current = self.y_act + line_height + 25
        act_max_y = self.y_act + self.act_rect_h  # Bottom of act rectangle
        justification_body = outputs['act'].get('justification', '')
        
        if justification_body in ['N/A', 'Initializing...']:
            # Show initializing message
            initializing_text = "Initializing..."
            y_current = self._draw_wrapped_text(panel, initializing_text, text_x_position, y_current, font, 0.8, (128, 128, 128), normal_thickness, max_text_width, small_line_height, act_max_y)
        elif justification_body:
            justification_text = justification_body
            just_thickness = 3 if act_passed else normal_thickness
            y_current = self._draw_wrapped_text(panel, justification_text, text_x_position, y_current, font, 0.8, text_color, just_thickness, max_text_width, small_line_height, act_max_y)

        # # Border
        # cv2.rectangle(panel, (0, 0), (panel_width-1, panel_height-1), (200, 200, 200), 1)

        return panel

    def get_latest_stitched_frame(self) -> np.ndarray:
        """Get the latest stitched frame from both cameras."""
        # Get latest frames from both cameras
        top_down_latest = self.top_down_capture.get_latest_frame()
        env_latest = self.env_capture.get_latest_frame()

        # Extract frame data (handle case where frames might be None)
        if top_down_latest.seq >= 0:
            if self.table_space is not None:
                # Use the cached transformed frame to avoid double transformation
                transformed_frame = self.table_space.get_latest_transformed_frame()
                if transformed_frame is not None:
                    # Draw bounding boxes on the already-transformed frame
                    top_down_frame = self._draw_bounding_boxes_on_frame(transformed_frame)
                else:
                    # Fallback to raw frame if no transformed frame available yet
                    top_down_frame = top_down_latest.frame_rgb
            else:
                # No table_space available, use raw frame
                top_down_frame = top_down_latest.frame_rgb
        else:
            top_down_frame = None

        env_frame = env_latest.frame_rgb if env_latest.seq >= 0 else None

        # Optionally overlay paths on top-down frame
        if top_down_frame is not None:
            top_down_frame = self._draw_paths_on_frame(top_down_frame)

        # Compose left column (two camera frames stacked vertically)
        left_column = self._compose_left_column(top_down_frame, env_frame)
        
        # Pad left column to match panel height if needed
        left_column_height = left_column.shape[0]
        if left_column_height < self.panel_height:
            padding_height = self.panel_height - left_column_height
            padding = np.zeros((padding_height, left_column.shape[1], 3), dtype=np.uint8)
            left_column = np.vstack((left_column, padding))

        # Create agent panel with calculated dimensions to fit all sections
        agent_panel = self._create_agent_panel(self.right_panel_width, self.panel_height)

        # Combine left (cameras) and right (text panel)
        full_display = np.hstack((left_column, agent_panel))

        # Add additional black height at the bottom if specified
        if self.additional_height > 0:
            _, current_width = full_display.shape[:2]
            black_padding = np.zeros((self.additional_height, current_width, 3), dtype=np.uint8)
            full_display = np.vstack((full_display, black_padding))

        return full_display

    def _draw_bounding_boxes_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes on a frame if table_space is available.
        Returns a copy of the frame with visualizations.
        """
        if self.table_space is None:
            return frame

        frame_with_boxes = frame.copy()

        # Draw bounding boxes for all detected objects
        self._draw_object_bounding_boxes(
            frame_with_boxes,
            self.table_space.detected_robotic_objects,
            (0, 255, 0),
            "Robotic"
        )  # Green
        self._draw_object_bounding_boxes(
            frame_with_boxes,
            self.table_space.detected_pushable_objects,
            (255, 0, 0),
            "Pushable"
        )  # Blue
        self._draw_object_bounding_boxes(
            frame_with_boxes,
            self.table_space.detected_landmark_objects,
            (0, 0, 255),
            "Landmark"
        )  # Red

        # Draw user bounding box if detected (YOLO only in main display)
        if (self.table_space.user is not None and
            hasattr(self.table_space.user, 'bbox') and
            self.table_space.user.bbox is not None):
            self._draw_user_bounding_box(
                frame_with_boxes,
                self.table_space.user,
                (255, 255, 0),
                "User"
            )  # Yellow

        return frame_with_boxes

    def _draw_paths_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw planned paths (waypoints) for all known objects on the frame.
        """
        if self.table_space is None:
            return frame

        frame_with_paths = frame.copy()
        path_color = (0, 255, 255)  # Yellow
        start_color = (0, 165, 255)  # Orange for start/current
        goal_color = (0, 255, 0)  # Green for goal

        for obj in getattr(self.table_space, 'known_objects', []):
            waypoints = getattr(obj, 'waypoints', None)
            if not waypoints or len(waypoints) == 0:
                continue

            # Build full path starting at current centroid
            try:
                points = [obj.centroid.astype(np.int32)] + [np.asarray(wp, dtype=np.int32) for wp in waypoints]
                # print(f"waypoints: {waypoints}")
            except Exception:
                continue

            # Draw polyline of the path
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_with_paths, [pts], isClosed=False, color=path_color, thickness=2)

            # Mark start and goal
            cv2.circle(frame_with_paths, tuple(points[0]), 6, start_color, thickness=-1)
            cv2.circle(frame_with_paths, tuple(points[-1]), 6, goal_color, thickness=-1)

    
        return frame_with_paths

    def _show_developer_window(self):
        """Display the developer window with enhanced top-down view."""
        if self.table_space is None:
            return
            
        # Get the latest transformed frame
        transformed_frame = self.table_space.get_latest_transformed_frame()
        if transformed_frame is None:
            return
            

        obstacle_map = self.table_space.construct_obstacle_map(r_expand=10, use_known_objects=True).copy()
        obstacle_map = cv2.cvtColor(obstacle_map, cv2.COLOR_GRAY2BGR)

        ## NEW ISSUE #####################################################
        # Resize obstacle_map to match transformed_frame dimensions
        if transformed_frame.shape[:2] != obstacle_map.shape[:2]:
            obstacle_map = cv2.resize(obstacle_map, (transformed_frame.shape[1], transformed_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        ###############################################################

        #overlay the obstacle map on the transformed frame
        alpha = 0.5  # Transparency factor
        beta = 1 - alpha
        gamma = 0  # Scalar added to each sum

        # Create a copy for the developer window
        # dev_frame = transformed_frame.copy()
        dev_frame = cv2.addWeighted(transformed_frame, alpha, obstacle_map, beta, gamma)
                                     
        # Draw all visualizations on the developer frame (including manual user)
        dev_frame = self._draw_dev_bounding_boxes_on_frame(dev_frame)
        dev_frame = self._draw_paths_on_frame(dev_frame)
        
        # Add developer-specific overlays
        self._add_developer_overlays(dev_frame)
        
        # Show in developer window
        cv2.imshow(self.dev_window_name, dev_frame)

    def _draw_dev_bounding_boxes_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes on developer frame including manual user position.
        Returns a copy of the frame with visualizations.
        """
        if self.table_space is None:
            return frame

        frame_with_boxes = frame.copy()

        # Draw bounding boxes for all detected objects
        self._draw_object_bounding_boxes(
            frame_with_boxes,
            self.table_space.detected_robotic_objects,
            (0, 255, 0),
            "Robotic"
        )  # Green
        self._draw_object_bounding_boxes(
            frame_with_boxes,
            self.table_space.detected_pushable_objects,
            (255, 0, 0),
            "Pushable"
        )  # Blue
        self._draw_object_bounding_boxes(
            frame_with_boxes,
            self.table_space.detected_landmark_objects,
            (0, 0, 255),
            "Landmark"
        )  # Red

        # Draw YOLO user bounding box if detected
        if (self.table_space.user is not None and
            hasattr(self.table_space.user, 'bbox') and
            self.table_space.user.bbox is not None):
            self._draw_user_bounding_box(
                frame_with_boxes,
                self.table_space.user,
                (255, 255, 0),
                "User (YOLO)"
            )  # Yellow

        # Draw manual user position if set (only in developer window)
        if self.manual_user_bbox is not None:
            self._draw_manual_user_bounding_box(
                frame_with_boxes,
                self.manual_user_bbox,
                (0, 255, 255),  # Cyan
                "User (Manual)"
            )

        return frame_with_boxes

    def _add_developer_overlays(self, frame):
        """Add developer-specific information overlays to the frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.2
        thickness = 1
        
        height, width = frame.shape[:2]
        
        # Update button position only once
        if not self._button_positioned:
            self._update_button_position(width, height)
            self._button_positioned = True
        
        # Add developer window title
        # title_text = "DEVELOPER WINDOW"
        # title_color = (0, 255, 255) if self.manual_selection_mode else (255, 255, 255)
        # cv2.putText(frame, title_text, (10, 30), font, 0.8, title_color, thickness)
        
        # Draw toggle button
        self._draw_toggle_button(frame)
        
        # Add mode status with click instructions
        if self.manual_selection_mode:
            mode_text = "Manual Mode: ON - Click anywhere to set user position"
            mode_color = (0, 255, 255)
        else:
            mode_text = "Manual Mode: OFF - Click button to enable"
            mode_color = (128, 128, 128)
        cv2.putText(frame, mode_text, (10, height - 60), font, font_scale, mode_color, thickness)
        
        # # Add frame info
        # info_text = f"Frame: {width}x{height}"
        # cv2.putText(frame, info_text, (10, height - 90), font, font_scale, (255, 255, 255), thickness)
        
        # # Add user position info
        # if self.manual_user_position is not None:
        #     pos_text = f"Manual User: ({self.manual_user_position[0]:.1f}, {self.manual_user_position[1]:.1f})"
        #     cv2.putText(frame, pos_text, (10, height - 120), font, font_scale, (0, 255, 255), thickness)
        
        # Add YOLO user position if available
        if (self.table_space.user is not None and 
            hasattr(self.table_space.user, 'bbox') and 
            self.table_space.user.bbox is not None):
            yolo_centroid = np.mean(self.table_space.user.bbox, axis=0)
            yolo_text = f"YOLO User: ({yolo_centroid[0]:.1f}, {yolo_centroid[1]:.1f})"
            cv2.putText(frame, yolo_text, (10, height - 150), font, font_scale, (255, 255, 0), thickness)

    def _draw_toggle_button(self, frame):
        """Draw the toggle button on the developer window."""
        if self.button_rect is None:
            return
            
        bx, by, bw, bh = self.button_rect
        
        # Button colors based on state
        if self.manual_selection_mode:
            button_color = (0, 200, 0)  # Green when ON
            text_color = (255, 255, 255)
            button_text = "Manual Mode: ON"
        else:
            button_color = (100, 100, 100)  # Gray when OFF
            text_color = (255, 255, 255)
            button_text = "Manual Mode: OFF"
        
        # Draw button background
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), button_color, -1)
        
        # Draw button border
        border_color = (255, 255, 255)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), border_color, 2)
        
        # Draw button text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Center text in button
        text_size = cv2.getTextSize(button_text, font, font_scale, thickness)[0]
        text_x = bx + (bw - text_size[0]) // 2
        text_y = by + (bh + text_size[1]) // 2
        
        cv2.putText(frame, button_text, (text_x, text_y), font, font_scale, text_color, thickness)

    def _draw_object_bounding_boxes(self, frame, objects, color, label_prefix):
        """Draw bounding boxes for a set of objects."""
        for obj in objects:
            if obj.bbox is not None and len(obj.bbox) == 4:
                # Convert bbox to integer coordinates
                bbox_int = np.array(obj.bbox, dtype=np.int32)

                # Draw the oriented bounding box
                cv2.polylines(frame, [bbox_int], True, color, 4)

                # Draw object name and confidence
                centroid = obj.centroid
                text = f"{label_prefix}: {obj.name}"
                if hasattr(obj, 'confidence') and obj.confidence is not None:
                    text += f" ({obj.confidence:.2f})"

                # Position text above the bounding box
                text_pos = (int(centroid[0]), int(centroid[1]) - 10)
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    def _draw_user_bounding_box(self, frame, user, color, label):
        """Draw bounding box for the user."""
        if user.bbox is not None and len(user.bbox) == 4:
            # Convert bbox to integer coordinates
            bbox_int = np.array(user.bbox, dtype=np.int32)

            # Draw the bounding box
            cv2.polylines(frame, [bbox_int], True, color, 4)

            # Draw user label and confidence
            centroid = user.centroid
            text = f"{label}"
            if hasattr(user, 'confidence') and user.confidence is not None:
                text += f" ({user.confidence:.2f})"

            # Position text above the bounding box
            text_pos = (int(centroid[0]), int(centroid[1]) - 10)
            cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    def _draw_manual_user_bounding_box(self, frame, bbox, color, label):
        """Draw bounding box for the manual user position."""
        # Convert bbox to integer coordinates
        bbox_int = np.array(bbox, dtype=np.int32)

        # Draw the bounding box
        cv2.polylines(frame, [bbox_int], True, color, 6)

        # Calculate centroid for text placement
        centroid = np.mean(bbox, axis=0)
        
        # Position text above the bounding box
        text_pos = (int(centroid[0]), int(centroid[1]) - 10)
        cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    async def loop(self):
        """
        Main display loop that continuously shows stitched camera feeds.

        Args:
            fps: Target frames per second for display
        """

        try:
            while True:
                # Get the latest stitched frame
                stitched_frame = self.get_latest_stitched_frame()

                # Display the frame
                cv2.imshow(self.window_name, stitched_frame)
                
                # Display developer window with top-down view if enabled
                if self.enable_dev_window:
                    self._show_developer_window()
                
                # Set up mouse callback for developer window after the first frame is shown
                if self.enable_dev_window and not self._dev_callback_set:
                    cv2.setMouseCallback(self.dev_window_name, self._dev_mouse_callback)
                 
                    self._dev_callback_set = True

                # Check for key presses with longer timeout for better responsiveness
                key = cv2.waitKey(5) & 0xFF  # Increased from 1ms to 5ms
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    # Clear manual user position
                    self.clear_manual_user_position()
                
                # Shorter sleep for better responsiveness
                await asyncio.sleep(0.01)  # Reduced from 0.05s to 0.01s

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
