import cv2
import numpy as np
import os
import json
import sys
from datetime import datetime

# Global variables
points = []
drawing = False
adjust_mode = False
label_mode = False
selected_point = -1
selected_line = -1  # Index of selected line in label mode
adjustment_step = 1
rotated = False
frame = None
cap = None
labels = {}  # Dictionary to store labels for line segments
current_label = ""  # Buffer for current label being typed

def point_to_line_distance(point, line_start, line_end):
    """Calculate the shortest distance from a point to a line segment"""
    # Vector from line_start to line_end
    line_vec = np.array(line_end) - np.array(line_start)
    line_length = np.linalg.norm(line_vec)
    if line_length == 0:
        return np.linalg.norm(np.array(point) - np.array(line_start))
    
    # Vector from line_start to point
    point_vec = np.array(point) - np.array(line_start)
    
    # Calculate projection
    projection = np.dot(point_vec, line_vec) / line_length
    
    # Clamp projection to line segment
    projection = max(0, min(line_length, projection))
    
    # Calculate closest point on line
    closest_point = np.array(line_start) + (projection / line_length) * line_vec
    
    # Return distance to closest point
    return np.linalg.norm(np.array(point) - closest_point)

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing, point selection, and line selection"""
    global points, drawing, adjust_mode, label_mode, selected_point, selected_line
    
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    
    if label_mode and len(points) == 4:
        # Find the closest line segment
        min_distance = float('inf')
        closest_line = -1
        
        for i in range(4):
            start_point = points[i]
            end_point = points[(i + 1) % 4]  # Wrap around to the first point
            
            distance = point_to_line_distance((x, y), start_point, end_point)
            if distance < min_distance and distance < 20:  # 20 pixel threshold
                min_distance = distance
                closest_line = i
        
        if closest_line != -1:
            selected_line = closest_line
    elif adjust_mode and len(points) > 0:
        # Find the closest point
        distances = [np.sqrt((p[0] - x)**2 + (p[1] - y)**2) for p in points]
        closest_idx = np.argmin(distances)
        selected_point = closest_idx if distances[closest_idx] < 30 else -1
    else:
        points.append((x, y))
        drawing = True

def create_mask(config_path):
    """Create and save a binary mask if we have at least 3 points"""
    if len(points) < 3:
        print("Need at least 3 points to create a mask")
        return
        
    # Get config filename without path or extension
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    # Create tablespaces directory if it doesn't exist
    os.makedirs("tablespaces", exist_ok=True)
    
    # Create blank mask
    mask = np.ones_like(frame) * 255  # White background
    
    # Fill polygon area with black
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (0, 0, 0))
    
    # Resize to 1280x720
    mask_resized = cv2.resize(mask, (1280, 720))
    
    # Create mask filename
    mask_filename = f"{config_name}-mask.png"
    
    # Save the mask
    save_path = f"tablespaces/{mask_filename}"
    cv2.imwrite(save_path, mask_resized)
    
    # Update config file with mask filename
    # Load existing config or create new
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Add mask filename to config
    config["table_mask"] = mask_filename
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Mask created and saved as {save_path} (1280x720)")
    print(f"Config file {config_path} updated with mask reference")

def save_coordinates(config_path):
    """Save quadrilateral coordinates if we have exactly 4 points"""
    if len(points) != 4:
        print(f"Need exactly 4 points to save coordinates (currently have {len(points)})")
        return
        
    # Create directory if needed
    os.makedirs("tablespaces", exist_ok=True)
    
    # Scale coordinates to 1280x720
    h, w = frame.shape[:2]
    scaled_coords = [[int((p[0] / w) * 1280), int((p[1] / h) * 720)] for p in points]
    
    # Load existing config or create new
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Update coordinates in config
    config["source_coords"] = scaled_coords
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Quadrilateral coordinates saved to {config_path} (scaled to 1280x720)")

def calculate_box(config_path):
    """Calculate box dimensions based on point[0], width, and label ratio"""
    if len(points) != 4:
        print("Need exactly 4 points to calculate box dimensions")
        return
        
    if 0 not in labels or 1 not in labels:
        print("Need labels for the first two edges to calculate ratio")
        return
    
    try:
        # Get width and label values
        width_label = float(labels[0])
        height_label = float(labels[1])
        ratio = height_label / width_label
        
        # Get width vector (from point[0] to point[1]) - only for physical measurement
        width_vec = np.array(points[1]) - np.array(points[0])
        width_pixels = np.linalg.norm(width_vec)
        
        # Calculate the height based on the ratio
        height_pixels = width_pixels * ratio
        
        # Create a non-rotated box aligned with screen axes
        top_left = points[0]
        top_right = (top_left[0] + width_pixels, top_left[1])
        bottom_right = (top_left[0] + width_pixels, top_left[1] + height_pixels)
        bottom_left = (top_left[0], top_left[1] + height_pixels)
        
        # Convert to list of points
        box_points = [top_left, top_right, bottom_right, bottom_left]
        
        # Scale coordinates to 1280x720
        h, w = frame.shape[:2]
        scaled_box = [[int((p[0] / w) * 1280), int((p[1] / h) * 720)] for p in box_points]
        
        # Load existing config or create new
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Update coordinates in config
        config["destination_coords"] = scaled_box
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Box dimensions calculated:")
        print(f"Width: {width_pixels:.1f} pixels ({width_label} units)")
        print(f"Height: {height_pixels:.1f} pixels ({height_label} units)")
        print(f"Aspect ratio: {ratio:.3f}")
        print(f"Box coordinates saved to {config_path} as 'destination_coords' (scaled to 1280x720)")
        
    except ValueError:
        print("Error: Labels must be valid numbers for ratio calculation")

def capture_background(config_path):
    """Capture and save the current frame as a background image"""
    # Get config filename without path or extension
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    # Create tablespaces directory if it doesn't exist
    os.makedirs("tablespaces", exist_ok=True)
    
    # Get a clean frame (with rotation if enabled, but no UI elements)
    ret, clean_frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        return
        
    # Apply rotation if enabled
    if rotated:
        clean_frame = cv2.rotate(clean_frame, cv2.ROTATE_180)
    
    # Resize to 1280x720
    clean_frame = cv2.resize(clean_frame, (1280, 720))
    
    # Create background filename
    background_filename = f"{config_name}-background.png"
    
    # Save the image
    save_path = f"tablespaces/{background_filename}"
    cv2.imwrite(save_path, clean_frame)
    
    # Update config file with background filename
    # Load existing config or create new
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Add background filename to config
    config["table_background"] = background_filename
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Background captured and saved as {save_path}")
    print(f"Config file {config_path} updated with background reference")

def toggle_rotation(config_path):
    """Toggle frame rotation and update config file"""
    global rotated
    
    # Toggle rotation state
    rotated = not rotated
    
    # Update config file
    # Load existing config or create new
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Add rotation state to config
    config["flip"] = rotated
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Rotation {'enabled' if rotated else 'disabled'} and config updated")

def toggle_label_mode():
    """Toggle label mode (only if we have a quadrilateral)"""
    global label_mode, selected_line, current_label
    
    if len(points) != 4:
        print("Label mode requires exactly 4 points (quadrilateral)")
        return False
    
    label_mode = not label_mode
    selected_line = -1
    current_label = ""
    
    if label_mode:
        print("Entered LABEL MODE - Click on a line to select it, then type a number and press Enter")
    else:
        print("Exited LABEL MODE")
    
    return True

def handle_key_press(key, config_path):
    """Process keyboard inputs"""
    global points, adjust_mode, label_mode, selected_point, selected_line, current_label, labels
    
    # Handle Enter key in label mode
    if label_mode and key == 13 and selected_line != -1 and current_label:  # 13 is Enter key
        labels[selected_line] = current_label
        current_label = ""
        selected_line = -1
        return True
    
    # Handle spacebar in label mode to calculate box
    if label_mode and key == 32:  # 32 is spacebar
        calculate_box(config_path)
        return True
    
    # Handle number keys and decimal point in label mode
    if label_mode and selected_line != -1:
        # Numbers 0-9
        if 48 <= key <= 57:
            current_label += chr(key)
            return True
        # Decimal point (period) - only add if not already present
        elif key == 46 and '.' not in current_label:
            current_label += '.'
            return True
    
    # Handle backspace in label mode
    if label_mode and selected_line != -1 and key == 8 and current_label:  # 8 is Backspace key
        current_label = current_label[:-1]
        return True
    
    # Standard key commands
    if key == ord('q'):
        return False  # Signal to exit
    elif key == ord('c'):
        points = []
        if label_mode:
            label_mode = False
            print("Exited LABEL MODE (points cleared)")
        labels = {}  # Clear labels when points are cleared
    elif key == ord('r'):
        toggle_rotation(config_path)
    elif key == ord('m'):
        create_mask(config_path)
    elif key == ord('s'):
        save_coordinates(config_path)
    elif key == ord('b'):
        capture_background(config_path)
    elif key == ord('a'):
        if not label_mode:  # Can't enter adjust mode from label mode
            adjust_mode = not adjust_mode
            selected_point = -1
            print(f"{'Entered' if adjust_mode else 'Exited'} ADJUST MODE")
    elif key == ord('l'):
        if not adjust_mode:  # Can't enter label mode from adjust mode
            toggle_label_mode()
    
    # Handle arrow keys in adjust mode
    if adjust_mode and selected_point != -1 and len(points) > selected_point:
        x, y = points[selected_point]
        # Check for arrow key presses (handle multiple key codes for compatibility)
        if key in [82, 0, 63232, 2490368]:  # Up
            points[selected_point] = (x, y - adjustment_step)
        elif key in [84, 1, 63233, 2621440]:  # Down
            points[selected_point] = (x, y + adjustment_step)
        elif key in [81, 2, 63234, 2424832]:  # Left
            points[selected_point] = (x - adjustment_step, y)
        elif key in [83, 3, 63235, 2555904]:  # Right
            points[selected_point] = (x + adjustment_step, y)
    
    return True  # Continue running

def draw_interface(frame):
    """Draw polygon, points, labels and status information on frame"""
    # Draw polygon if we have points
    if len(points) > 0:
        # Draw all points
        for i, point in enumerate(points):
            color = (0, 255, 0) if adjust_mode and i == selected_point else (0, 0, 255)
            cv2.circle(frame, point, 14, (0,0,0), -1)  # Black outline
            cv2.circle(frame, point, 10, color, -1)    # Fill color
        
        # Draw lines connecting points
        if len(points) > 1:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            
            if label_mode and len(points) == 4:
                # In label mode, draw individual lines so we can highlight the selected one
                for i in range(4):
                    start_point = points[i]
                    end_point = points[(i + 1) % 4]
                    
                    # Use green for selected line, blue for others
                    color = (0, 255, 0) if i == selected_line else (255, 0, 0)
                    thickness = 5 if i == selected_line else 3
                    
                    cv2.line(frame, start_point, end_point, color, thickness)
                    
                    # Draw label if it exists
                    if i in labels:
                        # Calculate midpoint of the line
                        mid_x = (start_point[0] + end_point[0]) // 2
                        mid_y = (start_point[1] + end_point[1]) // 2
                        
                        # Draw white background for text
                        text_size = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                        cv2.rectangle(frame, 
                                     (mid_x - text_size[0]//2 - 5, mid_y - text_size[1]//2 - 5),
                                     (mid_x + text_size[0]//2 + 5, mid_y + text_size[1]//2 + 5),
                                     (255, 255, 255), -1)
                        
                        # Draw text
                        cv2.putText(frame, labels[i], (mid_x - text_size[0]//2, mid_y + text_size[1]//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            else:
                # Standard drawing mode - draw all lines at once
                cv2.polylines(frame, [pts], False, (255, 0, 0), 3)
                
                # Close the polygon if we have at least 3 points
                if len(points) >= 3:
                    cv2.line(frame, points[-1], points[0], (255, 0, 0), 3)
    
    # Draw status information
    if label_mode:
        mode_text = "LABEL MODE"
        if selected_line != -1:
            mode_text += f" - Line {selected_line + 1} selected: {current_label}"
    elif adjust_mode:
        mode_text = "ADJUST MODE"
    else:
        mode_text = "DRAW MODE"
        
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Shadow
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # Text
    
    return frame

def print_instructions():
    """Print usage instructions"""
    print("Instructions:")
    print("- Click on the video feed to place polygon vertices")
    print("- Press 'c' to clear all points")
    print("- Press 'r' to rotate the frame 180 degrees")
    print("- Press 'm' to create and save a binary mask (black inside polygon, white outside)")
    print("- Press 's' to save quadrilateral coordinates to the config file (requires exactly 4 points)")
    print("- Press 'b' to capture and save the current frame as a background image")
    print("- Press 'a' to enter/exit adjust mode (click to select a point, use arrow keys to move it)")
    print("- Press 'l' to enter/exit label mode (requires exactly 4 points)")
    print("  > In label mode, click a line segment to select it")
    print("  > Type numbers (with optional decimal point) and press Enter to assign to the selected line")
    print("  > Press spacebar to calculate box dimensions using:")
    print("    * Point[0] as top-left corner")
    print("    * Width from Point[0] to Point[1]")
    print("    * Height calculated from ratio of first two edge labels")
    print("- Press 'q' to quit")

def main():
    global frame, cap, rotated, label_mode  # Make variables accessible to other functions
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_filename>")
        print("Example: python script.py config.json")
        return
    
    config_path = os.path.join("tablespaces", sys.argv[1])
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Setup window and callback
    window_name = "Webcam Polygon Drawing"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print_instructions()
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        # Apply rotation if enabled
        if rotated:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Draw points, lines, and UI elements
        frame = draw_interface(frame)
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF
        if not handle_key_press(key, config_path):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()