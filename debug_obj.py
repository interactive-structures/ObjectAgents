"""
Object Detection Debugging Module

Run this script separately to debug object detection issues.
It will show detailed information about what YOLO detects and how it matches to known objects.
"""

import asyncio
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera import CameraCapture, init_camera
from tablespace import TableSpace
from physical_object import PhysicalObject


class ObjectDetectionDebugger:
    def __init__(self, camera_index=0, config_path="config/scene_config.yaml"):
        self.camera_index = camera_index
        self.config_path = config_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize YOLO models
        self.yolo_obb = YOLO(self.config['obj_detection_model'])
        self.yolo_class_mapping = self.config['yolo_class_mapping']
        
        # Load known objects from config
        self.known_objects = []
        self._load_known_objects()
        
        print("=" * 80)
        print("OBJECT DETECTION DEBUGGER")
        print("=" * 80)
        print(f"Model: {self.config['obj_detection_model']}")
        print(f"Class mapping: {self.yolo_class_mapping}")
        print(f"Known objects: {[ob.name for ob in self.known_objects]}")
        print("=" * 80)
    
    def _load_known_objects(self):
        """Load known objects from config (same logic as TableSpace)"""
        for object_name in self.config.get("active_objects", {}):
            object_info = self.config["active_objects"][object_name]
            motor_info = self.config.get("motor_params", {}).get(object_name)
            if motor_info is not None:
                motor_range_r = (motor_info[0], motor_info[1])
                motor_range_l = (motor_info[2], motor_info[3])
            else:
                motor_range_r = (50, 70)
                motor_range_l = (45, 65)
            
            ob = PhysicalObject(
                name=object_name,
                uuid=object_info.get('uuid', ''),
                device_name=object_info.get('device_name', ''),
                heading_offset=object_info.get('heading_offset', 0.0),
                motor_range_r=motor_range_r,
                motor_range_l=motor_range_l
            )
            self.known_objects.append(ob)
        
        for object_name in self.config.get("non_active_objects", []):
            ob = PhysicalObject(name=object_name)
            self.known_objects.append(ob)
    
    def debug_detection(self, frame):
        """Run object detection with detailed debugging output"""
        print("\n" + "=" * 80)
        print("FRAME ANALYSIS")
        print("=" * 80)
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        
        # Run YOLO detection
        try:
            results = self.yolo_obb.track(frame, persist=True, verbose=False)
            
            if not results or len(results) == 0:
                print("❌ YOLO returned empty results")
                return None, frame
            
            result = results[0]
            print(f"✅ YOLO result type: {type(result)}")
            print(f"Has 'obb' attribute: {hasattr(result, 'obb')}")
            
            if not hasattr(result, 'obb') or result.obb is None:
                print("❌ No 'obb' attribute in results")
                return None, frame
            
            if len(result.obb) == 0:
                print("⚠️  No objects detected in this frame")
                return None, frame
            
            # Extract detection data
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            xywhr = result.obb.xywhr.cpu().numpy()
            classes = result.obb.cls.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()
            ids = result.obb.id.int().cpu().tolist() if result.obb.id is not None else []
            
            print(f"\n📊 DETECTION SUMMARY")
            print(f"   Total detections: {len(ids)}")
            print(f"   Classes detected: {classes.tolist()}")
            print(f"   Confidences: {[f'{c:.3f}' for c in confs]}")
            print(f"   Track IDs: {ids}")
            
            # Create visualization frame
            vis_frame = frame.copy()
            matched_objects = []
            
            print(f"\n🔍 DETAILED ANALYSIS")
            print("-" * 80)
            
            for i in range(len(ids)):
                class_id = int(classes[i])
                confidence = float(confs[i])
                track_id = ids[i]
                
                # Get class name from mapping
                detected_name = self.yolo_class_mapping.get(class_id, f"unknown_{class_id}")
                
                print(f"\n  Detection {i+1}:")
                print(f"    Class ID: {class_id} -> '{detected_name}'")
                print(f"    Confidence: {confidence:.3f}")
                print(f"    Track ID: {track_id}")
                
                # Check confidence threshold
                if confidence < 0.8:
                    print(f"    ⚠️  SKIPPED: Confidence {confidence:.3f} < 0.8 threshold")
                    # Draw in gray for low confidence
                    self._draw_bbox(vis_frame, boxes[i], f"{detected_name} ({confidence:.2f})", (128, 128, 128))
                    continue
                
                # Try to match to known object
                matched_object = None
                match_method = None
                
                # Method 1: Match by track ID
                for ob in self.known_objects:
                    if ob.yolo_id == track_id:
                        matched_object = ob
                        match_method = "track_id"
                        break
                
                # Method 2: Match by name
                if matched_object is None:
                    for ob in self.known_objects:
                        if ob.name == detected_name:
                            matched_object = ob
                            match_method = "name"
                            # Assign track ID
                            ob.yolo_id = track_id
                            break
                
                if matched_object:
                    print(f"    ✅ MATCHED: '{matched_object.name}' (via {match_method})")
                    print(f"       Is active: {matched_object.is_active}")
                    matched_objects.append((matched_object, boxes[i], confidence))
                    # Draw in green for matched
                    self._draw_bbox(vis_frame, boxes[i], f"{matched_object.name} ({confidence:.2f})", (0, 255, 0))
                else:
                    print(f"    ❌ NO MATCH: '{detected_name}' not in known objects")
                    print(f"       Known objects: {[ob.name for ob in self.known_objects]}")
                    # Draw in red for unmatched
                    self._draw_bbox(vis_frame, boxes[i], f"{detected_name} ({confidence:.2f})", (0, 0, 255))
            
            print("\n" + "=" * 80)
            print(f"✅ Matched objects: {len(matched_objects)}")
            print(f"❌ Unmatched detections: {len(ids) - len(matched_objects)}")
            print("=" * 80)
            
            return matched_objects, vis_frame
            
        except Exception as e:
            print(f"❌ ERROR in detection: {e}")
            import traceback
            traceback.print_exc()
            return None, frame
    
    def _draw_bbox(self, frame, bbox, label, color):
        """Draw oriented bounding box on frame"""
        bbox_int = bbox.astype(np.int32)
        cv2.polylines(frame, [bbox_int], isClosed=True, color=color, thickness=2)
        
        # Draw label
        if len(bbox_int) > 0:
            x, y = bbox_int[0]
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    async def run_interactive(self):
        """Run interactive debugging session"""
        print("\n🚀 Starting interactive debugging session...")
        print("Press 'q' to quit, 's' to save current frame, SPACE to pause/resume")
        
        # Initialize camera
        camera = CameraCapture(self.camera_index)
        await camera.ready()
        
        paused = False
        frame_count = 0
        
        try:
            while True:
                if not paused:
                    frame = camera.get_latest_frame_rgb()
                    if frame is None:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Transform frame (if using TableSpace transformation)
                    # For now, use raw frame - you can add transformation if needed
                    transformed_frame = frame.copy()
                    
                    matched_objects, vis_frame = self.debug_detection(transformed_frame)
                    frame_count += 1
                    
                    # Display frame
                    display_frame = cv2.resize(vis_frame, (1280, 720))
                    cv2.imshow("Object Detection Debug", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"debug_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    print(f"💾 Saved frame to {filename}")
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            camera.release()
            cv2.destroyAllWindows()
            print("✅ Debugging session ended")
    
    async def run_single_frame(self):
        """Run detection on a single frame"""
        print("\n🚀 Capturing single frame...")
        
        camera = CameraCapture(self.camera_index)
        await camera.ready()
        
        frame = camera.get_latest_frame_rgb()
        if frame is None:
            print("❌ Failed to capture frame")
            return
        
        matched_objects, vis_frame = self.debug_detection(frame)
        
        # Display result
        display_frame = cv2.resize(vis_frame, (1280, 720))
        cv2.imshow("Object Detection Debug - Single Frame", display_frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        camera.release()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug object detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--config", type=str, default="config/scene_config.yaml", 
                       help="Path to scene config (default: config/scene_config.yaml)")
    parser.add_argument("--mode", type=str, choices=["interactive", "single"], 
                       default="interactive", help="Debug mode (default: interactive)")
    
    args = parser.parse_args()
    
    debugger = ObjectDetectionDebugger(camera_index=args.camera, config_path=args.config)
    
    if args.mode == "interactive":
        await debugger.run_interactive()
    else:
        await debugger.run_single_frame()


if __name__ == "__main__":
    asyncio.run(main())