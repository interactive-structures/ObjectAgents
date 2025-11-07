import asyncio

from agent_loop import Agent
from camera import init_camera
from display import CameraDisplay
from tablespace import TableSpace
from controloop import ControlLoop

async def main():

    # Single source of truth for alignment threshold    
    alignment_threshold = 0.5

    TOP_DOWN_CAMERA_INDEX = 0
    ENV_CAMERA_INDEX = 0
    agent_loop_debugging = False
    env_video_path = "test_videos/p1_orgdoc-720p.mov"

    top_down_capture, env_capture = init_camera(TOP_DOWN_CAMERA_INDEX, ENV_CAMERA_INDEX, agent_loop_debugging, env_video_path)
    object_detection= TableSpace(top_down_capture, "config/scene_config.yaml")
    
    instruction_queue = asyncio.LifoQueue()
    robots_control = ControlLoop(object_detection, instruction_queue)   

    display = CameraDisplay(top_down_capture, env_capture, robots_control, object_detection, None, enable_dev_window=True)  # Agent will be set later
    
    agent = Agent(
        env_capture,
        top_down_capture,
        object_detection.detected_robotic_objects,
        object_detection.detected_pushable_objects,
        object_detection.detected_landmark_objects,
        max_inflight=2,
        min_spawn_interval=3.0,
        alignment_threshold=alignment_threshold,
        instruction_queue = instruction_queue
    )
    
    # Set the agent reference in display after agent is created
    display.agent = agent

    tasks = [
        top_down_capture.loop(),
        env_capture.loop(),
        object_detection.loop(),
        agent.loop(),
        robots_control.loop(),
        display.loop(),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
