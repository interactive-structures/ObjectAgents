import asyncio

from agent_loop import Agent
from camera import init_camera
from display import CameraDisplay
from tablespace import TableSpace
from controloop import ControlLoop


async def main():
    
    alignment_threshold = 0.5

    TOP_DOWN_CAMERA_INDEX = 0
    ENV_CAMERA_INDEX = 0
    agent_loop_debugging = False
    env_video_path = "test_videos/p1_orgdoc-720p.mov"

    top_down_capture, env_capture = init_camera(TOP_DOWN_CAMERA_INDEX, ENV_CAMERA_INDEX, agent_loop_debugging, env_video_path)
    object_detection= TableSpace(top_down_capture, "config/scene_config.yaml", "config/known_objects.yaml")
    
    display = CameraDisplay(top_down_capture, env_capture, object_detection, None, enable_dev_window=True)  # Agent will be set later
    
    #TODO: change it back, now just for debugging. V: did I put this here?
    robots_control= ControlLoop(object_detection, alignment_threshold=alignment_threshold, display=display)
    
    agent = Agent(
        env_capture,
        top_down_capture,
        object_detection.detected_robotic_objects,
        object_detection.detected_pushable_objects,
        object_detection.detected_landmark_objects,
        max_inflight=2,
        min_spawn_interval=3.0,
        on_alignment_update=robots_control.set_alignment_score,  # Wire alignment updates from Agent to ControlLoop gating
        on_action_plan=robots_control.submit_action_plan,
        alignment_threshold=alignment_threshold,
    )
    
    # Set the agent reference in display after agent is created
    display.agent = agent

    tasks = [
        top_down_capture.loop(),
        env_capture.loop(),
        object_detection.loop(),
        agent.loop(),
        # robots_control.loop(), 
        display.loop(),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
