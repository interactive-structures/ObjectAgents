import asyncio
import json
import time
from collections.abc import Callable
from asyncio import Queue, QueueFull

import numpy as np

from camera import CameraCapture
from perceive import (
    FrameDescriptor,
    FrameDescriptorInput,
    FrameDescriptorOutput,
    Narrator,
    NarratorInput,
    NarratorOutput,
)
from reason import GoalGenerator, GoalGeneratorInput
from act import (
    ActionGenerator,
    ActionGeneratorInput,
    AlignmentChecker,
    AlignmentCheckerInput,
)
from physical_object import PhysicalObject


class Agent:
    def __init__(
        self,
        env_capture: CameraCapture,
        topdown_capture: CameraCapture,
        robotic_objects: (PhysicalObject),
        pushable_objects: (PhysicalObject),
        landmark_objects: (PhysicalObject),
        location: str = "office",
        sleep: float = 1.0,
        max_inflight: int = 1,
        min_spawn_interval: float = 1.0,
        on_alignment_update: Callable[[float], None] | None = None,
        on_action_plan: Callable[[dict], None] | None = None,
        alignment_threshold: float = 0.5,
        instruction_queue : asyncio.Queue | None = None,
    ):
        self.frame_descriptor = FrameDescriptor()
        self.narrator = Narrator()
        self.goal_generator = GoalGenerator()
        self.action_generator = ActionGenerator()
        self.alignment_checker = AlignmentChecker()

        self.env_capture = env_capture
        self.topdown_capture = topdown_capture
        self.location = location
        self.sleep = sleep
        self.max_inflight = max(1, int(max_inflight))
        self.min_spawn_interval = max(0.0, float(min_spawn_interval))

        self.detected_robotic_objects = robotic_objects
        self.detected_pushable_objects = pushable_objects
        self.detected_landmark_objects = landmark_objects  # these will be updated whenever a detection occurs
        self.background_tasks = set()
        self.on_alignment_update = on_alignment_update
        self.on_action_plan = on_action_plan
        self.alignment_threshold = float(alignment_threshold)

        # Store latest agent outputs for display
        self.latest_outputs = {
            'perceive': {'activity': 'Initializing...', 'summary': 'Initializing...', 'narration': [], 'timestamp': 0},
            'reason': {'goal': 'Initializing...', 'timestamp': 0},
            'act': {
                'action': 'Initializing...',
                'justification': 'Initializing...',
                'alignment_score': None,
                'alignment_passed': False,
                'timestamp': 0,
            }
        }

        # Communicate to motion controller.
        self.instruction_queue = instruction_queue

    async def loop(self):
        await self.env_capture.ready()
        # while True:
            ## -- option 1: concurrent execution --
            # task = asyncio.create_task(self.query())
            # self.background_tasks.add(task)
            # task.add_done_callback(self.background_tasks.discard)

            # # -- option 2: sequential execution --
            # try:
            #     await self.query()
            # except Exception as e:
            #     print(f"agent loop error: {e}")
            # await asyncio.sleep(self.sleep)
        # -- option 3: concurrent execution with limited inflight --
        last_spawn_ts = 0.0
        while True:
            # prune finished tasks
            self.background_tasks = {t for t in self.background_tasks if not t.done()}

            now = time.time()
            can_spawn = (
                len(self.background_tasks) < self.max_inflight
                and (now - last_spawn_ts) >= self.min_spawn_interval
            )
            if can_spawn:
                task = asyncio.create_task(self.query())
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)
                last_spawn_ts = now

            await asyncio.sleep(self.sleep)

    def get_frame_descriptor_input(self) -> FrameDescriptorInput:
        latest_frame_description = self.frame_descriptor.get_latest_output()
        return FrameDescriptorInput(
            self.location,
            latest_frame_description.activity if latest_frame_description else "N/A",
        )

    def get_narrator_input(self, timestamp: float, activity: str) -> NarratorInput:
        prev_narrator_output = self.narrator.previous_narrator_output
        if prev_narrator_output:
            stored_narration = self.narrator.previous_narrator_output.model_dump_json(
                indent=2
            )
        else:
            stored_narration = "N/A (no previous activity)"
        return NarratorInput(
            stored_narration,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            activity,
        )

    def get_goal_generator_input(
        self,
        frame_descriptor_output: FrameDescriptorOutput,
        narrator_output: NarratorOutput,
    ) -> GoalGeneratorInput:
        return GoalGeneratorInput(
            self.location,
            frame_descriptor_output.activity,
            narrator_output.overall_summary,
            str(narrator_output.sequence_summaries),
            self.goal_generator.previous_goal,
        )

    def get_action_generator_input(
        self,
        frame_descriptor_output: FrameDescriptorOutput,
        narrator_output: NarratorOutput,
        selected_goal: str,
        object_kb: dict[str, str],
    ):

        object_knowledge = {}
        robotic_objects_names = []
        spatial_memory_objects = {}
        movable_objects = []
        for obj in self.detected_robotic_objects:
            object_knowledge[obj.name] = object_kb[obj.name]
            spatial_memory_objects[obj.name] = np.array2string(obj.bbox)
            robotic_objects_names.append(obj.name)
            movable_objects.append(obj.name)
        for obj in self.detected_pushable_objects:
            object_knowledge[obj.name] = object_kb[obj.name]
            spatial_memory_objects[obj.name] = np.array2string(obj.bbox)
            movable_objects.append(obj.name)
        for obj in self.detected_landmark_objects:
            object_knowledge[obj.name] = object_kb[obj.name]
            spatial_memory_objects[obj.name] = np.array2string(obj.bbox)
            movable_objects.append(obj.name)

        return ActionGeneratorInput(
            self.location,
            narrator_output.overall_summary,
            ",".join(narrator_output.sequence_summaries),
            frame_descriptor_output.activity,
            selected_goal,
            ",".join(movable_objects),
            json.dumps(object_knowledge),
            json.dumps(spatial_memory_objects),
            ",".join(robotic_objects_names),
        )

    def get_alignment_checker_input(self, user_goal: str, activity_narrative: str, action_justification: str) -> AlignmentCheckerInput:
        return AlignmentCheckerInput(
            user_goal,
            activity_narrative,
            action_justification,
        )

    async def query(self) -> None:
        """
        Step 1.1 PERCEIVE: grab a camera frame and send to frame descriptor
        """

        env_base64_image = self.env_capture.get_latest_frame_as_openai_format()
        topdown_base64_image = self.topdown_capture.get_latest_frame_as_openai_format()
        timestamp = time.time()

        frame_descriptor_input = self.get_frame_descriptor_input()
        frame_descriptor_output = await self.frame_descriptor.query(
            frame_descriptor_input, env_base64_image, timestamp
        )
        

        if frame_descriptor_output.similarity_flag:
            # early exits if frame is similar
            # print("SIMILAR frame_descriptor_output:", frame_descriptor_output.activity)
            return
        # print("NOT SIMILAR frame_descriptor_output:", frame_descriptor_output.activity)
        
        """
            Step 1.2 PERCEIVE: narrator
        """
        narrator_input = self.get_narrator_input(
            timestamp, frame_descriptor_output.activity
        )
        # print("narrator_input:", narrator_input)
        narrator_output = await self.narrator.query(narrator_input)
        # print("narrator_output:", narrator_output)
        # Store perceive outputs
        self.latest_outputs['perceive'] = {
            'activity': frame_descriptor_output.activity,
            'summary': narrator_output.overall_summary,
            'narration': narrator_output.sequence_summaries if narrator_output.sequence_summaries else [],
            'timestamp': timestamp
        }

        """
            Step 2.1 REASON: goal generation
        """
        goal_generator_input = self.get_goal_generator_input(
            frame_descriptor_output, narrator_output
        )
        user_goal = await self.goal_generator.query(goal_generator_input)
        # print("USER GOAL:", user_goal)

        # Store reason outputs
        self.latest_outputs['reason'] = {
            'goal': user_goal,
            'timestamp': timestamp
        }

        """
            Step 3.1 ACTION: action generation
        """
        action_generator_input = self.get_action_generator_input(
            frame_descriptor_output, narrator_output, user_goal, self.action_generator.object_knowledge
        )
        

        action_output = await self.action_generator.query(action_generator_input, topdown_base64_image)
    
        

        # # Emit action plan to control loop (stored until alignment passes)
        # if self.on_action_plan is not None:
        #     try:
        #         self.on_action_plan({
        #             'object_to_move': action_output.object_to_move,
        #             'robotic_object_to_push': action_output.robotic_object_to_push,
        #             'movement_destination': action_output.movement_destination,
        #             'presentation': action_output.presentation,
        #             'action': action_output.action,
        #         })


        """
            Step 3.2 ACTION: check action-goal alignment
        """
        alignment_checker_input = self.get_alignment_checker_input(
            user_goal, narrator_output.sequence_summaries, action_output.justification
        )
        alignment_checker_output = await self.alignment_checker.query(alignment_checker_input)

        # # If aligned, then send instruction to motion control loop.
        # if alignment_checker_output.score > self.alignment_threshold:
        #     try:
        #         self.instruction_queue.put_nowait(instruction)
        #     except Full:
        #         self.instruction_queue.get()
        #         self.instruction_queue.put_nowait(instruction)

        # Update latest act outputs with alignment results
        try:
            score_value_f = float(alignment_checker_output.score) if alignment_checker_output.score is not None else None
        except Exception:
            score_value_f = None
        passed = (score_value_f is not None) and (score_value_f >= self.alignment_threshold)
        action_plan_to_send = {
                                    'active_object': action_output.object_to_move,
                                    'target': action_output.target,
                                    'action': action_output.action,
                                }
        if passed:
            print(f"Alignment passed with score: {score_value_f}\n")
            print(f"Sending action plan to control loop: {action_plan_to_send}\n")

            try:
                self.instruction_queue.put_nowait(action_plan_to_send)
            except asyncio.QueueFull:
                self.instruction_queue.put_nowait(action_plan_to_send)
                self.instruction_queue.get_nowait()

            # Store act outputs for display 
            self.latest_outputs['act'] = {
                'move_goal': action_output.move_goal,
                'justification': action_output.justification,
                'alignment_score': score_value_f,
                'alignment_passed': passed,
                'timestamp': timestamp,
        }
        # print (f"=======================OUTPUT=======================\n"
        #        f"time: {timestamp}\n"
        #        f"move goal: {action_output.move_goal}\n"
        #        f"justification: {action_output.justification}\n"
        #        f"object to move: {action_output.object_to_move}\n"
        #        f"target: {action_output.target}\n"
        #        f"action: {action_output.action}\n")
 


async def main():
    import os

    from camera import init_camera

    """Demonstrate both the narrator and frame descriptor agents."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required but not set")
    _, env_capture = init_camera()
    frame_descriptor = FrameDescriptor(env_capture)
    tasks = [
        # these two tasks open and run capture on cameras
        env_capture.loop(),
        frame_descriptor.loop(),
    ]

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
