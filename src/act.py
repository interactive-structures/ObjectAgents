import yaml
from dataclasses import asdict, dataclass

from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from base_agent import LLMAgent


@dataclass
class ActionGeneratorInput:
    """Inputs required to generate a helpful action plan for moving objects."""

    location: str
    summarized_activity: str
    activity: str
    current_activity: str
    goal_selection: str
    movable_objects: str
    movable_objects_properties: str
    spatial_memory_objects: str
    robotic_objects_name_only: str

@dataclass
class AlignmentCheckerInput:
    goal: str
    activity_narrative: str
    action_justification: str


class AlignmentCheckerOutput(BaseModel):
    reasoning: str
    score: float

class ActionGeneratorOutput(BaseModel):
    """Structured output for an action plan to move objects."""

    object_to_move: str
    # robotic_object_to_push: str
    move_goal: str
    # movement_destination: str
    target: str
    # presentation: str
    action: str
    justification: str


class ActionGenerator(LLMAgent):
    """Agent that decides which object to move, where, and how to present it."""

    def __init__(
        self,
        prompt_path: str = "prompts/action_generator.yaml",
        object_knowledge_path: str = "prompts/object_knowledge.yaml",
        model: str = "gpt-4o-mini",
    ):
        super().__init__(prompt_path, model)
        with open(object_knowledge_path, 'r') as object_knowledge_yaml:
            self.object_knowledge = yaml.safe_load(object_knowledge_yaml)

    def format_user_message(self, input: ActionGeneratorInput) -> str:
        """Format the user message with provided action generator input variables."""
        return self.user_message_template.format(**asdict(input))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=10),
        reraise=True,
    )
    async def query(self, input: ActionGeneratorInput, base64_image: str) -> ActionGeneratorOutput:
        """Query the model with structured input and get a structured action plan."""
        formatted_user_message = self.format_user_message(input)
        # print("#####################ACTION INPUT######################\n", formatted_user_message)

        response = await self.client.responses.parse(
            model=self.model,
            instructions=self.system_message_template,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": formatted_user_message},
                        {
                            "type": "input_image",
                            "image_url": base64_image,
                        },
                    ],
                }
            ],
            temperature=0.0,
            # reasoning={"effort": "minimal"},
            text_format=ActionGeneratorOutput,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise Exception("query() returns None in ActionGenerator")

        return parsed

class AlignmentChecker(LLMAgent):
    def __init__(
        self,
        prompt_path: str = "prompts/alignment_checker.yaml",
        model: str = "gpt-4.1-nano",
    ):
        super().__init__(prompt_path, model)

    def format_user_message(self, input: AlignmentCheckerInput) -> str:
        return self.user_message_template.format(**asdict(input))

    async def query(self, input: AlignmentCheckerInput) -> float:
        formatted_user_message = self.format_user_message(input)
        response = await self.client.responses.parse(
            model=self.model,
            instructions=self.system_message_template,
            input=[{"role": "user", "content": [{"type": "input_text", "text": formatted_user_message}]}],
            temperature=0.0,
            # reasoning={"effort": "minimal"},
            text_format=AlignmentCheckerOutput,
        )
        return response.output_parsed
