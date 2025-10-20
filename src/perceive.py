import asyncio
import heapq
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from base_agent import LLMAgent


@dataclass
class FrameDescriptorInput:
    """Input data for the frame descriptor to analyze video frames."""

    location: str
    previous_description: str = ""


# class FrameDescriptorAdditionalObject(BaseModel):
#     object_name: str
#     object_position: str


class FrameDescriptorOutput(BaseModel):
    """Structured output from the frame descriptor analysis."""

    # additional_objects: list[FrameDescriptorAdditionalObject]
    activity: str
    similarity_flag: bool


class NarratorOutput(BaseModel):
    """Structured output from the narrator analysis."""

    overall_summary: str
    sequence_summaries: list[str]
    # important_details: list[str]


@dataclass
class NarratorInput:
    """Input data for the narrator to process activity descriptions."""

    stored_narration: str
    timestamp: str
    new_activity_description: str


class FrameDescriptor(LLMAgent):
    """Frame descriptor agent that analyzes webcam frames to describe human activities."""

    def __init__(
        self,
        prompt_path: str = "prompts/frame_descriptor.yaml",
        model: str = "gpt-4o-mini",
        outputs_max_size: int = 200,
    ):
        """
        Initialize the FrameDescriptor agent.

        Args:
            prompt_path: Path to the frame descriptor prompt YAML file
            model: OpenAI model to use
        """
        super().__init__(prompt_path, model)
        self.outputs_max_size = outputs_max_size
        self.outputs: list[tuple[float, FrameDescriptorOutput]] = []

        # Track background tasks to satisfy linters and avoid orphaned tasks
        self._tasks: set[asyncio.Task[Any]] = set()

    def format_user_message(self, input: FrameDescriptorInput) -> str:
        """Format the user message with provided frame descriptor input variables."""
        return self.user_message_template.format(**asdict(input))

    def get_latest_output(self) -> FrameDescriptorOutput | None:
        if not self.outputs:
            return None
        return max(self.outputs, key=lambda item: item[0])[1]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=10),
        reraise=True,
    )
    async def query(
        self, input: FrameDescriptorInput, base64_image: str, timestamp: float
    ) -> FrameDescriptorOutput:
        """Query the frame descriptor model with structured input and get structured output."""
        formatted_user_message = self.format_user_message(input)

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
            text_format=FrameDescriptorOutput,
        )

        parsed = response.output_parsed
        if parsed is None:
            raise Exception("query() returns None in FrameDescriptor")

        if len(self.outputs) < self.outputs_max_size:
            heapq.heappush(self.outputs, (timestamp, parsed))
            # print("\nHeap", self.outputs)
        else:
            heapq.heappushpop(self.outputs, (timestamp, parsed))

        return parsed


class Narrator(LLMAgent):
    """Narrator agent that analyzes activity descriptions and creates meaningful narratives."""

    def __init__(
        self,
        prompt_path: str = "prompts/narrator.yaml",
        model: str = "gpt-4.1-nano",
    ):
        """
        Initialize the Narrator agent.

        Args:
            prompt_path: Path to the narrator prompt YAML file
            model: OpenAI model to use
        """
        super().__init__(prompt_path, model)

        self.previous_narrator_output: NarratorOutput | None = None

    def format_user_message(self, input: NarratorInput) -> str:
        """Format the user message with provided narrator input variables."""
        return self.user_message_template.format(**asdict(input))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=10),
        reraise=True,
    )
    async def query(self, input: NarratorInput) -> NarratorOutput:
        """Query the narrator model with structured input and get structured output."""
        formatted_user_message = self.format_user_message(input)
        print("#####################NARRATOR INPUT######################\n", formatted_user_message)
        print(formatted_user_message)
        response = await self.client.responses.parse(
            model=self.model,
            instructions=self.system_message_template,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": formatted_user_message},
                    ],
                }
            ],
            # reasoning={"effort": "minimal"},
            temperature=0.0,
            text_format=NarratorOutput,
        )

        parsed = response.output_parsed

        if parsed is None:
            raise Exception("query() returns None in Narrator")

        self.previous_narrator_output = parsed
        return parsed
