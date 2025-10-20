# src/user_goal.py
from dataclasses import asdict, dataclass

from base_agent import LLMAgent


@dataclass
class GoalGeneratorInput:
    location: str
    current_activity: str
    summarized_activity: str
    activity_sequence: str
    previous_goal: str


class GoalGenerator(LLMAgent):
    def __init__(
        self,
        prompt_path: str = "prompts/user_goal.yaml",
        model: str = "gpt-5-mini",
    ):
        super().__init__(prompt_path, model)
        self.previous_goal = "N/A"

    def format_user_message(self, input: GoalGeneratorInput) -> str:
        return self.user_message_template.format(**asdict(input))

    async def query(self, input: GoalGeneratorInput) -> str:
        formatted_user_message = self.format_user_message(input)

        response = await self.client.responses.create(
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
            reasoning={"effort": "minimal"},
        )

        # # Return the raw text; prompt enforces "Selected Goal: ..." format
        # selected_goal = response.output_text.strip().replace("Selected Goal: ", "")
        selected_goal = response.output_text
        self.previous_goal = selected_goal
        return selected_goal


@dataclass
class GoalGeneratorInput:
    location: str
    current_activity: str
    summarized_activity: str
    activity_sequence: str
    previous_goal: str
