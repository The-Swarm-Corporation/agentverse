from swarms import (
    Agent as SwarmsAgent,
)  # Import the base Agent class from Swarms
from griptape.structures import Agent as GriptapeAgent
from typing import Any, List


class GriptapeSwarmsAgent(SwarmsAgent):
    """
    A custom agent class that inherits from SwarmsAgent, designed to execute tasks involving web scraping, summarization, and file management.
    """

    def __init__(
        self,
        name: str = "Griptape Agent",
        description: str = "A custom agent class that inherits from SwarmsAgent, designed to execute tasks involving web scraping, summarization, and file management.",
        system_prompt: str = None,
        tools: List[Any] = [],
        *args,
        **kwargs,
    ):
        """
        Initialize the GriptapeSwarmsAgent with its tools.

        Args:
            *args: Additional positional arguments to pass to the GriptapeAgent.
            **kwargs: Additional keyword arguments to pass to the GriptapeAgent.
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools
        # Initialize the Griptape agent with its tools
        self.agent = GriptapeAgent(
            input="Load {{ args[0] }}, summarize it, and store it in a file called {{ args[1] }}.",
            tools=tools,
            *args,
            **kwargs,
        )

    def run(self, task: str) -> str:
        """
        Execute a task using the Griptape agent.

        Args:
            task (str): The task to be executed, formatted as "URL, filename".

        Returns:
            str: The final result of the task execution as a string.

        Raises:
            ValueError: If the task string does not contain exactly one comma.
        """
        result = self.agent.run(self.system_prompt + task)
        # Return the final result as a string
        return str(result)


# # Example usage:
# griptape_swarms_agent = GriptapeSwarmsAgent(system_prompt="What is the weather in Tokyo?")
# output = griptape_swarms_agent.run("What is the weather in Tokyo?")
# print(output)
