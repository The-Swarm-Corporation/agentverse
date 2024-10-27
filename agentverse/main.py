from typing import Generator, List, Union
from interpreter import interpreter
from loguru import logger
from agentverse.formatter import ConversationParser

# Configure loguru logger
logger.add("app.log", rotation="500 MB")


class OpenInterpreterAgent:
    """
    Wrapper class for the open-interpreter package with lazy loading.

    This class provides a convenient interface to interact with the open-interpreter
    package, offering features like lazy loading, task execution, and message management.

    Attributes:
        interpreter: The underlying interpreter instance.
        model (str): The model to be used by the interpreter.
        auto_install (bool): Whether to automatically install dependencies.
        auto_run (bool): Whether to automatically run commands.

    """

    def __init__(
        self,
        name: str = "interpreter-agent-01",
        description: str = "An interpreter for the open-interpreter package",
        model: str = "gpt-3.5-turbo",
        auto_install: bool = True,
        auto_run: bool = True,
        system_prompt: str = "",
    ):
        """
        Initialize the OpenInterpreterAgent.

        Args:
            model (str): The model to be used by the interpreter.
            auto_install (bool): Whether to automatically install dependencies.
            auto_run (bool): Whether to automatically run commands.
        """
        self.name = name
        self.description = description
        self.interpreter = None
        self.model = model
        self.auto_install = auto_install
        self.auto_run = auto_run
        self.system_prompt = system_prompt
        self.interpreter = interpreter

        if self.auto_run:
            self.interpreter.auto_run = True

        logger.info(
            f"OpenInterpreterAgent initialized with model: {model}"
        )

    def __call__(self, task: str, **kwargs) -> Union[List, Generator]:
        """
        Allow the class to be called directly.

        Args:
            task (str): The task to be executed.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[List, Generator]: The result of the task execution.
        """
        return self.run(self.step(task), **kwargs)

    def step(self, task: str, **kwargs) -> Union[List, Generator]:
        return self.name + self.system_prompt + task

    def run(
        self, task: str, stream: bool = False, display: bool = True
    ) -> Union[List, Generator]:
        """
        Execute a task using the interpreter.

        Args:
            task (str): The task to be executed.
            stream (bool): Whether to stream the output.
            display (bool): Whether to display the output.

        Returns:
            Union[List, Generator]: The result of the task execution.

        Raises:
            Exception: If there's an error executing the task.
        """
        try:
            logger.info(f"Executing task: {task}")
            output = self.interpreter.chat(
                self.step(task), stream=stream, display=display
            )
            print(type(output))
            # Ensure the output is a string
            if isinstance(output, list):
                return "\n".join(str(item) for item in output)
            elif isinstance(output, dict):
                output = str(output)
            else:
                output = str(output)
            return ConversationParser.extract_conversation(output)
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            raise

    def reset(self):
        """Reset the interpreter's message history."""
        if self.interpreter:
            self.interpreter.messages = []
            logger.info("Interpreter message history reset")

    @property
    def messages(self) -> List:
        """
        Get current message history.

        Returns:
            List: The current message history.
        """
        return self.interpreter.messages if self.interpreter else []

    @messages.setter
    def messages(self, messages: List):
        """
        Set message history.

        Args:
            messages (List): The new message history to set.
        """
        if self.interpreter:
            self.interpreter.messages = messages
            logger.info("Message history updated")

    @property
    def system_message(self) -> str:
        """
        Get system message.

        Returns:
            str: The current system message.
        """
        return (
            self.interpreter.system_message
            if self.interpreter
            else ""
        )

    @system_message.setter
    def system_message(self, message: str):
        """
        Set system message.

        Args:
            message (str): The new system message to set.
        """
        if self.interpreter:
            self.interpreter.system_message = message
            logger.info("System message updated")
