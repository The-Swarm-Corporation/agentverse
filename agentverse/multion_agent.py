import json
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from multion.client import MultiOn



class SessionStatus(str, Enum):
    """Enum for session status values"""

    CONTINUE = "CONTINUE"
    ASK_USER = "ASK_USER"
    DONE = "DONE"


@dataclass
class SessionResponse:
    """Response data structure for session operations"""

    session_id: str
    status: SessionStatus
    message: Optional[str] = None
    screenshot: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class RetrieveResponse:
    """Response data structure for retrieve operations"""

    session_id: str
    data: List[Dict[str, Any]]
    screenshot: Optional[str] = None


def handle_errors(func: Callable) -> Callable:
    """Decorator for handling errors and logging"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


class MultiOnAgent:
    """
    Agent for handling MultiOn browser automation sessions.

    Provides functionality for creating, managing and executing browser automation
    sessions using the MultiOn API. Can be called directly with a task string.

    Example:
        >>> agent = MultiOnAgent("YOUR_API_KEY")
        >>> result = agent.run("Navigate to example.com and click login")
        # or
        >>> result = agent("Navigate to example.com and click login")
    """

    def __init__(
        self,
        api_key: str,
        name: str = "multion-agent-01",
        description: str = "An agent for handling MultiOn browser automation sessions",
        system_prompt: str = None,
        start_url: str = "https://www.google.com",
    ):
        """
        Initialize MultiOn Agent.

        Args:
            api_key (str): MultiOn API authentication key
            name (str): Name of the agent
            description (str): Description of the agent
            system_prompt (str): System prompt for the agent
            start_url (str): Starting URL for the agent
        """
        self.client = MultiOn(api_key=api_key)
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.start_url = start_url
        logger.info("MultiOnAgent initialized")

    def __call__(self, task: str, **kwargs) -> SessionResponse:
        """
        Execute a task by calling the agent directly.

        Args:
            task (str): Task description to execute
            **kwargs: Additional arguments to pass to run()

        Returns:
            SessionResponse: Response containing execution results
        """
        return self.run(task, **kwargs)

    @handle_errors
    def run(
        self,
        task: str,
        url: Optional[str] = None,
        local: bool = False,
        mode: str = "standard",
        use_proxy: bool = False,
        include_screenshot: bool = False,
    ) -> SessionResponse:
        """
        Execute a task using MultiOn browser automation.

        Args:
            task (str): Task description to execute
            url (Optional[str]): Starting URL. If None, extracted from task
            local (bool): Whether to run in local mode using Chrome extension
            mode (str): Speed mode - "standard" or "fast"
            use_proxy (bool): Whether to use proxy to bypass IP blocks
            include_screenshot (bool): Whether to include screenshot in response

        Returns:
            SessionResponse: Response containing execution results
        """
        logger.info(f"Executing task: {task}")

        # Create session
        if not url:
            # Extract URL from task if not provided
            # This is a simple example - could be made more sophisticated
            if "http" in task:
                url = task.split("http")[1].split(" ")[0]
                url = "http" + url
            else:
                raise ValueError(
                    "URL must be provided or included in task description"
                )

        session = self._create_session(
            self.start_url, local, mode, use_proxy
        )

        try:
            # Execute task
            final_response = self._execute_task(
                session.session_id, task, include_screenshot
            )
            logger.success("Task executed successfully")
            return final_response
        finally:
            # Always close session
            self._close_session(session.session_id)

    @handle_errors
    def _create_session(
        self,
        url: str,
        local: bool = False,
        mode: str = "standard",
        use_proxy: bool = False,
    ) -> SessionResponse:
        """Create a new browser automation session."""
        logger.info(f"Creating session for URL: {url}")
        response = self.client.sessions.create(
            url=url, local=local, mode=mode, use_proxy=use_proxy
        )
        return SessionResponse(
            session_id=response.session_id,
            status=SessionStatus.CONTINUE,
        )

    @handle_errors
    def _execute_task(
        self,
        session_id: str,
        task: str,
        include_screenshot: bool = False,
    ) -> SessionResponse:
        """Execute the task in the given session."""
        status = SessionStatus.CONTINUE
        last_response = None

        while status == SessionStatus.CONTINUE:
            response = self.client.sessions.step(
                session_id=session_id,
                cmd=task,
                include_screenshot=include_screenshot,
            )
            status = SessionStatus(response.status)
            last_response = response
            logger.debug(f"Step response status: {status}")

        return SessionResponse(
            session_id=session_id,
            status=status,
            message=last_response.message if last_response else None,
            screenshot=(
                last_response.screenshot if last_response else None
            ),
        )

    @handle_errors
    def _close_session(self, session_id: str) -> None:
        """Close an active session."""
        logger.info(f"Closing session: {session_id}")
        self.client.sessions.close(session_id=session_id)


class MultiOnRetrieverAgent:
    """
    Agent for retrieving structured data from web pages using MultiOn.

    Provides functionality for extracting structured data from web pages
    with configurable options for crawling and data extraction.
    Can be called directly with a task string.

    Example:
        >>> retriever = MultiOnRetrieverAgent("YOUR_API_KEY")
        >>> data = retriever.run("Get all product prices from example.com")
        # or
        >>> data = retriever("Get all product prices from example.com")
    """

    def __init__(
        self,
        api_key: str,
        name: str = "multion-retriever-agent-01",
        description: str = "An agent for retrieving structured data from web pages using MultiOn",
        system_prompt: str = None,
    ):
        """
        Initialize MultiOn Retriever Agent.

        Args:
            api_key (str): MultiOn API authentication key
        """
        self.client = MultiOn(api_key=api_key)
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        logger.info("MultiOnRetrieverAgent initialized")

    def __call__(self, task: str, **kwargs) -> RetrieveResponse:
        """
        Execute a retrieval task by calling the agent directly.

        Args:
            task (str): Task description
            **kwargs: Additional arguments to pass to run()

        Returns:
            RetrieveResponse: Response containing retrieved data
        """
        return self.run(task, **kwargs)

    @handle_errors
    def run(
        self,
        task: str,
        url: Optional[str] = None,
        fields: Optional[List[str]] = None,
        local: bool = False,
        max_items: Optional[int] = None,
        full_page: bool = True,
        render_js: bool = False,
        scroll_to_bottom: bool = False,
        include_screenshot: bool = False,
        use_proxy: bool = False,
    ) -> RetrieveResponse:
        """
        Execute a data retrieval task.

        Args:
            task (str): Task description
            url (Optional[str]): URL to retrieve from. If None, extracted from task
            fields (Optional[List[str]]): Fields to extract. If None, auto-detected
            local (bool): Whether to run in local mode
            max_items (Optional[int]): Maximum number of items to retrieve
            full_page (bool): Whether to crawl full page or viewport only
            render_js (bool): Whether to render JS elements
            scroll_to_bottom (bool): Whether to scroll page before retrieving
            include_screenshot (bool): Whether to include screenshot
            use_proxy (bool): Whether to use proxy

        Returns:
            RetrieveResponse: Response containing retrieved data
        """
        logger.info(f"Executing retrieval task: {task}")

        # Extract URL from task if not provided
        if not url:
            if "http" in task:
                url = task.split("http")[1].split(" ")[0]
                url = "http" + url
            else:
                raise ValueError(
                    "URL must be provided or included in task description"
                )

        # Auto-detect fields from task if not provided
        if not fields:
            # Simple field extraction - could be made more sophisticated
            fields = [
                word.strip(",.")
                for word in task.lower().split()
                if word.strip(",.")
                in ["title", "price", "description", "url", "date"]
            ]
            if not fields:
                fields = [
                    "text"
                ]  # Default to extracting text if no fields detected

        try:
            response = self.client.retrieve(
                cmd=task,
                url=url,
                fields=fields,
                local=local,
                max_items=max_items,
                full_page=full_page,
                render_js=render_js,
                scroll_to_bottom=scroll_to_bottom,
                include_screenshot=include_screenshot,
            )

            logger.success(
                f"Successfully retrieved {len(response.data)} items"
            )
            return RetrieveResponse(
                session_id=response.session_id,
                data=response.data,
                screenshot=response.screenshot,
            )

        except Exception as e:
            logger.error(f"Failed to retrieve data: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize agents
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Get API key from environment variable
    api_key = os.getenv("MULTION_API_KEY")

    if not api_key:
        raise ValueError("MULTION_API_KEY not found in environment variables")

    agent = MultiOnAgent(api_key=api_key)
    retriever = MultiOnRetrieverAgent(api_key=api_key)

    try:
        # Using run() method
        # response1 = agent.run(
        #     "Navigate to https://news.ycombinator.com and find the top post",
        #     local=True,
        # )

        # # Using direct call
        # response2 = agent(
        #     "Go to https://news.ycombinator.com and find the top post"
        # )

        # Using retriever with run()
        data1 = retriever.run(
            "Get all posts from https://news.ycombinator.com",
            fields=["title", "points"],
            max_items=10,
        )

        # Using retriever with direct call
        data2 = retriever(
            "Extract all post titles from https://news.ycombinator.com"
        )

        print(json.dumps(data2.data, indent=2))

    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
