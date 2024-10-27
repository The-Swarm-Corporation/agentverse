from typing import List
from dataclasses import dataclass


@dataclass
class Message:
    role: str
    type: str
    content: str
    format: str = None


class ConversationParser:
    @staticmethod
    def extract_conversation(text: str) -> List[Message]:
        """Extract conversation messages from the text."""
        # Find all dictionary-like strings
        messages = []
        lines = text.split("\n")

        current_dict_str = ""
        in_dict = False

        for line in lines:
            if line.startswith("{"):
                in_dict = True
                current_dict_str = line
            elif in_dict and "}" in line:
                current_dict_str += line
                try:
                    message_dict = eval(current_dict_str)
                    messages.append(
                        Message(
                            role=message_dict["role"],
                            type=message_dict["type"],
                            content=message_dict.get("content", ""),
                            format=message_dict.get("format"),
                        )
                    )
                except:
                    pass  # Skip invalid dictionary strings
                in_dict = False
                current_dict_str = ""
            elif in_dict:
                current_dict_str += line

        return messages

    @staticmethod
    def format_conversation(messages: List[Message]) -> str:
        """Format the conversation into a readable string."""
        formatted_conversation = []

        for msg in messages:
            # Create header with role and type
            header = f"[{msg.role.upper()}] ({msg.type})"
            if msg.format:
                header += f" <{msg.format}>"

            # Format content based on type
            if msg.type == "code":
                content = f"\n```{msg.format}\n{msg.content}\n```"
            else:
                content = f"\n{msg.content}"

            formatted_conversation.append(f"{header}{content}\n")

        return "\n".join(formatted_conversation)

    @staticmethod
    def parse_and_format(text: str) -> str:
        """Parse and format the conversation in one step."""
        messages = ConversationParser.extract_conversation(text)
        return ConversationParser.format_conversation(messages)


# Example usage
def parse_conversation(text: str) -> str:
    """Wrapper function to parse and format a conversation."""
    parser = ConversationParser()
    return parser.parse_and_format(text)
