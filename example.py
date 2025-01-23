from agentverse.open_interpreter import OpenInterpreterAgent
from loguru import logger

# Create screen control agent
screen_controller = OpenInterpreterAgent(
    name="ScreenController",
    auto_run=True,
    system_prompt="""You are an expert in computer vision and screen automation.
    You can:
    - Capture and analyze screen content
    - Control mouse movements and clicks
    - Send keyboard inputs
    - Detect and interact with UI elements
    - Process visual information in real-time
    
    Use computer vision and automation to interact with on-screen elements accurately and reliably.
    """,
)

# Run the screen controller
controller_output = screen_controller.run(
    """
Create a Python program that:
1. Captures the current screen state using screenshot
2. Processes the image to:
   - Detect UI elements and controls
   - Identify text and clickable areas
   - Track cursor position
   - Monitor screen changes
3. Implements screen interaction functions:
   - Mouse movement and clicking
   - Keyboard input
   - Window management
   - Element detection
4. Provides real-time visual feedback
5. Maintains a log of:
   - Screen states
   - User interactions
   - System events
   - Performance metrics
"""
)

print("Screen Controller Status:")
print(controller_output)

logger.info("Saving screen_controller.json")
screen_controller.save("screen_controller.json")
# logger.info("Loading screen_controller.json")
# screen_controller.load("screen_controller.json")
