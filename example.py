from agentverse.main import OpenInterpreterAgent

ai_architect = OpenInterpreterAgent(
    name="AI-Architect",
    auto_run=True,
    system_prompt="You are an expert AI architect specializing in creating innovative and efficient neural network architectures. Your task is to design and implement novel time series models using PyTorch, combining advanced techniques like transformers and LSTMs. Always strive for originality, efficiency, and state-of-the-art performance in your designs.",
)

print(
    ai_architect.run(
        "Design and implement a new, cutting-edge time series model architecture using transformers, LSTMs, and any other relevant techniques in PyTorch. Create a new project folder and write the complete Python code for the model, including any necessary utility functions or classes."
    )
)
