import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from vector_store import VectorMemory

# Load environment variables
load_dotenv()


class AegisAgent:
    def __init__(self):
        """Initializes the Agent, its Memory, and its Tools."""
        print("Initializing Aegis Agent & Vector Memory...")
        self.memory = VectorMemory()

        # Define the tools available to the agent
        self.tools = {
            "vector_search": self._tool_vector_search,
            "python_calculator": self._tool_calculator
        }

    def bootstrap_dummy_data(self):
        """Populates the agent's memory with initial dummy data after models are loaded."""
        print("Bootstrapping Agent with initial knowledge base...")
        self.memory.add_texts(
            [
                "The secret access code for the Aegis mainframe is 7734-X.",
                "Project Spark refers to the 124M parameter NanoTransformer model.",
                "The LexiByte tokenizer uses Sennrich frequency optimization."
            ],
            [
                {"filename": "aegis_core_manual.txt"},
                {"filename": "project_spark_spec.pdf"},
                {"filename": "lexibyte_docs.pdf"}
            ]
        )

    # --- Tool Implementations ---

    def _tool_vector_search(self, query: str) -> str:
        """Searches the ONNX/FAISS database for context and includes the source document name."""
        print(f"  [Tool Execution] vector_search('{query}')")
        results = self.memory.search(query, top_k=3)
        if not results:
            return "No relevant information found in the database."
        return "\n\n".join(results)

    def _tool_calculator(self, expression: str) -> str:
        """Safely evaluates basic mathematical expressions."""
        print(f"  [Tool Execution] python_calculator('{expression}')")
        try:
            # Restricted eval for basic math (prevents arbitrary code execution)
            allowed_names = {"abs": abs, "round": round, "max": max, "min": min}
            code = compile(expression, "<string>", "eval")
            for name in code.co_names:
                if name not in allowed_names:
                    raise NameError(f"Use of {name} not allowed")
            result = eval(code, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error evaluating math: {e}"

    # --- LLM Interface ---

    def _call_llm(self, prompt: str, api_key: str) -> str:
        """Sends a prompt to the Gemini LLM using the modern google.genai SDK."""
        if not api_key:
            return "Final Answer: Error: A Gemini API Key is required. Please provide it in the UI."

        try:
            # Initialize the new SDK Client
            client = genai.Client(api_key=api_key)

            # Dynamically fetch the model from .env, fallback to gemini-2.5-flash
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

            # Generate content using the new SDK structure
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    # CRITICAL: Stop generating so the Python loop can inject the tool observation!
                    stop_sequences=["\nObservation:"]
                )
            )
            return response.text
        except Exception as e:
            # If Google API fails, break the loop cleanly by returning a Final Answer
            return f"Final Answer: Gemini API Error: {str(e)}"

    # --- The ReAct (Reason + Act) Loop ---

    def run(self, question: str, max_iterations: int = 5, api_key: str = "") -> str:
        """The core ReAct loop that orchestrates the LLM and Tools."""

        system_prompt = """You are Aegis, a hardcore systems engineering AI Agent.
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output a Final Answer.

You have access to the following tools:
1. vector_search: Searches the internal document database. Use this to find knowledge. Input should be a search query.
2. python_calculator: Evaluates math expressions. Input should be a valid python math string (e.g. 10 * 2.5).

Use the following exact format:
Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, must be exactly one of [vector_search, python_calculator]
Action Input: the input to the action
PAUSE

(Once you output PAUSE, I will return an Observation to you.)
Observation: the result of the action

... (this Thought/Action/Action Input/PAUSE/Observation can repeat N times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT CITATION RULE:
When you output your Final Answer, if you used information from the vector_search tool, you MUST cite the source document name in your answer. (e.g. "According to [Source: manual.pdf]...")

Begin!
"""

        prompt = system_prompt + f"\nQuestion: {question}\n"
        print(f"\n[User Question]: {question}\n" + "-" * 50)

        for i in range(max_iterations):
            # 1. Ask the LLM what to do
            response_text = self._call_llm(prompt, api_key)
            print(f"{response_text}")
            prompt += response_text

            # 2. Check if the LLM reached a Final Answer
            final_answer_match = re.search(r"Final Answer:(.*)", response_text, re.DOTALL)
            if final_answer_match:
                return final_answer_match.group(1).strip()

            # 3. Parse the Action and Action Input using Regex
            action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)

            if not action_match:
                # LLM broke format, gently push it back on track
                obs = "Observation: Formatting error. You must provide an 'Action' and 'Action Input' or a 'Final Answer'."
                prompt += f"\n{obs}\n"
                print(f"\n{obs}")
                continue

            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip()

            # 4. Execute the chosen tool
            obs = ""
            if action in self.tools:
                tool_result = self.tools[action](action_input)
                obs = f"Observation: {tool_result}"
            else:
                obs = f"Observation: Unknown tool '{action}'. Valid tools are {list(self.tools.keys())}."

            # 5. Append observation to the context and loop
            print(f"\n{obs}\n" + "-" * 50)
            prompt += f"\n{obs}\n"

        return "Error: Maximum iterations reached without a Final Answer."


# --- Quick Local Test ---
if __name__ == "__main__":
    agent = AegisAgent()

    # Needs to be called explicitly here since it was moved out of __init__
    agent.bootstrap_dummy_data()

    # Note: Requires a valid GEMINI_API_KEY environment variable or hardcoded for local CLI test.
    test_key = os.getenv("GEMINI_API_KEY", "")

    test_question = ("What is the secret access code for the Aegis mainframe? Once you find it, extract the numerical "
                     "part and multiply it by 10.")

    final_answer = agent.run(test_question, api_key=test_key)

    print("\n" + "=" * 50)
    print(f"🎯 FINAL ANSWER: {final_answer}")
    print("=" * 50 + "\n")
