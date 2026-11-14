"""Minimal Langfuse v4 instrumentation example.

Run after installing `langfuse` and setting LANGFUSE_* environment variables.
This example is intentionally standalone so it does not change upstream runtime behavior.
"""
import os
from langfuse import observe, get_client

@observe(name="sample-assistant-operation")
def sample_operation(question: str) -> dict:
    # Replace this stub with a call to your application's assistant service.
    result = {"question": question, "status": "instrumentation-example"}
    return result

if __name__ == "__main__":
    print(sample_operation("How does retrieval-augmented generation work?"))
    get_client().flush()
