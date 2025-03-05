import os
from typing import Dict, Any, List, Callable

from anthropic import AsyncAnthropic


class CalculatorAgent:
    """
    A simplified agent that only supports basic arithmetic operations.
    Provides async process_query and spawn methods.
    """

    def __init__(self):
        # Initialize Async Anthropic client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = AsyncAnthropic(api_key=api_key)

        # Conversation history
        self.conversation = []

        # Tool registry
        self.tools = []
        self.tool_handlers = {}

        # Register calculator tools
        self.register_calculator_tools()

    def register_calculator_tools(self):
        """Register only the basic arithmetic tools"""
        # Addition
        self.register_tool(
            name="add",
            description="Add two numbers together.",
            handler=self._add,
            parameters={
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            required_params=["a", "b"],
        )

        # Subtraction
        self.register_tool(
            name="subtract",
            description="Subtract one number from another.",
            handler=self._subtract,
            parameters={
                "a": {"type": "number", "description": "Number to subtract from"},
                "b": {"type": "number", "description": "Number to subtract"},
            },
            required_params=["a", "b"],
        )

        # Multiplication
        self.register_tool(
            name="multiply",
            description="Multiply two numbers together.",
            handler=self._multiply,
            parameters={
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            required_params=["a", "b"],
        )

        # Division
        self.register_tool(
            name="divide",
            description="Divide one number by another.",
            handler=self._divide,
            parameters={
                "a": {"type": "number", "description": "Dividend"},
                "b": {"type": "number", "description": "Divisor (cannot be 0)"},
            },
            required_params=["a", "b"],
        )

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Dict[str, Dict[str, Any]] = None,
        required_params: List[str] = None,
    ):
        """Register a tool that Claude can use"""
        if parameters is None:
            parameters = {}

        if required_params is None:
            required_params = []

        tool_def = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": required_params,
            },
        }

        self.tools.append(tool_def)
        self.tool_handlers[name] = handler

    # Calculator tool implementations
    def _add(self, params, tool_id):
        """Add two numbers"""
        a = params.get("a", 0)
        b = params.get("b", 0)
        return str(a + b)

    def _subtract(self, params, tool_id):
        """Subtract one number from another"""
        a = params.get("a", 0)
        b = params.get("b", 0)
        return str(a - b)

    def _multiply(self, params, tool_id):
        """Multiply two numbers"""
        a = params.get("a", 0)
        b = params.get("b", 0)
        return str(a * b)

    def _divide(self, params, tool_id):
        """Divide one number by another"""
        a = params.get("a", 0)
        b = params.get("b", 1)
        if b == 0:
            return "Error: Cannot divide by zero"
        return str(a / b)

    async def call_tool(
        self, tool_name: str, tool_args: Dict[str, Any], tool_id: str
    ) -> str:
        """Call a registered tool with the given parameters"""
        try:
            if tool_name not in self.tool_handlers:
                return f"Error: Tool '{tool_name}' not found"

            handler = self.tool_handlers[tool_name]
            result = handler(tool_args, tool_id)
            return result
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"

    async def process_query(self, query: str, max_iterations: int = 20):
        """Process a query using Claude with tools

        Args:
            query: The user's query
            max_iterations: Maximum number of tool-calling iterations to allow

        Returns:
            The final agent response text
        """
        # Add user message to conversation history
        self.conversation.append({"role": "user", "content": query})

        # Track iteration count to prevent infinite loops
        iterations = 0
        final_response = ""

        # Continue the conversation until no more tool calls or max iterations reached
        while iterations < max_iterations:
            iterations += 1

            # Call Claude with current conversation
            response = await self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                messages=self.conversation,
                tools=self.tools,
            )

            # Process the response
            has_tool_calls = False
            tool_results = []
            response_text = ""

            # Add Claude's response to conversation
            self.conversation.append({"role": "assistant", "content": response.content})

            # Process text content and tool calls
            for content in response.content:
                if content.type == "text":
                    response_text = content.text
                    final_response = response_text  # Save the text response
                elif content.type == "tool_use":
                    has_tool_calls = True
                    tool_name = content.name
                    tool_args = content.input
                    tool_id = content.id

                    # Execute the tool
                    print(
                        f"Calling tool {tool_name} with args {tool_args} and tool_id {tool_id}"
                    )
                    result = await self.call_tool(tool_name, tool_args, tool_id)

                    # Add to tool results
                    tool_results.append({"tool_use_id": tool_id, "content": result})

            # If no tool calls were made, we're done with this query
            if not has_tool_calls:
                break

            # Add tool results to conversation
            for result in tool_results:
                self.conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": result["tool_use_id"],
                                "content": result["content"],
                            }
                        ],
                    }
                )

        return final_response
