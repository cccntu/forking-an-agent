import copy
import asyncio
import json
from typing import Dict, Any


from calculator_agent import CalculatorAgent


fork_tool_description = """
You can use this tool to fork the conversation into multiple instances of yourself and let each instance continue answering and using tools.
analogous to the fork() system call in Unix.
pid = fork([PROMPT]) # prompt to yourself to continue the conversation)

if pid == 0:
    # child process
    You'll receive PROMPT as the tool_result and continue the conversation
else:
    # parent process
    You'll wait for all children to finish and receive the final message from each instance in the following format
    [
        {
            "id": 0,
            "message": "the final message from the child process"
        },
    ]

Different from Unix, you can fork multiple children at once.
fork([PROMPT0, PROMPT1, PROMPT2, ...])

When to use this tool:
You can fork yourself to do task that would otherwise fill up the context length but only the final result matters.
For example, if you need to read a large file to find certain detail, or if you need to execute multiple tools step by step but you don't need the intermediate results.

You can fork multiple instances to perform tasks in parallel without performing them in serial which would also quickly fill up the context length.
"""


class CalculatorForkAgent(CalculatorAgent):
    """
    A simplified agent that only supports basic arithmetic operations.
    Provides async process_query and spawn methods.
    """

    global_pid = 1

    def __init__(self, model="claude-3-7-sonnet-20250219"):
        super().__init__()  # register calculator tools and initialize conversation history
        self.model = model

        self.allow_fork = True
        # NOTE: we don't remove the tool from definition, this is to avoid invalidating the prompt cache
        # also, the tool description is important context for the child process as well.

        # for debugging purposes
        self.pid = CalculatorForkAgent.global_pid
        CalculatorForkAgent.global_pid += 1

        self.register_tool(
            name="fork",
            handler=self._fork,
            description=fork_tool_description,
            parameters={
                "prompts": {
                    "type": "array",
                    "description": "List of prompts to continue the conversation with",
                    "items": {
                        "type": "string",
                        "description": "A prompt to continue the conversation",
                    },
                }
            },
            required_params=["prompts"],
        )

    def deepcopy(self):
        """returns a copy of the current instance with copied conversation history"""
        # NOTE: copy.deepcopy(self) is not working as expected, so we create a new instance instead
        # it's due to issues like the Anthropic client and its locks
        child = CalculatorForkAgent()  # Create new instance instead of deepcopy
        child.allow_fork = False
        # Only copy the conversation history
        child.conversation = copy.deepcopy(self.conversation)
        return child

    async def _fork(self, params, tool_id, last_assistant_response):
        """Fork a conversation"""
        if not self.allow_fork:
            return "Forking is not allowed for this agent, possible reason: You are already a forked instance"

        prompts = params["prompts"]
        print(f"Forking conversation with {len(prompts)} prompts: {prompts}")

        async def process_fork(i, prompt):
            child = self.deepcopy()
            # print(f'Forking conversation with prompt: {prompt}')

            # NOTE: we need to remove all other tool calls from the last assistant response
            # because we might not have the tool call results for other tool calls yet
            child.conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        content
                        for content in last_assistant_response
                        if content.type != "tool_use"
                        or (content.type == "tool_use" and content.id == tool_id)
                    ],
                }
            )
            # NOTE: return the fork result as tool result
            child.conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "pid==0, you are a child instance produced from a fork. you are not allowed to use the fork tool. please continue the conversation with only the assigned goal",
                        }
                    ],
                }
            )
            # NOTE: immediately add the prompt to the conversation as user message
            # I found this to work better than adding the prompt as the tool result
            child.conversation.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            response = await child.process_query(
                query=prompt, max_iterations=20, is_tool_continuation=False
            )
            print(f"{self.pid=}: Forked response from {child.pid=}: {response}")
            return {"id": i, "message": response}

        # Process all forks in parallel
        responses = await asyncio.gather(
            *[process_fork(i, prompt) for i, prompt in enumerate(prompts)]
        )
        result = json.dumps(responses)
        return result

    async def process_query(
        self, query: str, max_iterations: int = 20, is_tool_continuation: bool = False
    ):
        print(f"{self.pid=}: Processing query ...")
        """Process a query using Claude with tools

        Args:
            query: The user's query
            max_iterations: Maximum number of tool-calling iterations to allow

        Returns:
            The final agent response text
        """
        if is_tool_continuation:
            pass
        else:
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
                model=self.model,
                max_tokens=1000,
                messages=self.conversation,
                tools=self.tools,
            )

            # Process the response
            has_tool_calls = False
            tool_results = []
            response_text = ""

            num_tool_calls = len(
                [content for content in response.content if content.type == "tool_use"]
            )
            print(f"{self.pid=}: Tool calls: {num_tool_calls=}")

            # Process text content and tool calls
            for content in response.content:
                if content.type == "text":
                    response_text = content.text
                    final_response = response_text  # Save the text response
                    print(f"{self.pid=}: Text response: {response_text}")
                elif content.type == "tool_use":
                    has_tool_calls = True
                    tool_name = content.name
                    tool_args = content.input
                    tool_id = content.id

                    # Execute the tool
                    print(
                        f"{self.pid=}: Calling tool {tool_name} with args {tool_args} and tool_id {tool_id}"
                    )
                    if tool_name == "fork":
                        print(f"{self.pid=}: Forking with tool_args: {tool_args}")
                        result = await self._fork(
                            tool_args, tool_id, last_assistant_response=response.content
                        )
                    else:
                        result = await self.call_tool(tool_name, tool_args, tool_id)
                    print(f"{self.pid=}: Tool {tool_name} returned {result}")

                    # Add to tool results
                    tool_results.append({"tool_use_id": tool_id, "content": result})

            # If no tool calls were made, we're done with this query
            if not has_tool_calls:
                break

            self.conversation.append({"role": "assistant", "content": response.content})
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

    async def call_tool(
        self, tool_name: str, tool_args: Dict[str, Any], tool_id: str
    ) -> str:
        """Call a registered tool with the given parameters"""
        try:
            if tool_name not in self.tool_handlers:
                return f"Error: Tool '{tool_name}' not found"

            handler = self.tool_handlers[tool_name]
            result = handler(tool_args, tool_id)

            # If the result is a coroutine (async function), await it
            if asyncio.iscoroutine(result):
                result = await result

            return result
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"


async def test_basic_calculation():
    agent = CalculatorForkAgent()
    response = await agent.process_query("What is 2 + 2?")
    print("--- basic calculation test finished ---")


async def test_tools_list():
    agent = CalculatorForkAgent()
    response = await agent.process_query("what tools do you have?")
    print("--- tools list test finished ---")


async def test_fork():
    agent = CalculatorForkAgent()
    response = await agent.process_query(
        "fork yourself to calculate 1234231*1234231 and 12345/451341. show me the results. You don't need to verify the results."
    )
    print("--- fork test finished ---")


# Dictionary mapping test numbers to test functions
TESTS = {
    1: test_basic_calculation,
    2: test_tools_list,
    3: test_fork,
}


async def run_all_tests():
    for test_func in TESTS.values():
        await test_func()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run calculator fork agent tests")
    parser.add_argument(
        "test_number",
        type=int,
        nargs="?",
        help="Test number to run (1-4). If not provided, runs all tests.",
    )
    args = parser.parse_args()

    if args.test_number is not None:
        if args.test_number not in TESTS:
            print(
                f"Error: Invalid test number. Please choose from {list(TESTS.keys())}"
            )
            print("\nAvailable tests:")
            for num, func in TESTS.items():
                print(f"{num}: {func.__name__}")
        else:
            asyncio.run(TESTS[args.test_number]())
    else:
        asyncio.run(run_all_tests())
