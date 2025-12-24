# Removed deprecated imports

from langchain_core.language_models import LLM
from typing import Optional, List, Any
from app.core.llm_engine import llm_engine
from app.core.global_lock import gpu_lock
from app.agent.tools import ALL_TOOLS
import logging
import asyncio

logger = logging.getLogger(__name__)

# Adapt our llm_engine to LangChain's LLM interface
class LocalLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = llm_engine.get_model()
        
        logger.info("LocalLLM: Acquiring GPU Lock...")
        
        # Helper to run async lock and inference synchronously
        def run_inference():
            async def _async_infer():
                async with gpu_lock.request_priority_access():
                    # Phi-3 Instruct Format
                    # <|user|>\n...<|end|>\n<|assistant|>
                    
                    # Ensure prompt is wrapped if it's not already
                    if "<|user|>" not in prompt:
                        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
                    else:
                        formatted_prompt = prompt

                    output = await asyncio.to_thread(
                        model.create_completion,
                        prompt=formatted_prompt,
                        max_tokens=2048, # Increased for detailed answers
                        stop=stop or ["observation:", "<|end|>", "Observation:"], 
                        temperature=0.1, # Lower temperature for reasoning
                        repeat_penalty=1.1, # Prevent loops
                        top_p=0.9
                    )
                    return output["choices"][0]["text"]
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            if loop.is_running():
                import threading
                result = []
                def target():
                    l = asyncio.new_event_loop()
                    asyncio.set_event_loop(l)
                    result.append(l.run_until_complete(_async_infer()))
                    l.close()
                t = threading.Thread(target=target)
                t.start()
                t.join()
                return result[0]
            else:
                return loop.run_until_complete(_async_infer())

        return run_inference()

    @property
    def _llm_type(self) -> str:
        return "local-llama-cpp"

# Optimized ReAct Prompt for functionality
REACT_PROMPT = """You are Artemis, a helpful AI agent.
Solve the user's request using the available tools.

TOOLS:
------
{tools}

CONSTRAINTS:
------------
1. Do NOT use placeholders (e.g., [insert value]). If you cannot find info, say "I could not find the information".
2. Round long numbers to 1-2 decimal places (e.g., 23.45 instead of 23.45678).
3. Be concise.

FORMAT:
-------
To use a tool, use the following format:

thought: Think about what to do next
action: the action to take, should be one of [{tool_names}]
action_input: the input to the action
observation: the result of the action

... (this thought/action/action_input/observation can repeat N times)

When you have the final answer:
thought: I know the final answer
final_answer: the final answer to the original input question

USER REQUEST:
-------------
{input}

HISTORY:
--------
{agent_scratchpad}
"""

import re

class AgentExecutorService:
    def parse_action(self, llm_output: str):
        # Case insensitive parsing
        llm_output = llm_output.strip()
        
        # Check for Final Answer
        # Matches: final_answer: ... or Final Answer: ...
        final_answer_match = re.search(r"(?:final_answer|Final Answer):\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
             return "Final Answer", final_answer_match.group(1).strip()

        # Look for action: ... action_input: ...
        # Matches action: or Action:
        action_pattern = r"(?:action|Action):\s*(.*?)\n(?:action_input|Action Input):\s*(.*)"
        match = re.search(action_pattern, llm_output, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return None, None
            
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        return action, action_input

    def run(self, goal: str) -> str:
        llm = LocalLLM()
        tools_map = {t.name: t for t in ALL_TOOLS}
        tool_names = ", ".join(tools_map.keys())
        tools_desc = "\n".join([f"{t.name}: {t.description}" for t in ALL_TOOLS])
        
        scratchpad = ""
        max_steps = 10
        
        print(f"Starting Agent with Goal: {goal}")
        
        for step in range(max_steps):
            # Format prompt
            prompt = REACT_PROMPT.format(
                tools=tools_desc,
                tool_names=tool_names,
                input=goal,
                agent_scratchpad=scratchpad
            )
            
            # Call LLM
            logger.info(f"Step {step+1}: Calling LLM...")
            try:
                output = llm._call(prompt)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return f"Error calling LLM: {e}"
                
            logger.info(f"LLM Output: {output}")
            
            # Parse output
            # We strip trailing newlines and observations if the model hallucinated them
            # Split by observation: or Observation:
            if "observation:" in output.lower():
                output = re.split(r"observation:", output, flags=re.IGNORECASE)[0].strip()
            
            # Append to scratchpad
            scratchpad += output + "\n"
            
            action, action_input = self.parse_action(output)
            
            if action == "Final Answer":
                return action_input
            
            # Normalize action name to match tools dict keys (usually Title Case)
            # But the model might output 'search' vs 'Search'
            matched_tool_name = None
            if action:
                for tool_name in tools_map.keys():
                    if action.lower() == tool_name.lower():
                        matched_tool_name = tool_name
                        break
            
            if matched_tool_name:
                logger.info(f"Executing Tool: {matched_tool_name} with Input: {action_input}")
                tool = tools_map[matched_tool_name]
                try:
                    # Tools in ALL_TOOLS are LangChain tools, so they have .run method
                    observation = tool.run(action_input)
                except Exception as e:
                    observation = f"Error executing tool: {e}"
                
                logger.info(f"Observation: {observation}")
                scratchpad += f"observation: {observation}\nthought:"
            else:
                # If no action found, or invalid action
                if "final_answer:" in output.lower():
                     return re.split(r"final_answer:", output, flags=re.IGNORECASE)[-1].strip()
                     
                logger.warning("No Action found in output.")
                if step == 0 and len(output) > 20: # Heuristic: if it talks a lot, maybe it answered?
                    return output
                
                # If it's just a thought, continue
                if "thought:" in output.lower() and "action:" not in output.lower():
                    scratchpad += "observation: Please continue to action.\n"
                    continue
                    
                return output # Fallback

        return "Agent stopped: Max steps reached."

agent_executor = AgentExecutorService()
