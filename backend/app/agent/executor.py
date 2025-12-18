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
        
        # We need to run this with the lock
        # But this _call method is sync, expected by LangChain (if not using async LLM)
        # And our lock is async. 
        # So we use a threading approach similar to the tools, or we assume
        # this is running in a thread pool executor.
        
        # However, create_chat_completion is fairly stateless on the python side 
        # (mostly calling C++), but the lock is critical.
        
        logger.info("LocalLLM: Acquiring GPU Lock...")
        
        # Helper to run async lock and inference synchronously
        def run_inference():
            async def _async_infer():
                async with gpu_lock.request_priority_access():
                    # Simple completion for ReAct agent
                    # Phi-3 instruct format handling might be needed here
                    # For now, we pass the raw prompt as user message
                    # ReAct prompts are usually big blocks of text.
                    
                    # NOTE: Phi-3 is an Instruct model. It expects <|user|>...<|assistant|>.
                    # LangChain's ReAct agent generates a "Question: ... Thought: ..." string.
                    # We should wrap it in instruct tokens for better performance.
                    
                    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
                    
                    output = await asyncio.to_thread(
                        model.create_completion,
                        prompt=formatted_prompt,
                        max_tokens=612,
                        stop=stop or ["Observation:"], # ReAct stop token
                        temperature=0.7
                    )
                    return output["choices"][0]["text"]
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            if loop.is_running():
                # Nesting fix
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

# Standard ReAct prompt
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

import re

class AgentExecutorService:
    def parse_action(self, llm_output: str):
        # Look for Action: ... Action Input: ...
        # Regex covers multi-line action input if needed, but ReAct usually uses single line
        action_pattern = r"Action:\s*(.*?)\nAction Input:\s*(.*)"
        match = re.search(action_pattern, llm_output, re.DOTALL)
        
        if not match:
            # Check for Final Answer
            if "Final Answer:" in llm_output:
                return "Final Answer", llm_output.split("Final Answer:")[-1].strip()
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
            output = output.split("Observation:")[0].strip()
            
            # Append to scratchpad
            scratchpad += output + "\n"
            
            action, action_input = self.parse_action(output)
            
            if action == "Final Answer":
                return action_input
            
            if action and action in tools_map:
                logger.info(f"Executing Tool: {action} with Input: {action_input}")
                tool = tools_map[action]
                try:
                    # Tools in ALL_TOOLS are LangChain tools, so they have .run method
                    observation = tool.run(action_input)
                except Exception as e:
                    observation = f"Error executing tool: {e}"
                
                logger.info(f"Observation: {observation}")
                scratchpad += f"Observation: {observation}\nThought:"
            else:
                # If no action found, or invalid action
                if "Final Answer:" in output:
                     return output.split("Final Answer:")[-1].strip()
                     
                # Force a thought? Or stop?
                # If model didn't produce an action, maybe it's just thinking or failed format.
                # We can try to continue generating or stop.
                # For now, treat as Final Answer if it looks like text, or error.
                logger.warning("No Action found in output.")
                # Attempt to treat the whole thing as answer if it's the first step?
                # Or just return it.
                if step == 0:
                    return output
                return output # Fallback

        return "Agent stopped: Max steps reached."

agent_executor = AgentExecutorService()
