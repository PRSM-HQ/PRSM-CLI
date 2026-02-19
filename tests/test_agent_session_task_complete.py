
import asyncio
import sys
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from prsm.engine.agent_session import AgentSession
from prsm.engine.models import AgentDescriptor, AgentRole, AgentState

# Mock message classes to mimic claude_agent_sdk structures
@dataclass
class MockContentBlock:
    type: str
    text: str | None = None
    tool_use_id: str | None = None
    content: str | None = None
    is_error: bool = False
    name: str | None = None
    input: dict | None = None
    id: str | None = None

@dataclass
class MockMessage:
    content: list[MockContentBlock]
    result: str | None = None

class TestAgentSessionTaskComplete(unittest.IsolatedAsyncioTestCase):
    async def test_stops_on_task_complete(self):
        # Mocks
        manager = MagicMock()
        router = MagicMock()
        expert_registry = MagicMock()
        
        descriptor = AgentDescriptor(
            agent_id="test-agent",
            role=AgentRole.MASTER,
            state=AgentState.PENDING,
            model="claude-test",
            tools=[]
        )
        
        session = AgentSession(
            descriptor=descriptor,
            manager=manager,
            router=router,
            expert_registry=expert_registry,
            agent_timeout_seconds=10.0
        )
        
        async def custom_query(prompt, options):
            # 1. Yield text
            yield MockMessage(content=[MockContentBlock(type="text", text="Start.")])
            
            # 2. Simulate task_complete tool execution side effect
            descriptor.result_summary = "Task Done"
            descriptor.result_artifacts = {"verification_results": [{"passed": True}]}
            
            # 3. Yield tool result
            yield MockMessage(content=[
                MockContentBlock(
                    type="tool_result",
                    tool_use_id="call_1",
                    content="Task marked complete.",
                    is_error=False
                )
            ])
            
            # 4. Yield post-completion text
            yield MockMessage(content=[MockContentBlock(type="text", text="Babble.")])

        # Create a mock module for claude_agent_sdk
        mock_sdk = MagicMock()
        mock_sdk.query = custom_query
        mock_sdk.ClaudeAgentOptions = MagicMock()
        
        # Patch sys.modules to inject our mock sdk
        with patch.dict(sys.modules, {"claude_agent_sdk": mock_sdk}):
             with patch("prsm.engine.mcp_server.server.build_agent_mcp_config") as mock_build:
                mock_tools = MagicMock()
                mock_tools.time_tracker = MagicMock()
                mock_tools.time_tracker.accumulated_tool_time = 0
                mock_build.return_value = ({}, mock_tools)
                
                # Run session
                result = await session.run()
                
                # Assertions
                self.assertTrue(result.success)
                self.assertEqual(result.summary, "Task Done")
                self.assertEqual(
                    result.artifacts,
                    {"verification_results": [{"passed": True}]},
                )
                
                # Verify we didn't process the "Babble." text
                # We can check _accumulated_text
                self.assertIn("Start.", session._accumulated_text)
                self.assertNotIn("Babble.", session._accumulated_text)

if __name__ == "__main__":
    unittest.main()
