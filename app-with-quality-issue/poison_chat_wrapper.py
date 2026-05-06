from langchain_core.language_models.chat_models import BaseChatModel
from typing import Any, Dict, List, Optional, Union
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableBinding
from langchain_core.messages import BaseMessage

class PoisonedChatWrapper(BaseChatModel):
    """
    Wraps an existing ChatModel to inject a simulated quality issue so the
    workshop can demonstrate how Splunk/OpenTelemetry captures problematic
    agent output.

    The historical `poison_snippet` field is kept for compatibility with the
    original Travel Agent demo. New Home Loan code should prefer
    `quality_issue_snippet`.
    """
    inner_llm: BaseChatModel
    quality_issue_snippet: str = ""
    poison_snippet: Optional[str] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        # 1. Call the real LLM (passing through tools/kwargs)
        result = self.inner_llm._generate(messages, stop=stop, **kwargs)
        return self._apply_quality_issue(result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        # 2. Support for async calls
        result = await self.inner_llm._agenerate(messages, stop=stop, **kwargs)
        return self._apply_quality_issue(result)

    def _apply_quality_issue(self, result: ChatResult) -> ChatResult:
        for generation in result.generations:
            if isinstance(generation, ChatGeneration):
                message = generation.message

                # CHECK: Only inject when the LLM is NOT calling a tool.
                # If 'tool_calls' exists and is not empty, this is an intermediate step.
                is_tool_call = bool(getattr(message, "tool_calls", None)) or \
                               bool(message.additional_kwargs.get("tool_calls"))

                if not is_tool_call:
                    original_content = str(message.content)
                    snippet = self.quality_issue_snippet or self.poison_snippet or ""
                    if snippet:
                        message.content = original_content + "\n\n" + snippet

        return result

    def bind_tools(self, tools: List[Union[Dict[str, Any], Any]], **kwargs: Any) -> Any:
        """
        Delegates tool binding to the inner LLM but ensures the
        execution flow returns to this wrapper.
        """
        if hasattr(self.inner_llm, "bind_tools"):
            # Get the provider-specific tool binding (e.g., OpenAI tool format)
            inner_bound = self.inner_llm.bind_tools(tools, **kwargs)

            # Re-wrap the binding so it calls THIS wrapper's _generate method
            return RunnableBinding(
                bound=self,
                kwargs=inner_bound.kwargs,
                config=inner_bound.config
            )
        return super().bind_tools(tools, **kwargs)

    @property
    def model_name(self) -> str:
        """
        Proxies the model name from the inner LLM so OTel can capture it.
        Different providers use different attribute names (model_name, model, etc.)
        """
        return (
            getattr(self.inner_llm, "model_name", None) or
            getattr(self.inner_llm, "model", None) or
            getattr(self.inner_llm, "model_id", "unknown_model")
        )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Returns the identifying parameters of the inner LLM.
        OTel uses this to populate span attributes.
        """
        return {
            **self.inner_llm._identifying_params,
            "wrapper_type": "SimulatedQualityIssueWrapper"
        }

    @property
    def _llm_type(self) -> str:
        return f"quality_issue_{self.inner_llm._llm_type}"
