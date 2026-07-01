"""ACP → OpenAI gateway.

Expose any agent that speaks the Agent Client Protocol (ACP, JSON-RPC 2.0 over
streamable HTTP) behind an OpenAI-compatible ``/v1/chat/completions`` +
``/v1/models`` API, so OpenAI-native UIs and SDKs can drive it unchanged.
"""

__version__ = "0.1.0"
