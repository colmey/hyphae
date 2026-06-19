"""Concrete LLMClient implementations, one file per provider.

Each module here subclasses `llm.client.LLMClient` and owns all the
SDK-specific translation for one provider (request shaping, response parsing,
transient-error classification). They are imported *lazily* by the provider
registry in `llm/client.py` (`_PROVIDERS`), so importing the LLM layer's
abstraction never drags in any provider SDK.

Adding a provider is two steps:
  1. Drop a `<name>.py` here implementing the LLMClient ABC.
  2. Register a builder for it in `llm/client.py`'s `_PROVIDERS`.

Nothing else in the harness enumerates providers.
"""
