"""Concrete LLMClient implementations, one module or package per provider.

Each provider here subclasses `hyphae.llm.client.LLMClient` and owns its SDK lifetime
and wire translation. Focused providers may use a package to separate client,
codec, and stream responsibilities. Providers are imported *lazily* by the
registry in `hyphae/llm/client.py` (`_PROVIDERS`), so importing the LLM abstraction
never drags in a provider SDK.

Adding a provider is two steps:
  1. Add a `<name>.py` module or `<name>/` package implementing the LLMClient ABC.
  2. Register a builder for it in `hyphae/llm/client.py`'s `_PROVIDERS`.

Nothing else in the harness enumerates providers.
"""
