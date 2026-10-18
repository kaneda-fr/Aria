"""Prime the operational prompt without admitting a conversation or executing tools."""
from dataclasses import replace
import time

from app.llm.ollama_client import generate_llm_response


async def warmup(config, client, system_prompt, tools, *, with_tools=True):
    started = time.monotonic()
    text = 'Say exactly: Aria is ready.'
    # Let the short reply finish: Ministral/Ollama 0.14.2 leaked a buffered "A"
    # into the next answer when this warm-up was truncated at one token.
    # Never dispatch calls, speak, or save this exchange in a user's history.
    result = await generate_llm_response(
        text=text,
        messages=[{'role': 'system', 'content': system_prompt},
                  {'role': 'user', 'content': '[speaker=unknown] ' + text}],
        config=replace(config, num_predict=8, stream=False) if with_tools else config,
        client=client, tools=tools if with_tools else None)
    return {'tools': with_tools, 'seconds': time.monotonic() - started,
            'discarded_tool_calls': len(result.get('tool_calls') or []),
            'discarded_content_chars': len(result.get('content') or '')}
