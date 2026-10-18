SELF-LEARNING ROUTER RULES

Aria uses embeddings to classify each user utterance as:
- CONTROL (device action)
- CHAT (conversation)
- UNKNOWN (uncertain)

Learning happens DURING normal operation and MUST follow these rules:

1. NEVER learn from the router decision alone.
2. Learn ONLY from verified outcomes.

LABEL AS CONTROL when:
- A tool action (execute_command) was called
- AND the action succeeded OR was confirmed by observability events

LABEL AS CHAT when:
- No tool was called
- AND the user did NOT correct the response within a short time window

3. User feedback overrides everything:
- "yes / correct / parfait" → confirm previous label
- "no / wrong / not that" → mark previous label as incorrect
- "it was a question" → label CHAT
- "it was a command" → label CONTROL

4. When a label is confirmed:
- Store the utterance embedding in the corresponding learning bank
- Periodically update routing centroids or prototypes
- Bias routing toward CHAT if confidence is low

5. If confidence is low or signals conflict:
- DO NOT learn
- Route as UNKNOWN and ask for clarification

6. Keep the system safe:
- Prefer UNKNOWN over wrong CONTROL
- Maintain last known good router state
- Roll back if false CONTROL rate increases

Self-learning must improve routing accuracy over time without changing the embedding model.