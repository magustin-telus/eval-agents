# Complaint Resolution Quality Rubric

You are evaluating a customer complaint **resolution** produced by an AI agent. The agent
was given a complaint narrative and the bank policy it retrieved, and asked to recommend a
resolution grounded in that policy.

> Note on the input: complaint narratives are pre-processed (lemmatized, stop-words removed,
> no punctuation), so they read as keyword sequences rather than fluent prose. Do **not**
> penalize the resolution for the garbled phrasing of the input. Judge the resolution on the
> dimensions below, interpreting the complaint by topic.

## Metrics

Emit exactly the following four metrics, each with a binary value (0 or 1) and a one-sentence comment.

1. **groundedness**
   - Value `1` only if every factual claim, promise, timeframe, and step in the resolution is
     supported by the retrieved policy text (the Expected Output).
   - Value `0` if the resolution invents remedies, timeframes, or guarantees not in the policy,
     or contradicts the policy.

2. **completeness**
   - Value `1` only if the resolution addresses the core issue raised in the complaint AND gives
     a concrete next step or resolution (not just an acknowledgement).
   - Value `0` otherwise.

3. **tone**
   - Value `1` only if the resolution is professional, empathetic, and does not blame the customer.
   - Value `0` if it is dismissive, accusatory, or unprofessional.

4. **actionability**
   - Value `1` only if the resolution gives the customer (or the agent) a concrete action or a
     correct escalation path consistent with the policy.
   - Value `0` if it is vague or offers no usable next step.

## Scoring instructions

- Use binary values only: `0` or `1`.
- Judge groundedness strictly against the Expected Output (the retrieved policy text). If the
  policy does not cover something the resolution promises, groundedness is `0`.
- Keep each metric comment to one sentence, citing the specific part of the resolution that
  drove the score.
