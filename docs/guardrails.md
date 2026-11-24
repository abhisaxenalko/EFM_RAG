# Guardrails

1. **Flag instruction-like retrieved text at ingestion.** Scan every child chunk
   for instruction-like phrases such as `ignore previous instructions`,
   `reveal`, and `system prompt`. Persist a `suspicious` tag on matching chunks;
   do not drop them. Keep them indexed, and add a warning line when showing a
   flagged chunk to the generator.
2. **Reject unsafe generated output before model verification.** Reject answers
   that contain the generator system prompt, an API-key pattern, or a citation
   naming a document outside the retrieved set.
3. **Require citations for verification.** An answer with zero structured
   citations is a failed verification, even if its text is non-empty.
