# Browser event policy

The browser may receive only these Server-Sent Events. Every event has a
positive, monotonically increasing `id`; the same value is sent in the SSE `id:`
field so a client can reconnect with `Last-Event-ID`.

| `event:` type | JSON `data:` fields | Purpose |
| --- | --- | --- |
| `node` | `id`, `type`, `label` | A safe, human-readable progress label, such as `Retrieving...`, `Verifying sources...`, or `Generating...`. |
| `token` | `id`, `type`, `text` | Answer text emitted only by the generate node. |
| `citations` | `id`, `type`, `citations` | Citation list; each citation has only `doc_name` and `page`. |
| `final` | `id`, `type`, `answer`, `abstained`, `cache_hit` | Final answer and completion flags. |
| `error` | `id`, `type`, `message` | A fixed generic error message with no implementation details. |

No other event types or payload fields are allowed. In particular, never send
grader scores, prompts, rewritten queries, raw tracebacks or stack traces, model
usage details, tenant identifiers, or content belonging to another tenant.
Retrieved text is emitted only as answer text from the generate node, after the
existing verification and output filters have accepted it. Citation document
names and pages must come from the current caller's retrieved set.

`data:` is a single JSON object validated by the corresponding Pydantic model.
The SSE `event:` value must match its `type` field. The `id:` value must match
the model's `id`.
