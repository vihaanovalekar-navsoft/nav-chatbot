"""Render a self-contained HTML report for RAG evaluation results."""

from html import escape
from statistics import mean

STYLES = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #f3f4f6;
  color: #111827;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial,
    sans-serif;
  line-height: 1.45;
}
.wrap { max-width: 1200px; margin: 0 auto; padding: 32px 20px 60px; }
header h1 { font-size: 30px; margin: 0 0 6px; }
.meta { color: #6b7280; font-size: 14px; }
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
  margin: 24px 0;
}
.card {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 14px 16px;
}
.card .label {
  color: #6b7280;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.card .value { font-size: 26px; font-weight: 700; margin-top: 4px; }
.card .note { color: #6b7280; font-size: 12px; margin-top: 2px; }
h2 { font-size: 19px; margin: 28px 0 10px; }
table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  overflow: hidden;
}
th, td {
  padding: 9px 11px;
  border-bottom: 1px solid #e5e7eb;
  text-align: left;
  vertical-align: top;
  font-size: 14px;
  overflow-wrap: anywhere;
}
th {
  background: #f9fafb;
  color: #374151;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
tr:last-child td { border-bottom: none; }
.num { white-space: nowrap; font-variant-numeric: tabular-nums; }
.pass { color: #047857; font-weight: 700; }
.fail { color: #b91c1c; font-weight: 700; }
.reason { max-width: 340px; }
.muted { color: #6b7280; }
.note-box {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-left: 4px solid #2563eb;
  border-radius: 6px;
  padding: 12px 14px;
  color: #374151;
  font-size: 14px;
  margin: 12px 0;
}
details {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin: 8px 0;
  padding: 10px 14px;
}
details summary { cursor: pointer; font-weight: 600; }
details h3 { font-size: 14px; margin: 14px 0 4px; color: #374151; }
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  padding: 10px;
  border-radius: 6px;
  margin: 4px 0 0;
  font-size: 13px;
}
ul { margin: 4px 0 0; padding-left: 20px; }
footer { margin-top: 32px; color: #6b7280; font-size: 12px; }
@media print {
  body { background: #fff; }
  .wrap { max-width: none; }
  details { break-inside: avoid; }
}
"""


def _mean(values):
    return mean(values) if values else None


def _percent(value):
    return "n/a" if value is None else f"{value:.1%}"


def _number(value):
    return "n/a" if value is None else f"{value:,.0f}"


def _ratio(rows):
    return (
        sum(row["score"] for row in rows) / len(rows)
        if rows
        else None
    )


def _case_note(correct, total):
    noun = "case" if total == 1 else "cases"
    return f"{correct} / {total} {noun}"


def _config_table(config):
    items = [
        ("Chunk size", f"{config['chunk_size']} tokens"),
        ("Retrieval mode", config["retrieval_mode"]),
        ("Candidates per retriever", f"{config['retrieval_k']} chunks"),
        ("Cross-encoder top", f"{config['crossencoder_k']} chunks"),
        ("RRF k", str(config["rrf_k"])),
        ("BM25 k1", str(config["bm25_k1"])),
        ("BM25 b", str(config["bm25_b"])),
        ("Answer temperature", str(config["temperature"])),
        ("Judge temperature", str(config["judge_temperature"])),
        ("Reasoning effort", config["reasoning_effort"]),
        ("Request RPM cap", f"{config['rpm']} requests/min"),
        ("Prompt version", config["prompt_version"]),
        ("Prompt fingerprint", config["prompt_fingerprint"]),
        ("Judge version", config["judge_version"]),
        ("Judge fingerprint", config["judge_fingerprint"]),
        ("Answer model", config["answer_model"]),
        ("Judge model", config["judge_model"]),
        ("Token counter", config["tokenizer_model"]),
        ("Embedding model", config["embedding_model"]),
        ("Reranker model", config["reranker_model"]),
        ("Vector store", config["chroma_path"]),
        ("Cases file", config["cases_file"]),
    ]
    return "\n".join(
        f"<tr><th>{escape(str(label))}</th>"
        f"<td>{escape(str(value))}</td></tr>"
        for label, value in items
    )


def _prompt_section(data):
    config = data["config"]
    return (
        "<section>\n<h2>System Prompt</h2>\n"
        f"<details><summary>{escape(config['prompt_version'])} &middot; "
        f"fingerprint {escape(config['prompt_fingerprint'])}</summary>"
        f"<pre>{escape(data['prompt'])}</pre></details>\n"
        "</section>\n"
    )


def _judge_prompt_section(data):
    config = data["config"]
    return (
        "<section>\n<h2>Judge Prompt</h2>\n"
        f"<details><summary>{escape(config['judge_version'])} &middot; "
        f"fingerprint {escape(config['judge_fingerprint'])}</summary>"
        f"<pre>{escape(data['judge_prompt'])}</pre></details>\n"
        "</section>\n"
    )


def _metric_table(metrics):
    items = [
        ("Correct cases", f"{metrics['correct']} / {metrics['total']}"),
        ("Overall accuracy", _percent(metrics["accuracy"])),
        (
            "Answerable accuracy",
            f"{_percent(metrics['answerable_accuracy'])} "
            f"({metrics['answerable_correct']} / "
            f"{metrics['answerable_total']})",
        ),
        (
            "Unanswerable accuracy",
            f"{_percent(metrics['unanswerable_accuracy'])} "
            f"({metrics['unanswerable_correct']} / "
            f"{metrics['unanswerable_total']})",
        ),
        (
            "Average answer length",
            f"{metrics['avg_answer_words']:.1f} words / "
            f"{metrics['avg_answer_chars']:.0f} chars",
        ),
        (
            "Context tokens",
            f"avg {_number(metrics['avg_context_tokens'])} | "
            f"max {_number(metrics['max_context_tokens'])} | "
            f"total {_number(metrics['total_context_tokens'])}",
        ),
        (
            "Prompt tokens",
            f"avg {_number(metrics['avg_prompt_tokens'])} | "
            f"total {_number(metrics['total_prompt_tokens'])}",
        ),
        (
            "Total input tokens",
            f"avg {_number(metrics['avg_total_input_tokens'])} | "
            f"total {_number(metrics['total_input_tokens'])}",
        ),
        (
            "System instruction",
            f"{_number(metrics['system_instruction_tokens'])} tokens",
        ),
    ]
    return "\n".join(
        f"<tr><th>{escape(str(label))}</th>"
        f"<td>{escape(str(value))}</td></tr>"
        for label, value in items
    )


def _case_table(results):
    rows = []
    for result in results:
        status = "pass" if result["score"] else "fail"
        case_type = "Answerable" if result["answerable"] else "Unanswerable"
        rows.append(
            "<tr>"
            f"<td class='num'>{result['number']}</td>"
            f"<td>{escape(result['question'])}</td>"
            f"<td>{case_type}</td>"
            f"<td class='{status}'>{result['score']}</td>"
            f"<td class='num'>{result['answer_words']} w / "
            f"{result['answer_chars']} c</td>"
            f"<td class='num'>{result['context_tokens']:,}</td>"
            f"<td class='num'>{result['prompt_tokens']:,}</td>"
            f"<td class='reason'>{escape(result['reason'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _case_details(results):
    blocks = []
    for result in results:
        status = "PASS" if result["score"] else "FAIL"
        points = "".join(
            f"<li>{escape(point)}</li>" for point in result["key_points"]
        )
        sources = "".join(
            f"<li>{escape(source)}</li>" for source in result["sources"]
        )
        if not sources:
            sources = "<li>No sources returned.</li>"
        blocks.append(
            "<details>"
            f"<summary>#{result['number']} &middot; {status} &middot; "
            f"{escape(result['question'])}</summary>"
            "<h3>Ideal answer</h3>"
            f"<p>{escape(result['ideal_answer'])}</p>"
            "<h3>Key points</h3>"
            f"<ul>{points}</ul>"
            "<h3>Generated answer</h3>"
            f"<pre>{escape(result['answer'])}</pre>"
            "<h3>Judge reason</h3>"
            f"<p>{escape(result['reason'])}</p>"
            "<h3>Sources</h3>"
            f"<ul>{sources}</ul>"
            "</details>"
        )
    return "\n".join(blocks)


def render_report(data):
    """Return a self-contained HTML report for evaluation `data`."""
    results = data["results"]
    total = len(results)
    correct = sum(result["score"] for result in results)
    answerable = [result for result in results if result["answerable"]]
    unanswerable = [result for result in results if not result["answerable"]]

    answer_words = [result["answer_words"] for result in results]
    answer_chars = [result["answer_chars"] for result in results]
    context_tokens = [result["context_tokens"] for result in results]
    prompt_tokens = [result["prompt_tokens"] for result in results]
    system_tokens = data["system_instruction_tokens"]

    avg_context = _mean(context_tokens)
    avg_prompt = _mean(prompt_tokens)
    avg_total_input = (
        avg_prompt + system_tokens if avg_prompt is not None else None
    )
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "answerable_total": len(answerable),
        "answerable_correct": sum(row["score"] for row in answerable),
        "answerable_accuracy": _ratio(answerable),
        "unanswerable_total": len(unanswerable),
        "unanswerable_correct": sum(row["score"] for row in unanswerable),
        "unanswerable_accuracy": _ratio(unanswerable),
        "avg_answer_words": _mean(answer_words) or 0,
        "avg_answer_chars": _mean(answer_chars) or 0,
        "avg_context_tokens": avg_context or 0,
        "max_context_tokens": max(context_tokens, default=0),
        "total_context_tokens": sum(context_tokens),
        "avg_prompt_tokens": avg_prompt or 0,
        "total_prompt_tokens": sum(prompt_tokens),
        "avg_total_input_tokens": avg_total_input or 0,
        "total_input_tokens": sum(prompt_tokens) + total * system_tokens,
        "system_instruction_tokens": system_tokens,
    }

    cards = [
        (
            "Overall accuracy",
            _percent(metrics["accuracy"]),
            _case_note(correct, total),
        ),
        (
            "Answerable accuracy",
            _percent(metrics["answerable_accuracy"]),
            _case_note(
                metrics["answerable_correct"],
                metrics["answerable_total"],
            ),
        ),
        (
            "Unanswerable accuracy",
            _percent(metrics["unanswerable_accuracy"]),
            _case_note(
                metrics["unanswerable_correct"],
                metrics["unanswerable_total"],
            ),
        ),
        (
            "Avg answer length",
            f"{metrics['avg_answer_words']:.1f} words",
            f"{metrics['avg_answer_chars']:.0f} characters",
        ),
        (
            "Avg context tokens",
            _number(metrics["avg_context_tokens"]),
            f"max {metrics['max_context_tokens']:,} | "
            f"total {metrics['total_context_tokens']:,}",
        ),
        (
            "Avg total input",
            _number(metrics["avg_total_input_tokens"]),
            "prompt + system instruction",
        ),
    ]
    card_html = "\n".join(
        "<div class='card'>"
        f"<div class='label'>{escape(label)}</div>"
        f"<div class='value'>{escape(value)}</div>"
        f"<div class='note'>{escape(note)}</div>"
        "</div>"
        for label, value, note in cards
    )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>NavBot RAG Evaluation</title>\n"
        f"<style>{STYLES}</style>\n"
        "</head>\n"
        "<body>\n"
        '<main class="wrap">\n'
        "<header>\n"
        "<h1>NavBot RAG Evaluation</h1>\n"
        f'<div class="meta">Generated {escape(data["generated_at"])} &middot; '
        f'{escape(data["config"]["cases_file"])}</div>\n'
        "</header>\n"
        f'<section class="cards">\n{card_html}\n</section>\n'
        "<section>\n<h2>Configuration</h2>\n"
        f"<table>{_config_table(data['config'])}</table>\n"
        "</section>\n"
        f"{_prompt_section(data)}"
        f"{_judge_prompt_section(data)}"
        "<section>\n<h2>Metrics</h2>\n"
        '<div class="note-box">Hybrid retrieval fuses dense and BM25 '
        "rankings with reciprocal rank fusion, then reranks with the "
        "cross-encoder. Context tokens count only the retrieved "
        "&lt;1&gt;...&lt;/1&gt; blocks. Prompt tokens add the history and "
        "question. Total input tokens add the fixed system instruction to "
        "every answer request. Judge requests are excluded. Counts use "
        "Gemini's token counter as a proxy for the Groq tokenizer.</div>\n"
        f"<table>{_metric_table(metrics)}</table>\n"
        "</section>\n"
        "<section>\n<h2>Case results</h2>\n"
        "<table>\n"
        "<thead><tr><th>#</th><th>Question</th><th>Type</th>"
        "<th>Score</th><th>Answer length</th><th>Context tokens</th>"
        "<th>Prompt tokens</th><th>Judge reason</th></tr></thead>\n"
        f"<tbody>{_case_table(results)}</tbody>\n"
        "</table>\n"
        "</section>\n"
        "<section>\n<h2>Answer details</h2>\n"
        f"{_case_details(results)}\n"
        "</section>\n"
        "<footer>AI-judged evaluation. Each case receives a score of 0 or 1."
        "</footer>\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )


def write_report(path, data):
    """Write the rendered report to `path` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(data), encoding="utf-8")
    return path
