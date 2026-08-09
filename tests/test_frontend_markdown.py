import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def render_markdown(markdown: str) -> str:
    script = """
const renderer = require('./frontend/markdown.js');
process.stdout.write(renderer.renderSafeMarkdown(process.argv[1]));
"""
    completed = subprocess.run(
        ["node", "-e", script, markdown],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_recipe_source_http_url_renders_as_an_escaped_safe_anchor() -> None:
    """An approved HTTP(S) Markdown URL must become a safe clickable anchor."""
    rendered = render_markdown(
        "[Tomato <Soup>](https://recipes.example/soup?item=tomato&lang=en)"
    )

    assert rendered == (
        '<p><a href="https://recipes.example/soup?item=tomato&amp;lang=en" '
        'target="_blank" rel="noopener noreferrer">Tomato &lt;Soup&gt;</a></p>'
    )


def test_non_http_markdown_url_never_renders_as_an_anchor() -> None:
    """The inline renderer must leave a non-allowlisted URL inert and escaped."""
    rendered = render_markdown("[unsafe <link>](javascript:alert(1))")

    assert "<a " not in rendered
    assert rendered == "<p>[unsafe &lt;link&gt;](javascript:alert(1))</p>"
