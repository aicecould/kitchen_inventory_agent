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


def render_profile_cards(payload: str) -> str:
    script = """
const cards = require('./frontend/profile-proposals.js');
const proposals = JSON.parse(process.argv[1]);
process.stdout.write(cards.renderProfileProposalCards(proposals));
"""
    completed = subprocess.run(
        ["node", "-e", script, payload],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_profile_card_escapes_values_and_exposes_two_decisions() -> None:
    rendered = render_profile_cards(
        """[{"proposal_id":"prof_11111111111111111111111111111111",
        "category":"preference","key":"ingredient.cilantro",
        "value":"<不喜欢香菜>","reason":"用户说 & 永久避免",
        "evidence_ids":["hist_22222222222222222222222222222222"],
        "preview":{"current":null}}]"""
    )

    assert "&lt;不喜欢香菜&gt;" in rendered
    assert "用户说 &amp; 永久避免" in rendered
    assert "hist_22222222222222222222222222222222" in rendered
    assert 'data-decision="confirm"' in rendered
    assert 'data-decision="reject"' in rendered
    assert "<不喜欢香菜>" not in rendered


def test_empty_profile_proposals_render_no_card_markup() -> None:
    assert render_profile_cards("[]") == ""


def test_profile_card_endpoint_encodes_id_and_allowlists_decision() -> None:
    script = """
const cards = require('./frontend/profile-proposals.js');
process.stdout.write(cards.profileProposalEndpoint(process.argv[1], process.argv[2]));
"""
    completed = subprocess.run(
        ["node", "-e", script, "prof_/unsafe", "confirm"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == "/api/profile-proposals/prof_%2Funsafe/confirm"
