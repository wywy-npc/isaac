"""Tests for the auto-linker — memory_write auto-wikilink and stub creation."""
import tempfile
from pathlib import Path

import pytest

from isaac.memory.store import MemoryStore
from isaac.memory.linker import auto_link


@pytest.fixture
def store(tmp_path):
    return MemoryStore(memory_dir=tmp_path)


class TestAutoLinkNoExistingNodes:
    def test_empty_content_unchanged(self, store):
        content, stubs = auto_link("", "notes/test.md", store)
        assert content == ""
        assert stubs == []

    def test_no_matches_returns_unchanged(self, store):
        content, stubs = auto_link("just some plain text", "notes/test.md", store)
        assert "## Related" not in content
        assert stubs == []


class TestAutoLinkWithExistingNodes:
    def test_links_to_matching_nodes(self, store):
        # Create some existing nodes
        store.write("projects/isaac.md", "# ISAAC\nMulti-agent terminal harness", {"tags": ["project"], "importance": 0.8})
        store.write("people/wyatt.md", "# Wyatt Wilson\nAI developer", {"tags": ["person"], "importance": 0.7})

        content = "# Meeting Notes\nDiscussed ISAAC project with Wyatt Wilson today."
        result, stubs = auto_link(content, "logs/2026-04-02.md", store)

        assert "## Related" in result
        # Should find at least one of the existing nodes
        assert "[[" in result

    def test_self_path_excluded(self, store):
        store.write("projects/isaac.md", "# ISAAC\nMulti-agent harness", {"tags": ["project"], "importance": 0.8})

        content = "# ISAAC\nThis is the ISAAC project itself."
        result, stubs = auto_link(content, "projects/isaac.md", store)

        # Should not self-link
        assert "[[projects/isaac.md]]" not in result

    def test_explicit_links_not_duplicated(self, store):
        store.write("people/jane.md", "# Jane Doe\nFounder of Acme", {"tags": ["person"], "importance": 0.7})

        content = "# Deal Note\nMet with [[people/jane.md]] about the deal."
        result, stubs = auto_link(content, "deals/acme.md", store)

        # Count occurrences of jane link — should still be just the original one
        assert result.count("[[people/jane.md]]") == 1

    def test_max_links_cap(self, store):
        # Create many nodes
        for i in range(10):
            store.write(f"entities/company-{i}.md", f"# Company {i}\nA company about testing", {"tags": ["company"], "importance": 0.5})

        content = "# Research\nLooking at testing companies in the market."
        result, stubs = auto_link(content, "logs/research.md", store, max_links=3)

        # Count link entries in Related section
        links = result.split("## Related")[-1].count("[[") if "## Related" in result else 0
        assert links <= 3

    def test_appends_to_existing_related_section(self, store):
        store.write("projects/alpha.md", "# Alpha\nAlpha project details", {"tags": ["project"], "importance": 0.8})

        content = "# Notes\nSome notes about Alpha project.\n\n## Related\n- [[people/bob.md]]"
        result, stubs = auto_link(content, "logs/notes.md", store)

        # Should still have the original link
        assert "[[people/bob.md]]" in result
        # Should not create a second ## Related header
        assert result.count("## Related") == 1


class TestStubCreation:
    def test_creates_stubs_for_proper_nouns(self, store):
        content = "# Meeting\nDiscussed partnership with Mercato Partners about the deal."
        result, stubs = auto_link(content, "logs/meeting.md", store)

        assert len(stubs) > 0
        assert "entities/mercato-partners.md" in stubs

        # Verify the stub was actually written
        stub_node = store.read("entities/mercato-partners.md")
        assert stub_node is not None
        assert "Mercato Partners" in stub_node.content
        assert "stub" in stub_node.tags

    def test_stub_has_proper_format(self, store):
        content = "# Note\nTalking to Acme Corporation about investment."
        result, stubs = auto_link(content, "deals/note.md", store)

        stub_node = store.read("entities/acme-corporation.md")
        assert stub_node is not None
        assert stub_node.content.startswith("# Acme Corporation")
        assert "[[deals/note.md]]" in stub_node.content
        assert stub_node.meta.get("importance") == 0.3
        assert "stub" in stub_node.meta.get("tags", [])

    def test_no_stub_if_node_exists(self, store):
        # Pre-create the node
        store.write("entities/mercato-partners.md", "# Mercato Partners\nA venture firm.", {"importance": 0.8})

        content = "# Note\nWorking with Mercato Partners on this."
        result, stubs = auto_link(content, "logs/note.md", store)

        # Should not create a stub
        assert "entities/mercato-partners.md" not in stubs

        # But should still link to it
        assert "[[entities/mercato-partners.md]]" in result

    def test_stub_cap_at_3(self, store):
        content = (
            "# Big Meeting\n"
            "Met with Alpha Beta, Charlie Delta, Echo Foxtrot, "
            "Golf Hotel, and India Juliet to discuss the deal."
        )
        result, stubs = auto_link(content, "logs/big-meeting.md", store)

        assert len(stubs) <= 3

    def test_no_self_stub(self, store):
        content = "# Mercato Partners\nWe are a venture firm."
        result, stubs = auto_link(content, "entities/mercato-partners.md", store)

        assert "entities/mercato-partners.md" not in stubs

    def test_stubs_linked_in_related(self, store):
        content = "# Note\nDiscussed with Acme Corporation today."
        result, stubs = auto_link(content, "logs/note.md", store)

        if stubs:
            # At least one stub should appear in the Related section
            assert "## Related" in result
            for stub in stubs:
                assert f"[[{stub}]]" in result


class TestGracefulDegradation:
    def test_no_embedding_store(self, store):
        store.write("projects/test.md", "# Test Project\nSome content", {"importance": 0.5})
        content = "# Note\nWorking on the Test Project."
        result, stubs = auto_link(content, "logs/note.md", store, embedding_store=None)
        # Should still work with keyword search only
        assert isinstance(result, str)
        assert isinstance(stubs, list)

    def test_broken_embedding_store(self, store):
        class BrokenEmbeddings:
            def search_similar(self, *args, **kwargs):
                raise RuntimeError("embeddings broken")

        store.write("projects/test.md", "# Test\nContent", {"importance": 0.5})
        content = "# Note\nAbout the Test project."
        result, stubs = auto_link(content, "logs/note.md", store, embedding_store=BrokenEmbeddings())
        # Should degrade gracefully
        assert isinstance(result, str)


class TestSkipNouns:
    def test_skips_common_headings(self, store):
        content = "# Summary\nThis is an Overview of Next Steps and Action Items."
        result, stubs = auto_link(content, "logs/note.md", store)

        # Should not create stubs for "Next Steps", "Action Items", etc.
        assert "entities/next-steps.md" not in stubs
        assert "entities/action-items.md" not in stubs
