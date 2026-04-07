"""Tests for the spreadsheet plugin — Excel/CSV read, write, edit, diff, convert."""
import csv
from pathlib import Path

import pytest

from isaac.plugins.spreadsheet import build_spreadsheet_tools, _file_fingerprint


# ── Helpers ──────────────────────────────────────────────────────────

def _make_csv(tmp_path: Path, name: str = "test.csv", rows: list[list[str]] | None = None) -> Path:
    """Create a CSV file and return its path."""
    if rows is None:
        rows = [
            ["Name", "Revenue", "Region"],
            ["Acme Corp", "1000000", "US"],
            ["Globex", "2500000", "EU"],
            ["Initech", "750000", "US"],
        ]
    path = tmp_path / name
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return path


def _get_tool(registry: dict, name: str):
    """Extract the handler function from the registry."""
    return registry[name][1]


@pytest.fixture
def tools(tmp_path):
    """Build spreadsheet tools rooted at tmp_path."""
    return build_spreadsheet_tools(cwd=str(tmp_path))


# ── CSV tests ────────────────────────────────────────────────────────

class TestCSV:
    @pytest.mark.asyncio
    async def test_read_csv(self, tools, tmp_path):
        csv_path = _make_csv(tmp_path)
        read = _get_tool(tools, "spreadsheet_read")
        result = await read(path=str(csv_path))

        assert "error" not in result
        assert result["format"] == "csv"
        sheet = result["sheets"]["Sheet1"]
        assert sheet["header"] == ["Name", "Revenue", "Region"]
        assert sheet["row_count"] == 4  # header + 3 data rows
        assert len(sheet["data"]) == 4

    @pytest.mark.asyncio
    async def test_write_csv(self, tools, tmp_path):
        write = _get_tool(tools, "spreadsheet_write")
        out = tmp_path / "output.csv"
        data = [["Col1", "Col2"], ["a", "1"], ["b", "2"]]
        result = await write(path=str(out), data=data)

        assert "error" not in result
        assert result["format"] == "csv"
        assert result["rows"] == 3
        assert out.exists()

        # Verify content
        with open(out) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows == [["Col1", "Col2"], ["a", "1"], ["b", "2"]]

    @pytest.mark.asyncio
    async def test_inspect_csv(self, tools, tmp_path):
        csv_path = _make_csv(tmp_path)
        inspect = _get_tool(tools, "spreadsheet_inspect")
        result = await inspect(path=str(csv_path))

        assert result["format"] == "csv"
        assert result["row_count"] == 4
        assert result["col_count"] == 3
        assert result["header"] == ["Name", "Revenue", "Region"]
        assert "fingerprint" in result

    @pytest.mark.asyncio
    async def test_diff_csv(self, tools, tmp_path):
        csv_a = _make_csv(tmp_path, "a.csv")
        csv_b = _make_csv(tmp_path, "b.csv", rows=[
            ["Name", "Revenue", "Region"],
            ["Acme Corp", "1200000", "US"],  # changed revenue
            ["Globex", "2500000", "EU"],      # unchanged
            ["Initech", "750000", "APAC"],    # changed region
        ])
        diff = _get_tool(tools, "spreadsheet_diff")
        result = await diff(path_a=str(csv_a), path_b=str(csv_b))

        assert "error" not in result
        assert result["total_diffs"] == 2
        changes = result["diffs"]
        cells_changed = {d["cell"] for d in changes}
        assert "B2" in cells_changed  # Revenue change
        assert "C4" in cells_changed  # Region change


# ── XLSX tests ───────────────────────────────────────────────────────

class TestXLSX:
    @pytest.fixture
    def xlsx_path(self, tmp_path):
        """Create a sample XLSX file."""
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Revenue"
        ws.append(["Company", "Q1", "Q2", "Total"])
        ws.append(["Acme", 100, 200, "=B2+C2"])
        ws.append(["Globex", 300, 400, "=B3+C3"])
        ws.append(["Sum", "=SUM(B2:B3)", "=SUM(C2:C3)", "=SUM(D2:D3)"])
        path = tmp_path / "test.xlsx"
        wb.save(str(path))
        wb.close()
        return path

    @pytest.mark.asyncio
    async def test_read_xlsx(self, tools, xlsx_path):
        read = _get_tool(tools, "spreadsheet_read")
        result = await read(path=str(xlsx_path), include_formulas=True)

        assert "error" not in result
        assert result["format"] == "xlsx"
        assert "Revenue" in result["sheet_names"]
        sheet = result["sheets"]["Revenue"]
        assert sheet["header"] == ["Company", "Q1", "Q2", "Total"]
        assert sheet["row_count"] == 4
        # Should have formulas
        assert len(sheet.get("formulas", [])) > 0

    @pytest.mark.asyncio
    async def test_read_xlsx_data_only(self, tools, xlsx_path):
        read = _get_tool(tools, "spreadsheet_read")
        result = await read(path=str(xlsx_path), include_formulas=False)

        assert "error" not in result
        # With data_only, formulas won't show as formulas
        sheet = result["sheets"]["Revenue"]
        # data_only on a fresh file (never opened in Excel) returns None for formula cells
        assert sheet["data"] is not None

    @pytest.mark.asyncio
    async def test_write_xlsx(self, tools, tmp_path):
        write = _get_tool(tools, "spreadsheet_write")
        out = tmp_path / "output.xlsx"
        data = [["Name", "Score"], ["Alice", 95], ["Bob", 87]]
        result = await write(path=str(out), data=data, sheet="Scores")

        assert "error" not in result
        assert result["format"] == "xlsx"
        assert result["sheet"] == "Scores"
        assert out.exists()

        # Verify by reading back
        read = _get_tool(tools, "spreadsheet_read")
        readback = await read(path=str(out))
        assert "Scores" in readback["sheets"]

    @pytest.mark.asyncio
    async def test_write_xlsx_preserves_other_sheets(self, tools, xlsx_path):
        """Writing to a new sheet should preserve existing sheets."""
        write = _get_tool(tools, "spreadsheet_write")
        data = [["Extra", "Data"], ["x", 1]]
        result = await write(path=str(xlsx_path), data=data, sheet="NewSheet")

        assert "error" not in result

        # Verify both sheets exist
        read = _get_tool(tools, "spreadsheet_read")
        readback = await read(path=str(xlsx_path))
        assert "Revenue" in readback["sheets"]
        assert "NewSheet" in readback["sheets"]

    @pytest.mark.asyncio
    async def test_edit_xlsx(self, tools, xlsx_path):
        edit = _get_tool(tools, "spreadsheet_edit")
        result = await edit(
            path=str(xlsx_path),
            edits=[
                {"cell": "B2", "value": 999},
                {"cell": "D4", "formula": "=SUM(D2:D3)"},
            ],
        )

        assert "error" not in result
        assert result["edits_applied"] == 2

        # Verify
        read = _get_tool(tools, "spreadsheet_read")
        readback = await read(path=str(xlsx_path), include_formulas=True)
        sheet = readback["sheets"]["Revenue"]
        # Row 2 (index 1), col B (index 1) should be 999
        assert sheet["data"][1][1] == "999"

    @pytest.mark.asyncio
    async def test_inspect_xlsx(self, tools, xlsx_path):
        inspect = _get_tool(tools, "spreadsheet_inspect")
        result = await inspect(path=str(xlsx_path))

        assert result["format"] == "xlsx"
        assert "Revenue" in result["sheet_names"]
        assert result["sheets"]["Revenue"]["row_count"] == 4
        assert result["sheets"]["Revenue"]["col_count"] == 4
        assert "fingerprint" in result

    @pytest.mark.asyncio
    async def test_diff_xlsx(self, tools, tmp_path, xlsx_path):
        """Diff two XLSX files."""
        openpyxl = pytest.importorskip("openpyxl")
        # Create a modified copy
        import shutil
        modified = tmp_path / "modified.xlsx"
        shutil.copy(xlsx_path, modified)

        edit = _get_tool(tools, "spreadsheet_edit")
        await edit(path=str(modified), edits=[{"cell": "B2", "value": 999}])

        diff = _get_tool(tools, "spreadsheet_diff")
        result = await diff(path_a=str(xlsx_path), path_b=str(modified))

        assert "error" not in result
        assert result["total_diffs"] >= 1
        changed_cells = {d["cell"] for d in result["diffs"]}
        assert "B2" in changed_cells


# ── Convert tests ────────────────────────────────────────────────────

class TestConvert:
    @pytest.mark.asyncio
    async def test_csv_to_xlsx(self, tools, tmp_path):
        csv_path = _make_csv(tmp_path)
        convert = _get_tool(tools, "spreadsheet_convert")
        result = await convert(path=str(csv_path), output_format="xlsx")

        assert "error" not in result
        assert result["output_format"] == "xlsx"
        xlsx_out = Path(result["output"])
        assert xlsx_out.exists()

    @pytest.mark.asyncio
    async def test_xlsx_to_csv(self, tools, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        # Create XLSX first
        write = _get_tool(tools, "spreadsheet_write")
        xlsx_path = tmp_path / "data.xlsx"
        await write(path=str(xlsx_path), data=[["A", "B"], ["1", "2"]])

        convert = _get_tool(tools, "spreadsheet_convert")
        result = await convert(path=str(xlsx_path), output_format="csv")

        assert "error" not in result
        assert result["output_format"] == "csv"
        csv_out = Path(result["output"])
        assert csv_out.exists()

    @pytest.mark.asyncio
    async def test_convert_same_format_error(self, tools, tmp_path):
        csv_path = _make_csv(tmp_path)
        convert = _get_tool(tools, "spreadsheet_convert")
        result = await convert(path=str(csv_path), output_format="csv")
        assert "error" in result


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_read_missing_file(self, tools):
        read = _get_tool(tools, "spreadsheet_read")
        result = await read(path="/nonexistent/file.xlsx")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_read_unsupported_format(self, tools, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("not a spreadsheet")
        read = _get_tool(tools, "spreadsheet_read")
        result = await read(path=str(txt))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_edit_csv_error(self, tools, tmp_path):
        """spreadsheet_edit should reject CSV files."""
        csv_path = _make_csv(tmp_path)
        edit = _get_tool(tools, "spreadsheet_edit")
        result = await edit(path=str(csv_path), edits=[{"cell": "A1", "value": "x"}])
        assert "error" in result

    @pytest.mark.asyncio
    async def test_fingerprint_changes(self, tools, tmp_path):
        """Fingerprint should change when file content changes."""
        csv_path = _make_csv(tmp_path)
        fp1 = _file_fingerprint(csv_path)

        # Modify the file
        with open(csv_path, "a") as f:
            f.write("NewRow,999,UK\n")
        fp2 = _file_fingerprint(csv_path)

        assert fp1["sha256_prefix"] != fp2["sha256_prefix"]
        assert fp2["size_bytes"] > fp1["size_bytes"]

    @pytest.mark.asyncio
    async def test_max_rows_truncation(self, tools, tmp_path):
        """Reading with max_rows should truncate."""
        rows = [["Col"]] + [[str(i)] for i in range(100)]
        csv_path = _make_csv(tmp_path, rows=rows)
        read = _get_tool(tools, "spreadsheet_read")
        result = await read(path=str(csv_path), max_rows=10)

        sheet = result["sheets"]["Sheet1"]
        assert sheet["rows_returned"] == 10
        assert sheet["truncated"] is True


# ── Tool registry tests ──────────────────────────────────────────────

class TestRegistry:
    def test_all_tools_registered(self, tools):
        expected = {
            "spreadsheet_read",
            "spreadsheet_write",
            "spreadsheet_edit",
            "spreadsheet_diff",
            "spreadsheet_inspect",
            "spreadsheet_convert",
        }
        assert set(tools.keys()) == expected

    def test_tool_defs_have_required_fields(self, tools):
        for name, (tool_def, handler) in tools.items():
            assert tool_def.name == name
            assert tool_def.description
            assert tool_def.input_schema
            assert callable(handler)

    def test_read_only_tools(self, tools):
        read_only = {"spreadsheet_read", "spreadsheet_diff", "spreadsheet_inspect"}
        for name in read_only:
            assert tools[name][0].is_read_only is True
        write_tools = {"spreadsheet_write", "spreadsheet_edit", "spreadsheet_convert"}
        for name in write_tools:
            assert tools[name][0].is_read_only is False
