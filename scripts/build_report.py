"""Assemble the HTML report with figures inlined as base64 data URIs."""

import base64
from pathlib import Path

OUT = Path("outputs")
TEMPLATE = Path("scripts/report_template.html").read_text()


def data_uri(path):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


html = TEMPLATE.replace("{{CHIBA_PNG}}", data_uri(OUT / "chiba_change_detection.png"))
html = html.replace("{{BASSIN_PNG}}", data_uri(OUT / "bassin_change_detection.png"))
html = html.replace("{{CHIBA_MONITOR_PNG}}", data_uri(OUT / "monitor/chiba_alerting.png"))
html = html.replace("{{BASSIN_MONITOR_PNG}}", data_uri(OUT / "monitor/bassin_alerting.png"))
html = html.replace("{{CHIBA_PAIR_PNG}}", data_uri(OUT / "chiba_ssl4eo_pair.png"))
html = html.replace("{{BASSIN_PAIR_PNG}}", data_uri(OUT / "bassin_ssl4eo_pair.png"))
out = OUT / "report.html"
out.write_text(html)
print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")
