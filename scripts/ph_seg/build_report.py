"""Assemble the HTML report (self-contained, images inlined as JPEG data URIs)."""
import base64, io, json, sys, pandas as pd, numpy as np
from PIL import Image
from common import OUT

FIG_MAX_W = 2200


def img_uri(path, max_w=FIG_MAX_W, q=82):
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def table(df, fmt=None, index_name=""):
    fmt = fmt or {}
    h = ["<table><thead><tr><th>" + index_name + "</th>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr></thead><tbody>"]
    for idx, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                v = fmt.get(c, "{:.2f}").format(v)
            cells.append(f"<td>{v}</td>")
        h.append(f"<tr><th>{idx}</th>" + "".join(cells) + "</tr>")
    h.append("</tbody></table>")
    return "\n".join(h)


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    out = open(sys.argv[2], "w")
    css = open(spec["css"]).read()
    body = open(spec["body"]).read()
    # substitute {{fig:path}} and {{table:name}}
    import re
    def rep_fig(m):
        p, cap = m.group(1), m.group(2) or ""
        return f'<figure><img src="{img_uri(p)}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
    body = re.sub(r"\{\{fig:([^|}]+)\|?([^}]*)\}\}", rep_fig, body)
    def rep_tab(m):
        name = m.group(1)
        t = spec["tables"][name]
        df = pd.read_csv(t["csv"], index_col=0)
        if "cols" in t: df = df[t["cols"]]
        if "rows" in t: df = df.loc[t["rows"]]
        if "rename" in t: df = df.rename(columns=t["rename"])
        return table(df, t.get("fmt"), t.get("index_name", ""))
    body = re.sub(r"\{\{table:([^}]+)\}\}", rep_tab, body)
    out.write(f"<title>{spec['title']}</title>\n<style>{css}</style>\n{body}")
    out.close()
    import os
    print("wrote", sys.argv[2], os.path.getsize(sys.argv[2]) / 1e6, "MB")
