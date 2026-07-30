"""Self-contained HTML report generator for batch operations.

Single output file: dashboard.html, no external assets, no JS, no CDN.
Charts are inlined as SVG (matplotlib) so the report can be emailed,
attached to a paper, or stored offline as an audit artifact.

Usage
-----
    from lamquant_codec.batch import validate_batch
    from lamquant_codec.report import write_html_report

    report = validate_batch(inputs=['archive/'], reference_dir='data/',
                            level='C', recursive=True)
    write_html_report(report, 'archive_quality.html', level='C')

CLI:
    lamquant validate archive/ --reference data/ --level C \\
        --report-html archive_quality.html

What's in the HTML
------------------
1. Header card: pass-rate, totals, mean/p95/max for PRD/R/CR
2. Histograms: PRD, Pearson R, compression ratio
3. Scatter: CR vs R (one dot per file, coloured by pass/fail)
4. Top-10 failures table
5. Full per-file table (sortable client-side via vanilla JS)

The HTML is intentionally minimal — no jQuery, no Bootstrap, no React.
A clinical archive admin opens the file in any browser and gets an
artifact they can put in front of an IRB or hospital IT.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional, List

import numpy as np

from lamquant_codec.batch import BatchReport, BatchResult


# ============================================================
# Inline chart helpers — matplotlib → SVG string
# ============================================================

def _svg_histogram(values: list, title: str, xlabel: str,
                   bins: int = 30, threshold: Optional[float] = None,
                   threshold_label: Optional[str] = None) -> str:
    """Return an inline SVG <svg>...</svg> string for a histogram.

    Handles degenerate inputs gracefully: empty data produces a placeholder,
    all-identical values (range=0, common for lossless metrics) get a single
    centered bar instead of crashing matplotlib's binning.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Filter NaN / inf so the binner doesn't see them.
    arr = np.array([v for v in values
                    if v is not None and np.isfinite(v)],
                   dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    if arr.size == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                color='#9aa5b1', transform=ax.transAxes, fontsize=12)
    elif arr.max() - arr.min() < 1e-12:
        # All values identical — show a single centered bar so the chart
        # still renders meaningfully (typical for lossless: every R == 1.0).
        v = float(arr[0])
        ax.bar([v], [arr.size], width=max(abs(v) * 0.05, 0.01),
               color='#3a6ea5', edgecolor='white', linewidth=0.5)
        ax.set_xlim(v - 1, v + 1)
        ax.text(v, arr.size, f'  {arr.size} files @ {v:g}',
                va='center', fontsize=9, color='#52606d')
        if threshold is not None:
            ax.axvline(threshold, color='#c0392b', linestyle='--', linewidth=1.2,
                       label=threshold_label or f'threshold={threshold}')
            ax.legend(loc='upper right', fontsize=9)
    else:
        ax.hist(arr, bins=bins, color='#3a6ea5', edgecolor='white',
                linewidth=0.5)
        if threshold is not None:
            ax.axvline(threshold, color='#c0392b', linestyle='--', linewidth=1.2,
                       label=threshold_label or f'threshold={threshold}')
            ax.legend(loc='upper right', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('count', fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    svg = buf.getvalue()
    # Strip the XML declaration so it embeds cleanly inline.
    if svg.startswith('<?xml'):
        svg = svg.split('?>', 1)[-1]
    return svg


def _svg_scatter_cr_vs_r(results: List[BatchResult], min_r: float,
                          min_cr: float) -> str:
    """CR vs R scatter; pass/fail colour-coded against an LQS contract."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    pass_x = [r.cr for r in results if r.lqs_pass]
    pass_y = [r.pearson_r for r in results if r.lqs_pass]
    fail_x = [r.cr for r in results if r.lqs_pass is False]
    fail_y = [r.pearson_r for r in results if r.lqs_pass is False]
    if pass_x:
        ax.scatter(pass_x, pass_y, c='#27ae60', s=14, alpha=0.55,
                   label=f'pass ({len(pass_x)})')
    if fail_x:
        ax.scatter(fail_x, fail_y, c='#c0392b', s=18, alpha=0.7,
                   label=f'fail ({len(fail_x)})')
    ax.axhline(min_r, color='#c0392b', linestyle=':', linewidth=1, alpha=0.6)
    ax.axvline(min_cr, color='#c0392b', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_title('Compression ratio vs reconstruction quality',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('compression ratio', fontsize=10)
    ax.set_ylabel('Pearson R', fontsize=10)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if pass_x or fail_x:
        ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    svg = buf.getvalue()
    if svg.startswith('<?xml'):
        svg = svg.split('?>', 1)[-1]
    return svg


# ============================================================
# Data preparation
# ============================================================

def _summary_stats(values: list, fmt: str = '{:.3f}') -> dict:
    """Mean, p5, p50, p95, min, max — robust to empty / NaN."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if arr.size == 0:
        return {k: '—' for k in ('mean', 'p5', 'p50', 'p95', 'min', 'max', 'n')}
    return {
        'n': int(arr.size),
        'mean': fmt.format(float(arr.mean())),
        'p5':   fmt.format(float(np.percentile(arr, 5))),
        'p50':  fmt.format(float(np.percentile(arr, 50))),
        'p95':  fmt.format(float(np.percentile(arr, 95))),
        'min':  fmt.format(float(arr.min())),
        'max':  fmt.format(float(arr.max())),
    }


# ============================================================
# Jinja template — single-file, self-contained, no JS dependencies
# (small inline JS for table sort only — vanilla, ~30 lines).
# ============================================================

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  :root {
    --fg: #1f2933;
    --fg-soft: #52606d;
    --bg: #f5f7fa;
    --card: #ffffff;
    --border: #e4e7eb;
    --pass: #27ae60;
    --fail: #c0392b;
    --warn: #d68910;
    --accent: #3a6ea5;
  }
  * { box-sizing: border-box; }
  body {
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI",
          Roboto, Helvetica, Arial, sans-serif;
    color: var(--fg); background: var(--bg); margin: 0;
    padding: 24px;
  }
  h1 { font-size: 22px; margin: 0 0 4px 0; }
  h2 { font-size: 15px; text-transform: uppercase;
       letter-spacing: 0.05em; color: var(--fg-soft);
       margin: 28px 0 10px 0; border-bottom: 1px solid var(--border);
       padding-bottom: 4px; }
  .sub { color: var(--fg-soft); margin-bottom: 24px; font-size: 13px; }
  .cards { display: grid; gap: 12px; grid-template-columns:
           repeat(auto-fit, minmax(170px, 1fr)); }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 6px; padding: 14px 16px; }
  .card .label { font-size: 11px; text-transform: uppercase;
                 color: var(--fg-soft); letter-spacing: 0.04em; }
  .card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .card.pass .value { color: var(--pass); }
  .card.fail .value { color: var(--fail); }
  .card.warn .value { color: var(--warn); }
  table { width: 100%; border-collapse: collapse;
          background: var(--card); border-radius: 6px; overflow: hidden;
          font-size: 13px; }
  th, td { padding: 7px 10px; text-align: left;
           border-bottom: 1px solid var(--border); }
  th { background: #fafbfc; font-weight: 600; font-size: 12px;
       text-transform: uppercase; letter-spacing: 0.03em;
       color: var(--fg-soft); cursor: pointer; user-select: none; }
  th:hover { background: #f0f2f5; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr.pass { }
  tr.fail td:first-child::before { content: '⚠ '; color: var(--fail); }
  tr.fail { background: #fef5f4; }
  .charts { display: grid; gap: 16px;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }
  .chart-card { background: var(--card); border: 1px solid var(--border);
                border-radius: 6px; padding: 6px; overflow: hidden; }
  .chart-card svg { width: 100%; height: auto; display: block; }
  footer { color: var(--fg-soft); font-size: 12px; margin-top: 28px;
           text-align: center; }
  .stats td { font-variant-numeric: tabular-nums; }
  .stats td:first-child { font-weight: 600; width: 80px; }
</style>
</head>
<body>

<h1>{{ title }}</h1>
<div class="sub">{{ subtitle }}</div>

<h2>Summary</h2>
<div class="cards">
  <div class="card"><div class="label">files</div>
       <div class="value">{{ n_total }}</div></div>
  <div class="card pass"><div class="label">pass</div>
       <div class="value">{{ n_pass }}</div></div>
  <div class="card fail"><div class="label">fail</div>
       <div class="value">{{ n_fail }}</div></div>
  <div class="card"><div class="label">pass rate</div>
       <div class="value">{{ pass_rate }}%</div></div>
  <div class="card"><div class="label">avg CR</div>
       <div class="value">{{ avg_cr }}:1</div></div>
  <div class="card"><div class="label">total wall</div>
       <div class="value">{{ wall }}</div></div>
</div>

<h2>Quality distribution{{ ' (LQS Level ' + level + ')' if level else '' }}</h2>
<table class="stats">
  <thead><tr><th></th><th class="num">mean</th><th class="num">p5</th>
            <th class="num">p50</th><th class="num">p95</th>
            <th class="num">min</th><th class="num">max</th></tr></thead>
  <tbody>
    <tr><td>PRD %</td>
        <td class="num">{{ prd.mean }}</td><td class="num">{{ prd.p5 }}</td>
        <td class="num">{{ prd.p50 }}</td><td class="num">{{ prd.p95 }}</td>
        <td class="num">{{ prd.min }}</td><td class="num">{{ prd.max }}</td></tr>
    <tr><td>Pearson R</td>
        <td class="num">{{ r.mean }}</td><td class="num">{{ r.p5 }}</td>
        <td class="num">{{ r.p50 }}</td><td class="num">{{ r.p95 }}</td>
        <td class="num">{{ r.min }}</td><td class="num">{{ r.max }}</td></tr>
    <tr><td>CR</td>
        <td class="num">{{ cr.mean }}</td><td class="num">{{ cr.p5 }}</td>
        <td class="num">{{ cr.p50 }}</td><td class="num">{{ cr.p95 }}</td>
        <td class="num">{{ cr.min }}</td><td class="num">{{ cr.max }}</td></tr>
  </tbody>
</table>

{% if has_quality %}
<h2>Charts</h2>
<div class="charts">
  <div class="chart-card">{{ chart_prd | safe }}</div>
  <div class="chart-card">{{ chart_r | safe }}</div>
  <div class="chart-card">{{ chart_cr | safe }}</div>
  <div class="chart-card">{{ chart_scatter | safe }}</div>
</div>
{% endif %}

{% if top_failures %}
<h2>Top failures</h2>
<table>
  <thead><tr><th>file</th><th class="num">PRD %</th>
            <th class="num">R</th><th class="num">CR</th>
            <th>error</th></tr></thead>
  <tbody>
  {% for r in top_failures %}
    <tr class="fail">
      <td>{{ r.input_path }}</td>
      <td class="num">{% if r.prd is not none %}{{ '%.2f' % r.prd }}{% endif %}</td>
      <td class="num">{% if r.pearson_r is not none %}{{ '%.4f' % r.pearson_r }}{% endif %}</td>
      <td class="num">{% if r.cr %}{{ '%.1f' % r.cr }}{% endif %}</td>
      <td>{{ r.error or 'LQS bound exceeded' }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<h2>All files ({{ n_total }})</h2>
<table id="all-files">
  <thead><tr>
    <th>file</th><th>status</th>
    <th class="num">raw</th><th class="num">compressed</th>
    <th class="num">CR</th><th class="num">PRD %</th>
    <th class="num">R</th><th class="num">ms</th>
  </tr></thead>
  <tbody>
  {% for r in results %}
    <tr class="{{ 'fail' if r.status == 'failed' or r.lqs_pass is false else 'pass' }}">
      <td>{{ r.input_path }}</td>
      <td>{{ r.status }}</td>
      <td class="num">{{ human(r.raw_bytes) }}</td>
      <td class="num">{{ human(r.compressed_bytes) }}</td>
      <td class="num">{% if r.cr %}{{ '%.1f' % r.cr }}{% endif %}</td>
      <td class="num">{% if r.prd is not none %}{{ '%.2f' % r.prd }}{% endif %}</td>
      <td class="num">{% if r.pearson_r is not none %}{{ '%.4f' % r.pearson_r }}{% endif %}</td>
      <td class="num">{{ '%.0f' % r.duration_ms }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<footer>
  Generated by LamQuant {{ version }} on {{ generated }}.
</footer>

<script>
// Lightweight column sort — click a TH to sort, click again to reverse.
document.querySelectorAll('table#all-files thead th').forEach((th, idx) => {
  let asc = true;
  th.addEventListener('click', () => {
    const tbody = th.closest('table').querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const numeric = th.classList.contains('num');
    rows.sort((a, b) => {
      const av = a.children[idx].textContent.trim();
      const bv = b.children[idx].textContent.trim();
      if (numeric) return (parseFloat(av) || 0) - (parseFloat(bv) || 0);
      return av.localeCompare(bv);
    });
    if (!asc) rows.reverse();
    asc = !asc;
    rows.forEach(r => tbody.appendChild(r));
  });
});
</script>

</body>
</html>
"""


# ============================================================
# Main entry
# ============================================================

def write_html_report(report: BatchReport, path,
                      *, level: Optional[str] = None,
                      title: Optional[str] = None) -> Path:
    """Write a self-contained HTML dashboard for a BatchReport.

    Args:
        report: BatchReport from validate_batch / verify_batch / etc.
        path:   Output .html path.
        level:  LQS level the report is graded against (for thresholds).
        title:  Optional dashboard title (default: derived from report).

    Returns: the written Path.
    """
    from datetime import datetime
    try:
        from jinja2 import Environment, BaseLoader
    except ImportError:
        raise ImportError(
            "HTML reports require jinja2. Install with: pip install jinja2") from None

    path = Path(path)

    # ---- Aggregate stats ----
    s = report.summary()
    n = s['total']
    n_pass = sum(1 for r in report.results if r.lqs_pass) if level else s['success']
    n_fail = (n - n_pass) if level else s['failed']
    pass_rate = round(100 * n_pass / n, 1) if n else 0.0

    prd_vals = [r.prd for r in report.results if r.prd is not None]
    r_vals = [r.pearson_r for r in report.results if r.pearson_r is not None]
    cr_vals = [r.cr for r in report.results if r.cr]

    has_quality = bool(prd_vals or r_vals)

    # ---- LQS thresholds (must match batch._validate_one) ----
    thresh = {'A': (25.0, 0.85), 'M': (15.0, 0.92),
              'C': (9.0, 0.96), 'L': (0.001, 0.9999)}
    max_prd, min_r = thresh.get(level, (9.0, 0.96))
    min_cr_for_scatter = 50.0   # cosmetic only — drawn as a visual band

    # ---- Charts (if we have quality data) ----
    if has_quality:
        chart_prd = _svg_histogram(prd_vals, 'PRD distribution', 'PRD %',
                                    threshold=max_prd, threshold_label=f'level {level} max ({max_prd}%)')
        chart_r = _svg_histogram(r_vals, 'Pearson R distribution', 'R',
                                  threshold=min_r, threshold_label=f'level {level} min ({min_r})')
        chart_cr = _svg_histogram(cr_vals, 'Compression ratio distribution', 'CR (×)')
        chart_scatter = _svg_scatter_cr_vs_r(report.results, min_r, min_cr_for_scatter)
    else:
        chart_prd = chart_r = chart_cr = chart_scatter = ''

    # ---- Top failures (max 10, sorted by PRD descending) ----
    failures = [r for r in report.results
                if r.status == 'failed' or r.lqs_pass is False]
    top_failures = sorted(failures,
                          key=lambda r: -(r.prd if r.prd is not None else -1))[:10]

    # ---- Render ----
    env = Environment(loader=BaseLoader(), autoescape=True)
    env.globals['human'] = _human   # already defined in batch.py
    tpl = env.from_string(_TEMPLATE)

    from lamquant_codec import __version__

    html = tpl.render(
        title=title or 'LamQuant Quality Report',
        subtitle=f'{n} files validated against LQS Level {level or "—"} '
                 f'· generated by LamQuant',
        n_total=n, n_pass=n_pass, n_fail=n_fail,
        pass_rate=pass_rate,
        avg_cr=f'{s["avg_cr"]:.1f}',
        wall=f'{s["wall_seconds"]:.1f} s',
        level=level,
        prd=_summary_stats(prd_vals, '{:.2f}'),
        r=_summary_stats(r_vals, '{:.4f}'),
        cr=_summary_stats(cr_vals, '{:.1f}'),
        has_quality=has_quality,
        chart_prd=chart_prd, chart_r=chart_r,
        chart_cr=chart_cr, chart_scatter=chart_scatter,
        top_failures=top_failures,
        results=report.results,
        version=__version__,
        generated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )
    path.write_text(html)
    return path


# Reuse the human-bytes formatter from batch.py.
def _human(n_bytes) -> str:
    if n_bytes is None or not n_bytes:
        return '—'
    n = float(n_bytes)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


__all__ = ['write_html_report']
