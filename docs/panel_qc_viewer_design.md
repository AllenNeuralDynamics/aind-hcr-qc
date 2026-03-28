# Panel QC Viewer — System Design

## Goal

A browser-based dashboard that lets scientists browse, view, and compare QC
plots stored in S3 (`aind-scratch-data/ctl/hcr/qc/`) without touching raw
data or running any pipeline code.  Deployed as a persistent internal server.

---

## Decisions

| # | Decision |
|---|----------|
| 1 | **Two modes**: single-mouse (all plot types for one mouse) and comparison (one plot type across multiple mice) |
| 2 | **Show all known plot types**; indicate missing ones with a visual badge rather than hiding them |
| 3 | **Curated metadata summary** shown inline; full sidecar available via a collapsible "Details" panel |
| 4 | **Internal server** deployment (Docker container, persistent) |
| 5 | **PNG and PDF** both served |

---

## Architecture Overview

```
S3 (aind-scratch-data)
  ctl/hcr/qc/
    {mouse_id}/
      {plot_type}.png
      {plot_type}.pdf    (optional)
      {plot_type}.json   ← sidecar metadata
        │
        │  s3_qc utilities (list_plots, load_plot_metadata, get_plot_bytes)
        ▼
Panel Server  (panel serve, persistent)
  ├── catalog.py      TTL-cached S3 catalog
  ├── image_cache.py  LRU in-memory bytes cache
  └── app.py          Two-tab UI
        │  HTTP / WebSocket
        ▼
Browser (internal network)
```

No database.  S3 is the single source of truth.  The catalog is refreshed on
demand or on a TTL.

---

## UI — Two-tab layout

### Tab 1 · Single Mouse

```
┌─────────────────────────────────────────────────────────────────┐
│  HCR QC Viewer          [Single Mouse]  [Compare Mice]   [↺]   │
├─────────────────┬───────────────────────────────────────────────┤
│ Mouse           │  Plot grid (wraps)                            │
│ ┌─────────────┐ │  ┌─────────────┐ ┌─────────────┐             │
│ │ ● 782149    │ │  │ intensity_  │ │ spot_count  │             │
│ │   783551    │ │  │ violins ✓   │ │  ⚠ missing  │             │
│ │   785054    │ │  │ [PNG] [PDF] │ │             │             │
│ └─────────────┘ │  └─────────────┘ └─────────────┘             │
│                 │                                               │
│ Format          │  ── selected plot ──────────────────────────  │
│ ○ PNG  ○ PDF    │  [full-width image]                           │
│                 │                                               │
│                 │  ▸ created 2026-03-28 · thresh=25 · v0.6.1   │
│                 │  ▼ Details  (collapsible JSON)                │
└─────────────────┴───────────────────────────────────────────────┘
```

- Mouse selector: `pn.widgets.Select` (single)
- Plot grid: one card per known plot type; card shows plot type name, a ✓/⚠
  badge, and PNG/PDF download links if the file exists
- Clicking a card loads the full-size image below
- Metadata strip: `created_at`, `aind_hcr_qc_version`, key `plot_kwargs`
  always visible; `pn.Card(collapsed=True)` below it exposes the full sidecar

### Tab 2 · Compare Mice

```
┌─────────────────────────────────────────────────────────────────┐
│  HCR QC Viewer          [Single Mouse]  [Compare Mice]   [↺]   │
├─────────────────┬───────────────────────────────────────────────┤
│ Plot type       │  Layout: ( ↔ Row )  ( ↕ Stack )              │
│ ┌─────────────┐ │  [x] Hide missing plots                       │
│ │●intensity_  │ │                                               │
│ │  violins    │ │  ── Row layout (hide missing = off) ────────  │
│ │  spot_count │ │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│ └─────────────┘ │  │ 782149   │ │ 783551   │ │ 785054   │ ──► │
│                 │  │ [image]  │ │ [image]  │ │ ⚠ missing│     │
│ Mice            │  └──────────┘ └──────────┘ └──────────┘     │
│ ┌─────────────┐ │                                               │
│ │ ✓ 782149    │ │  ── Row layout (hide missing = on) ─────────  │
│ │ ✓ 783551    │ │  ┌──────────┐ ┌──────────┐                   │
│ │ ✓ 785054    │ │  │ 782149   │ │ 783551   │                   │
│ └─────────────┘ │  │ [image]  │ │ [image]  │                   │
│                 │  └──────────┘ └──────────┘                   │
└─────────────────┴───────────────────────────────────────────────┘
```

- **Layout toggle**: `pn.widgets.RadioButtonGroup(options=["↔ Row", "↕ Stack"])`
  — switches between a `pn.FlexBox` (horizontal, scrollable) and a
  `pn.Column` (vertical, full-width images)
- **Hide missing**: `pn.widgets.Checkbox(name="Hide missing plots")` — when
  checked, mice that lack the selected plot type are removed from both the
  image area and the metadata table entirely; when unchecked, they show as
  grey placeholders
- **Row layout**: fixed-width cards (~350 px each) in a horizontally
  scrollable `pn.FlexBox`; good for quick side-by-side scanning of many mice
- **Stack layout**: each mouse gets a full-width image row with its ID as a
  header label; good for detailed inspection of individual plots and easier
  to scroll on a tall monitor
- In both layouts, missing plots render as a grey placeholder card (unless
  hidden)
- The metadata comparison table is always shown below, regardless of layout

---

## Components

### `catalog.py`

```python
load_catalog(bucket) → pd.DataFrame
    # columns: mouse_id, plot_type, s3_key, has_pdf
    # TTL-cached (5 min), bust with refresh()

known_plot_types(catalog) → list[str]
    # union of all plot_types seen across all mice

mice_for_plot_type(catalog, plot_type) → list[str]
```

### `image_cache.py`

```python
get_plot_bytes(bucket, mouse_id, plot_type, fmt="png") → bytes | None
    # LRU cache (e.g. 50 entries); returns None if object missing
    # fmt: "png" or "pdf"
```

This replaces `download_plot()` for the server context — no local disk writes,
bytes go straight to `pn.pane.PNG` or a download button.

### `app.py`

Top-level `pn.Tabs([single_mouse_tab, compare_tab])` wired with `pn.bind()`.
A global `[↺ Refresh]` button calls `catalog.refresh()`.

---

## Missing-plot indicators

The complete set of plot types is derived from `known_plot_types(catalog)` —
the union across all mice.  For any mouse that lacks a given type:

- **Single-mouse grid**: card is greyed out, badge shows ⚠, no image/links
- **Comparison row**: placeholder card with mouse ID + "Not generated" text
- **Comparison table**: row cells are `—`

This makes coverage gaps immediately visible without hiding anything.

---

## Format handling (PNG / PDF)

`list_plots()` already returns `s3_key` for PNGs.  The catalog layer will also
check for a corresponding `.pdf` key and record `has_pdf: bool`.

- Single-mouse view: `[PNG]` and `[PDF]` download buttons in each card;
  PDF rendered in an `<iframe>` or offered as a direct download link
- Compare view: PNG only (side-by-side PDF rendering is awkward in-browser)

---

## Serving

```bash
panel serve src/aind_hcr_qc/viz_server/app.py \
    --address 0.0.0.0 --port 5006 \
    --allow-websocket-origin "qc-viewer.internal.example.org"
```

Deployed as a Docker container behind an internal reverse proxy (nginx/traefik).
No application-level auth in v1 — relies on network access control.  Can add
`--basic-auth` or OAuth via Panel's built-in support later if needed.

---

## File layout

```
src/aind_hcr_qc/
    viz_server/
        __init__.py
        app.py            ← pn.Tabs entrypoint, global widgets
        catalog.py        ← load_catalog(), known_plot_types(), TTL cache
        image_cache.py    ← get_plot_bytes(), LRU cache
        tabs/
            single_mouse.py   ← Tab 1 layout + callbacks
            compare.py        ← Tab 2 layout + callbacks
```

---

## Session State & Multi-user Concurrency

### How Panel handles concurrency

`panel serve` creates an **independent Python session per browser tab** by
default.  Widget state (selected mouse, plot type, format) is fully isolated
between users with no extra work needed.  The viewer is **read-only against
S3**, so there are no write races and no user conflict possible.

### Server-scoped vs. session-scoped state

| State | Scope | Rationale |
|-------|-------|-----------|
| Catalog (`list_plots` result) | **Server** (shared) | Read-only; sharing reduces S3 list calls for all users |
| Image/PDF bytes cache | **Server** (shared LRU) | Read-only; sharing avoids re-fetching the same plot for every new tab |
| Widget selections | **Session** (automatic) | Panel default — fully isolated per tab |

**Thread safety**: the shared catalog and image caches must be wrapped with a
`threading.RLock` (or `cachetools.LRUCache` + `RLock`) since Panel may call
into them from concurrent threads.  The catalog `refresh()` path should hold
the lock during the S3 list call to prevent duplicate refreshes.

### Startup flags

```bash
panel serve src/aind_hcr_qc/viz_server/app.py \
    --address 0.0.0.0 --port 5006 \
    --allow-websocket-origin "qc-viewer.internal.example.org" \
    --num-threads 4
```

`--num-threads 4` enables Panel's thread pool so concurrent S3 fetches don't
block each other.

---

## Remaining open questions

1. **Catalog TTL** — 5 min hardcoded or `QC_CATALOG_TTL_SECONDS` env var?

2. **PDF serving** — presigned S3 URL (simpler, URL visible in browser) or
   proxied through Panel (cleaner for internal use, no S3 URL exposure)?

3. **Comparison max mice** — suggest ~8 before layout breaks with horizontal
   scroll; does that feel right?
