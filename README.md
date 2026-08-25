# ECO 629 — Dynamic Structural Models

Lecture notes for ECO 629 *Studies in Quantitative Methods*, Stony Brook University,
Fall 2026. Taught by Fedor Iskhakov.

Built with [Jupyter Book 2](https://jupyterbook.org) / [MyST](https://mystmd.org).
Note that Jupyter Book 2 is a different tool from Jupyter Book 1: it is a thin
wrapper around the MyST engine, configured by `myst.yml` rather than
`_config.yml` + `_toc.yml`, and it runs on Node rather than Sphinx.

## Set up the conda environment

One-time setup, from this directory:

```bash
conda env create -f environment.yml
conda activate eco629
```

That installs Python, Jupyter Book 2, and the scientific stack used by the lecture
code. Node.js comes in automatically as a dependency of `jupyter-book` — you do not
need to install it yourself.

To update the environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

To remove it and start over:

```bash
conda env remove -n eco629
```

<details>
<summary>Using pip or uv instead of conda</summary>

`requirements.txt` holds the same dependencies for a plain virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pip build of `jupyter-book` downloads its own private copy of Node via
`nodeenv`, so this route also works without a system Node install.

</details>

## Build the book

With the environment activated:

```bash
jupyter book start          # live preview with hot reload at http://localhost:3000
jupyter book build --html   # static site written to _build/html
```

Add `--execute` to either command to run the code cells at build time:

```bash
jupyter book start --execute
jupyter book build --html --execute
```

Without `--execute` the code cells are rendered as source with no outputs, which is
much faster and is the right mode when you are editing text. Executed outputs are
cached, so only changed cells re-run.

Some cells in the algorithms chapters time their own code with `%timeit` over many
repetitions and take minutes to run. Reduce the `-n` / `-r` counts there if you want
quick executed previews.

`jupyter book` and `myst` are the same CLI — `myst start` and `jupyter book start`
do the same thing, so documentation at [mystmd.org](https://mystmd.org) applies
directly.

## Publishing

`.github/workflows/jb_compile_pages.yml` builds the book and deploys it to GitHub
Pages on every push to `main`, and on demand from the Actions tab. The site is
served at <https://dse.iskh.me> (custom domain, Pages source set to *GitHub
Actions*).

The workflow runs `jupyter book build --html --execute`, so the published pages
carry executed outputs and figures. The manual (`workflow_dispatch`) run exposes an
**execute** checkbox — untick it for a fast text-only rebuild.

Because the site sits at the root of a custom domain, no `BASE_URL` is set. If the
custom domain is ever dropped and the book is served from
`https://fediskhakov.github.io/sb_dse/`, set `BASE_URL: /sb_dse` on the build step,
otherwise every asset and internal link will point at the wrong prefix.

## Replacing the logo

The emblem is shown large above the left-hand menu rather than in the top bar.
The theme has no option for that, so it is done with custom CSS in
`_static/custom.css`, which paints the image as a background on
`.myst-primary-sidebar-nav::before`.

Both images are inlined into that CSS as data URIs, because the build renames static
files with a content hash and a plain `url(...)` path would break whenever the
image changes. So after replacing `_static/img/sbu_dse_emblem.png`, regenerate the
CSS block:

```bash
python tools/embed_logo.py                       # standard light and dark files
python tools/embed_logo.py light.png             # different light source
python tools/embed_logo.py light.png dark.png    # supply both explicitly
```

The script crops the artwork out of its white margins, scales it for retina
displays, and rewrites only the block below the `---- sidebar logo` marker —
anything you add above that marker is preserved.

**Light and dark variants.** The emblem is line art drawn on white, so its black
outlines disappear against the dark theme. The script therefore embeds two images
and swaps them with a `.dark` rule. The dark one comes from
`_static/img/sbu_dse_emblem_dark.png` if that file exists; otherwise it is derived
from the light artwork and saved there, so you can look at it and replace it with a
hand-made version whenever you like — delete the file to have it derived again.

The derivation inverts lightness in HSL while preserving hue, so black outlines turn
white but the bus stays blue rather than flipping to orange, and it drops the paper
background to transparency so the emblem sits on whatever the theme paints. Keep
`site.options.logo` and `logo_dark` in `myst.yml` pointed at the same two files, so
the social/preview metadata stays in step.

## Layout

| File | Role |
| :-- | :-- |
| `myst.yml` | project configuration and table of contents |
| `index.md` | course home page: outline, logistics, assessment |
| `1_*.md`, `2_*.md` | chapters, numbered by course week |
| `bibliography.md` | reading list, organized by course part |
| `references.bib` | BibTeX database for all citations |
| `_static/` | images, PDFs and custom CSS |
| `environment.yml` | conda environment specification |
| `requirements.txt` | the same dependencies for pip |
| `.github/workflows/` | GitHub Actions workflow that builds and publishes the site |

`_build/` is generated output and is not tracked in git.
