# ECO 629 — Dynamic Structural Models

Lecture notes for ECO 629 *Studies in Quantitative Methods*, Stony Brook University,
Fall 2026. Taught by Fedor Iskhakov.

**Read them at <https://dse.iskh.me>.**

How to solve dynamic structural models numerically, and how to estimate them from
data: single-agent discrete and continuous choice, micro-founded equilibrium models,
and dynamic games, each with the estimation toolkit that goes with it — NFXP, MPEC,
CCP, NPL and EPL.

## Reading the notes

The site is the primary format. It also offers downloads:

- the whole book as one PDF, linked from the home page
- each chapter as a PDF or as MyST Markdown, from the download menu on that chapter

This repository holds the source of the site. You do not need to clone it to follow
the course.

## Running the code

The code in the notes is plain Python — `numpy`, `scipy`, `matplotlib`, `sympy`, and
JupyterLab to run the notebooks. Clone this repository and set up the course
environment either way:

**uv** (the course default):

```bash
uv venv                                 # creates .venv with a suitable Python
uv pip install -r requirements.txt
uv run jupyter lab                      # no activation needed
```

**conda** ([Miniforge](https://conda-forge.org/download/)):

```bash
conda env create -f environment.yml
conda activate eco629
jupyter lab
```

These are the two supported ways; the *Work environment and submission workflow*
chapter covers both in full, including Windows.

You can then copy code out of any chapter into a notebook or script, or open the
chapter's `.md` file directly — the chapters are Jupyter notebooks in MyST Markdown
form and run as-is in Jupyter.

## Corrections

Found an error, a broken link, or something that does not run? Open an issue or a
pull request. Both are welcome.

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0). See [LICENSE](LICENSE).
