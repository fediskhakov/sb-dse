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

The code in the notes is plain Python — `numpy`, `scipy`, `matplotlib`, `sympy`. To
run it locally, clone this repository and create the course environment:

```bash
conda env create -f environment.yml
conda activate eco629
```

Or, with pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You can then copy code out of any chapter into a notebook or script, or open the
chapter's `.md` file directly — the chapters are Jupyter notebooks in MyST Markdown
form and run as-is in Jupyter.

Setting up a working environment is covered in the *Work environment and submission
workflow* chapter, which is where to start if any of the above is unfamiliar.

## Corrections

Found an error, a broken link, or something that does not run? Open an issue or a
pull request. Both are welcome.

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0). See [LICENSE](LICENSE).
