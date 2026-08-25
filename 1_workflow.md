---
title: 💻 Work environment and submission workflow
short_title: 💻 Setup and workflow
subtitle: Class 2 — Thursday, August 27
---

In addition to theory 📖 this course has a strong ⚙️ practical component, with
hands-on exercises and a semester-long project. To follow the course effectively you
need a local development environment to interact with the course materials, and to
write and run code.

This class also sets up the workflow you will use to submit every homework: a pull
request to the course repository.

```{admonition} Local install, or the cloud
:class: hint

The information on this page refers to a *local install*, which lets you run course
code on your own machine. An alternative is to rely on *cloud based resources* such
as [Google Colab](https://colab.research.google.com), which have the development
environment already configured — but give you less control.
```

## Prerequisites

- You have a fairly modern computer: desktop or laptop, Windows or Mac or Linux
- You have administrative access to this computer (can install programs)

```{admonition} Software components
:class: note

1. Install the **Python** environment manager **Anaconda**
2. Install the version control system **Git** and a graphical user interface for it
3. Register at **GitHub** to have access to the course repository
4. Install a good **text editor**
```

## Python and Anaconda

The Anaconda distribution is an easy way to install:

- Python
- Jupyter Notebook
- Scientific programming libraries

[Anaconda Distribution](https://www.anaconda.com/download)

The course uses `numpy`, `scipy`, `matplotlib` and `sympy`. All of these ship with
Anaconda; no other frameworks are required.

## Git and Git GUI

- Git is the *command line* version control software
- A GUI makes Git a lot more practical

[Git](https://git-scm.com/) · [Git GUI applications](https://git-scm.com/downloads/guis)

Recommended options:

- [GitHub Desktop](https://desktop.github.com/)
- [SourceTree](https://www.sourcetreeapp.com/)
- [VS Code editor](https://code.visualstudio.com/) (has Git built in)

## GitHub

- Social coding network website
  - hosting code
  - version control and integrations
  - community of coders and open source projects
  - [benefits for students](https://github.com/education/students)
  - free hosting for static web pages — this book is one
- Register on [GitHub](https://github.com/join) if you have not yet
  - *be mindful about using your personal data when registering*
  - [apply for the Student Developer Pack](https://docs.github.com/en/education/explore-the-benefits-of-teaching-and-learning-with-github-education/github-education-for-students/apply-to-github-education-as-a-student)
- Course materials and all homework are distributed and collected through the course
  GitHub repository

```{admonition} Example
:class: tip

[Fedor's public repositories](https://github.com/fediskhakov)
```

## Text editor

- A good text editor is *invaluable* for editing source files
- Editing could be done in Jupyter or other default editors, but it is less convenient
- Essential for bigger coding projects — such as your course project
- Good options are:
  - [VS Code](https://code.visualstudio.com/)
  - [Sublime Text](https://www.sublimetext.com/) (paid)
  - [PyCharm](https://www.jetbrains.com/pycharm/) (full IDE)

## Jupyter notebooks

- An excellent way to present and discuss code
  - this entire course is taught using notebooks
- A good instrument to develop new ideas
  - especially together with coauthors
- Saved to disk as JSON files with multiple sections
  - text
  - math and formulas in LaTeX
  - code (different languages are possible)
  - output from the code

**Jupyter notebooks have limitations**

- NOT a good way to store developed code — use libraries (modules)
- NOT good for version control
  - changes in metadata are tracked
  - changes in output are tracked
  - merging changed files may break the JSON format
- Require additional tools to work well with Git
- We don't worry about this for most of the course, but you should for your project

## Homework submission workflow

Every homework in this course is submitted as a **pull request** to the course
repository. The mechanics are the same every time:

1. **Fork** the course repository to your own GitHub account (once, at the start of
   the semester)
2. **Clone** your fork to your machine
   `git clone <address copied from the GitHub page>`
3. **Branch** for each assignment: `git checkout -b hw1-yourname`
4. **Work** in your own folder — `submissions/yourname/hw1/` — so that submissions
   never collide with each other
5. **Commit** with a meaningful message: `git add . && git commit -m "HW1: inventory model solver"`
6. **Push** the branch to your fork: `git push origin hw1-yourname`
7. **Open a pull request** against the course repository before the deadline

Homework is discussed at the start of the class that follows it, with one student
presenting the solution at the board. The presenter rotates, so plan on presenting
several times over the semester — including code that does not work yet, which is
usually the more instructive case.

````{admonition} Practical task
:class: warning

1. Fork and clone the course repository
2. Create a new file in your own submission folder
3. Write and edit the new file using your text editor
4. Stage and commit the change
5. Observe the diff
6. Push to your fork and open a pull request

The first pull request of the semester is the student survey — no code required.
````

````{admonition} References and additional resources
:class: note

- QuantEcon page on setting up a local environment
  [link](https://python-programming.quantecon.org/getting_started.html)

- Workspace setup lecture from the *Foundations of Computational Economics* course
  [YouTube video](https://youtu.be/UrZnRv3_IUc)

- Simple guide to Git [link](https://rogerdudler.github.io/git-guide/)

- Full reference to Git [link](https://git-scm.com/doc)

- GitHub intro [30 min online course](https://education.github.com/experiences/intro_to_github)

- Understanding Markdown [20 min online course](https://education.github.com/experiences/understanding_markdown)
````
