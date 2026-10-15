---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

The third edition of *Think Stats* is available now from [Bookshop.org](https://bookshop.org/a/98697/9781098190255) and [Amazon](https://amzn.to/42lmxwu) (those are affiliate links). If you find this book useful, please consider [supporting free, open educational materials](https://greenteapress.com/support/?book=ts3) from Green Tea Press.


# Preface


From the earliest history of statistics, there have been two ideas about what statistics is. In one view, it is a branch of mathematics with the goal of establishing a theoretical foundation for probability and statistical inference.
In another view, it is a set of tools and practices for working with data, answering questions, and making better decisions.
Many introductory classes in statistics are based on the first view.
This book is based on the second.


*Think Stats* is an introduction to practical methods for exploring and visualizing data, discovering relationships and trends, and communicating results. 
The organization of the book follows the process I use when I start working with a dataset:

- Importing and cleaning: Whatever format the data is in, it usually takes some time and effort to read the data, clean and transform it, and check that everything made it through the translation process intact.

- Single variable explorations: I usually start by examining one variable at a time, finding out what the variables mean, looking at distributions of the values, and choosing appropriate summary statistics.

- Pair-wise explorations: To identify possible relationships between variables, I look at tables and scatter plots, and compute correlations and linear fits.

- Multivariate analysis: If there are apparent relationships between variables, I use multiple regression to add control variables and investigate more complex relationships.

- Estimation and hypothesis testing: When reporting statistical results, it is important to answer three questions: How big is the effect? How much variability should we expect if we run the same measurement again? Is it plausible that the apparent effect is due to chance?

- Visualization: During exploration, visualization is an important tool for finding possible relationships and effects. Then if an apparent effect holds up to scrutiny, visualization is an effective way to communicate results.


This book takes a computational approach, which has several advantages over more mathematical treatments:

-  I present most ideas using Python code, rather than mathematical notation. In general, Python code is more readable -- also, because it is executable, the reader can run it and modify it to develop insight.

-  Each chapter includes exercises readers can do to check and solidify their learning. When you write programs, you express your understanding in code -- while you are debugging the program, you are also checking your understanding.

-  Some exercises involve experiments to test statistical behavior. For example, you can explore the Central Limit Theorem (CLT) by generating random samples and computing their sums. The resulting visualizations show why the CLT works and when it doesn't.

-  Some ideas that are hard to grasp mathematically are easy to understand by simulation. For example, we approximate p-values by running random simulations, which reinforces the meaning of hypothesis testing.

-  Because the book is based on a general-purpose programming language (Python), readers can import data from almost any source. They are not limited to datasets that have been cleaned and formatted for a particular statistical tool.


To demonstrate my approach to statistical analysis, the examples and exercises use data from several sources, including:

-  The National Survey of Family Growth (NSFG), conducted by the U.S. Centers for Disease Control and Prevention (CDC) to gather "information on family life, marriage and divorce, pregnancy, infertility, use of contraception, and men's and women's health." (See <https://www.cdc.gov/nchs/nsfg/index.htm>.)

-  The Behavioral Risk Factor Surveillance System (BRFSS), conducted by the National Center for Chronic Disease Prevention and Health Promotion to "track health conditions and risk behaviors in the United States." (See <http://cdc.gov/BRFSS/>.)

-  The Palmer Penguins, which includes measurements from a sample of penguins near Palmer  Station in Antarctica (see <https://allisonhorst.github.io/palmerpenguins/>).

-  Data from the U.S. Energy Information Administration (EIA) on electricity generation from renewable sources in the United States.

I am grateful to the people and agencies that collected this data and made it available -- and I hope that working with real data from a variety of domains makes the book more engaging for readers.


## What's New?

For this third edition, I started by moving the book into Jupyter notebooks.
This change has one immediate benefit -- you can read the text, run the code, and work on the exercises all in one place.
And the notebooks are designed to work on Google Colab, so you can get started without installing anything.

The move to notebooks has another benefit -- the code is more visible.
In the first two editions, some of the code was in the book and some was in supporting files available online.
In retrospect, it's clear that splitting the material in this way was not ideal, and it made the code more complicated than it needed to be.
In the third edition, I was able to simplify the code and make it more readable.

Since the last edition was published, I've developed a library called `empiricaldist` that provides objects that represent statistical distributions.
This library is more mature now, so the updated code makes better use of it.

When I started this project, NumPy and SciPy were not as widely used, and Pandas even less, so the original code used Python data structures like lists and dictionaries.
This edition uses arrays and Pandas structures extensively, and makes more use of functions these libraries provide.

The third edition covers the same topics as the original, in almost the same order, but the text is substantially revised.
Some of the examples are new; others are updated with new data.
I've developed new exercises, revised some of the old ones, and removed a few.
I think the updated exercises are better connected to the examples, and more interesting.

Since the first edition, this book has been based on the thesis that many ideas that are hard to explain with math are easier to explain with code.
In this edition, I have doubled down on this idea, to the point where there is almost no mathematical notation left.

Overall, I think these changes make *Think Stats* a better book.
I hope you like it!


## Using the code

The code and data used in this book are available from <https://github.com/AllenDowney/ThinkStats>, which is a Git repository on GitHub.
Git is a version control system that helps to keep track of the files that make up a project.
A collection of files under Git's control is called a **repository**.
GitHub is a hosting service that provides storage for Git repositories and a convenient web interface.

For each chapter in this book, the repository provides a Jupyter notebook, which is a document that contains the text, code, and the results of running the code.
You can use these notebooks to run the code and work on the exercises.

There are two ways you can run the notebooks.
By far the easier one is to use Colab, which is a service provided by Google where you can run the notebooks in a web browser without installing anything on your computer.
If you start from the *Think Stats* home page at <https://allendowney.github.io/ThinkStats/>, you will find links to the notebooks, including one that introduces Colab and Jupyter notebooks.


If you don't want to use Colab, you can download the notebooks and run them on your computer, but in that case you will have to install Python, Jupyter, and the libraries the book uses, including NumPy, SciPy, and StatsModels.
If you have experience installing software, setting up an environment where you can run the notebooks is not difficult.
But if you don't have that experience, your first attempt can be challenging, and sometimes frustrating.
In that case, it can be a barrier to getting the most out of this book.
If you want to learn about exploratory data analysis in Python, you don't want to spend your time and cognitive capacity on installing software!

So I strongly recommend that you run at least the first few chapters on Colab.
Then, if you want to set up your own environment, you can do it without interrupting your progress in the book.
And one last suggestion: if you have any problems installing software, take advantage of tools like ChatGPT -- they generally provide good guidance on these topics.


I wrote this book assuming that the reader is familiar with core Python, including object-oriented features.
If you are familiar with NumPy and Pandas, that will help, but it's not necessary -- I'll explain what you need to know.
I assume that the reader knows basic mathematics, including logarithms, for example, and summations.
You don't need to know linear algebra or calculus.
There is one place where I mention derivatives and integrals, but if you are not familiar with those concepts, they are entirely optional.
Finally, I don't assume you know anything about statistics.


## Acknowledgments

Thanks to the readers who contributed corrections and suggestion to previous editions of this book, and to the students at Olin College who suffered through the rougher drafts.

Many thanks to the technical reviewers of this edition: Zachary del Rosario, Jerzy Wieczorek, Thomas Nield, Walter Paczkowski, and Peter Bruce.

And thank you to everyone at O'Reilly Media, especially editors Sara Hunter and Aaron Black, and ...TODO

<!-- #region tags=["remove-print"] -->
[Think Stats: Exploratory Data Analysis in Python, 3rd Edition](https://allendowney.github.io/ThinkStats/index.html)

Copyright 2024 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->
