# Introduction

![phone](/images/undraw_phone_no_bg.png)

This book provides a pipeline for data analysis, data cleaning, and
feature extraction that can be applied to a range of smartphone and wearable
datasets. It is built from a [GitHub repository](https://github.com/MentalHealthMission/data-book), which contains all the code
for this pipeline and is designed to be converted into a Jupyter Book once the
pipeline is complete, with each Jupyter notebook becoming a `chapter` of the
book that records the data analysis results, code used, and decisions made for
one specific type of data.

The repository includes a {doc}`general template <general_template>` which gives a step-by-step
method for processing the raw data, ranging from data analysis to feature
extraction. The data analysis includes three main steps that are helpful for all
data types, as well as additional steps that will be useful for certain types of
data. The cleaning and feature extraction stage includes a function to create
`minutely`, `hourly` or `daily` features from the raw data and also to save a
cleaned version of the raw data. For all data types, the features produced
include `metadata features` that describe the quality/quantity of the data for
that interval. These can be useful either as a direct input to a machine
learning model trained on the data, or to determine whether or not each interval
should be classified as `missing data` during subsequent data processing.

In addition to the {doc}`general template <general_template>`, three other
templates are provided for specific types of data: {doc}`step count <steps_specific_template>`,
{doc}`sleep <sleep_specific_template>`, and
{doc}`heart rate <heart_rate_specific_template>`. These help to illustrate how the functions
provided can be applied (and sometimes adjusted) to different data types, and
suggest additional analyses that may be useful for each data type. We have tried
to make these templates as general as possible within these data types, but
further tailoring may be required for specific datasets.

Each of the templates is given as a Jupyter notebook which is stored under the
`content` folder. These notebooks call functions that are defined in the python
scripts (`.py` files in the `src` folder). The other files in the repository are
all required for constructing of a Jupyter
Book.

```{tip}
If you want to publish the Jupyter Book locally, you can use the command `jupyter-book start` and open the book in your local browser. For more information, including on how to generate static *html* pages or *pdfs*, please check their website: https://jupyterbook.org/
```

The {doc}`general template <general_template>` gives details on the data analysis, data cleaning, and
feature extraction functions used, so should be read first. The data type
specific templates can then be used as guides to build pipelines for those
specific types of data. Although specific functions are currently given for only
three different data types, the {doc}`general template <general_template>` can be tailored to a wide range of data types,
possibly with some additional specialized analyses needed in some cases.
