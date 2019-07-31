## Machine learning with Keras

Worked examples following _Deep Learning with Python_ by François Chollet
It's an amazing [book](https://www.manning.com/books/deep-learning-with-python) - likely the fastest way to get started with _machine learning_!

Source code from the book available on [GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks)

### About

This repository is WIP. Feel free to reach out with questions via _Issues_

### How to get started

Coding is likely less scary than you think. Getting everything above to work will just take some research and patience.

Install requirements:

0. Install [git](https://git-scm.com/downloads)
    - this will allow you to copy all the code in this repository
1. Install [Python 3.6+](https://www.python.org/downloads/)
    - this will automatically install *pip* (_Pip Installs Python_ - a way to download [packages](https://pypi.org/))
2. Install [virtualenv](https://virtualenv.pypa.io/en/latest/installation/)
    - open your _terminal_ and run the command `pip install virtualenv` 

Open your terminal and enter these commands:

0. `cd ~`
    - this will navigate to your "home" directory
    - please navigate using `cd` to a folder where you'd like to work
        - perhaps create a folder called "code" 
        - you can do that from the terminal with: `mkdir code` 
1. `git clone https://github.com/whyboris/ml-with-python-and-keras.git`
    - this will create a folder `ml-with-python-and-keras` inside the folder you are in
2. `cd ml-with-python-and-keras`
    - this will enter the folder
3. `virtualenv venv`
    - this will create  a folder named "venv" inside your dir
    - having a virtual environment allows packages you instal to not interfere with other packages in other projects
4. `source venv/bin/activate`
    - this will activate the environment, allowing you to 
5. `pip install -r requirements.txt`
    - This installs all the packages listed in `requirements.txt`

As you code more and more, many actions become too repetitive, so you can create short versions of commands (called *alias*) to save yourself time (and the hassle of remembering some of the longer ones). These need to be saved in `.bashrc`, `.zshrc`, or somewhere else (depends on the terminal and settings you use).
- `alias py="python3"`
    - allows you to run scripts with `py script.py` rather than `python3 script.py`
- `alias activate="source venv/bin/activate"`
    - allows you to activate the environment with the command `activate` (see step 4 above).
