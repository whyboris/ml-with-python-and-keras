## Machine learning with Keras

Worked examples following _Deep Learning with Python_ by François Chollet
It's an amazing [book](https://www.manning.com/books/deep-learning-with-python) - likely the fastest way to get started with _machine learning_!

Source code from the book available on [GitHub](https://github.com/fchollet/deep-learning-with-python-notebooks)

### About

This repository is WIP. Feel free to reach out with questions via _Issues_

### How to get started

Coding is likely less scary than you think. Going from 0 knowledge to running the code above will take you less than 10 minutes, just follow the steps below!

Install requirements:

0. Install [git](https://git-scm.com/downloads)
    - this will allow you to copy all the code in this repository
1. Install [Python 3.6+](https://www.python.org/downloads/)
    - this will automatically install *pip* (_Pip Installs Python_ - a way to download [packages](https://pypi.org/))
2. Install [virtualenv](https://virtualenv.pypa.io/en/latest/installation/)
    - open your _terminal_ and run the command `pip install virtualenv` 

You'll now be using your *terminal*. On Mac or Linux you're ready to go. On Windows you'll want to use _git-bash_ (that was installed when you installed _git_). If you're more comfortable with setting things up, consider these alternative terminals: [Hyper](https://hyper.is/), [FluentTerminal](https://github.com/felixse/FluentTerminal) or [Terminus](https://github.com/Eugeny/terminus). On Mac or Linux, consider installing [oh my zsh](https://ohmyz.sh/) (requires you install 'zsh' first).

Open your _terminal_ and enter these commands:

0. `cd ~`
    - this will navigate to your "home" directory
    - please navigate using `cd` to a folder where you'd like to work
        - perhaps create a folder called "code" 
        - you can do that from the terminal with: `mkdir code` 
1. `git clone https://github.com/whyboris/ml-with-python-and-keras.git`
    - this will create a folder `ml-with-python-and-keras` inside the folder you are in copying this whole repository
2. `cd ml-with-python-and-keras`
    - this will enter the folder
3. `virtualenv venv`
    - this will create  a folder named "venv" inside your dir
    - having a virtual environment allows packages you instal to not interfere with other packages in other projects
4. `source venv/bin/activate`
    - this will activate the environment, allowing you to 
5. `pip install -r requirements.txt`
    - This installs all the packages listed in `requirements.txt`

You're ready to run any of the scripts above!

I recommend you also install [Visual Studio Code](https://code.visualstudio.com/) to work with code.

Open any of the above scripts to see what it does. Run them in your terminal, for example `python3 news.py` :+1:

As you spend more time in the terminal, many actions become too repetitive, so you can create shortcuts of commands (called *alias*) to save yourself time (and the hassle of remembering some of the longer ones). These need to be saved in a specific file on your computer, `.bashrc`, `.zshrc`, or somewhere else (depends on the terminal and settings you use). I recommend these:

- `alias py="python3"`
    - allows you to run scripts with `py script.py` rather than `python3 script.py`
- `alias activate="source venv/bin/activate"`
    - allows you to activate the environment with the command `activate` (see step 4 above).
