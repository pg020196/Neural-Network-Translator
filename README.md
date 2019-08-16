# Neural Network Translator

# 1 Set up environment

## 1.1 Required software

The following software components and dependencies are needed:

- GIT: https://git-scm.com/
- Anaconda Python 2019.07: https://www.anaconda.com/download/

## 1.2 Installationprocess

### 1.2.1 Install Anaconda Python 2019.07

### 1.2.2 Clone GIT repository

- Navigate to the desired target directory
- Open Git Bash:
- `git clone https://github.com/pg020196/Neural-Network-Translator.git`

### 1.2.3 Set up python environment with requirements.txt

- Start Anaconda Prompt and execute the following commands:

  `conda create -n nnt pip python=3.7`

  `conda update -n base -c defaults conda`

  `activate nnt`

  `python -m pip install --upgrade pip==19.2.1`

- Open Anaconda Navigator and install Jupyter Notebook in version **6.0.0**

- Open Anaconda Prompt and navigate to `setup\`

- Install NNT-Requirements: `pip install -r nnt_requirements.txt`

### 1.2.4 Set up python environment with environment.yml

- Create Anaconda environment from environment.yml
  `conda env create -f environment.yml`
- Activate new environment
  `activate nnt`

# 2 Components

## 2.1 Frameworks

- [ ] **Keras**
- [ ] TensorFlow
- [ ] Theano
- [ ] Caffe
- [ ] Microsoft Cognitive Toolkit
- [ ] PyTorch
- [ ] Apache mxnet

## 2.2 Activation Functions

- [ ] **ReLu**
- [ ] Tanh
- [ ] Sigmoid
- [ ] Softmax

## 2.3. Neural Network Structures

- [ ] **(Deep) Feed Forward**
- [ ] Recurrent

## 2.4. Layers

- [ ] **Dense**
- [ ] Convolutional

# 3 Working with Git

Clone Git-repository: `git clone https://github.com/pg020196/Neural-Network-Translator.git`

Show changed files: `git status`

Add modified files to your commit: `git add /Path/to/modified/file`

Perform commit: `git commit -m "Commit message with #Issue"`

Push commits to remote host: `git push`

Change to specific branch: `git checkout SprintXX`

Process to upload new changes:

1. `git pull`
2. `git status`
3. `git add`
4. `git status`
5. `git commit`
6. `git push`

Merge branches:

Checkout the branch in which you want to merge (`TARGETBRANCH` (Mostly master- or development-branch):

`git checkout master`

Pull desired branch from the online repositroy. This type of merge keeps the commit history:

`git pull https://github.com/pg020196/Neural-Network-Translator.git BRANCH_NAME`

Solve conflicts.

Commit the merge and verify the changes.

Push the merge to the github repository.

`git push origin TARGETBRANCH`

# 4 How to contribute

Contributing is not available at this stage.