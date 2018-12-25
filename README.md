# ssl-deep-srl
Code for "Mehta, S. V., Lee, J. Y., and Carbonell, J. (2018). Towards Semi-Supervised Learning for Deep Semantic Role Labeling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 4958-4963).

## Requirements
Python 3.6, PyTorch 0.4.1, AllenNLP v0.7.2

### Setting up a virtual environment

[Conda](https://conda.io/) can be used to set up a virtual environment
with Python 3.6 in which you can
sandbox dependencies required for our implementation:

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```
    conda create -n ssl-deep-srl python=3.6
    ```

3.  Activate the Conda environment.  (You will need to activate the Conda environment in each terminal in which you want to run our implementation).

    ```
    source activate ssl-deep-srl
    ```

### Setting up our environment

1. Visit http://pytorch.org/ and install the PyTorch 0.4.1 package for your system.

2.  Clone our repo:

    ```
    git clone git@github.com:sanketvmehta/ssl-deep-srl.git
    ```
#### Installing AllenNLP from source

1.  Change your directory to ``allennlp`` submodule present under the parent repo directory:

    ```
    cd ssl-deep-srl/allennlp
    ```

2. Install the necessary requirement by running 

   ```
   INSTALL_TEST_REQUIREMENTS=true scripts/install_requirements.sh
   ```

3. Once the requirements have been installed, run:

   ```
   pip install --editable .
   ```

4. Test AllenNLP installation by running:

   ```
   bin/allennlp test-install
   ``` 
That's it! You're now ready to reproduce our results.

### Citing

If you use our code in your research, please cite: [Towards Semi-Supervised Learning for Deep Semantic Role Labeling](https://www.aclweb.org/anthology/D18-1538)  

   ```
   @inproceedings{mehta2018towards,
    title={Towards Semi-Supervised Learning for Deep Semantic Role Labeling},
    author={Mehta, Sanket Vaibhav and Lee, Jay Yoon and Carbonell, Jaime},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
    pages={4958--4963},
    year={2018}
   }
   ```

