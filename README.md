# Gloss
Local Subspace-Based Outlier Detection using Global Neighbourhoods


---

The Gloss algorithm is implemented in loop.py as an extention of the LoOP algorithm.
Download the files and datasets and place them in a folder of your choice.
To run the same experiments as shown in the paper toggle the synthetic or real world data in the file `experiments.py` at the bottom.

Then run: 
  
    python experiments.py 

for the real world datasets.  
And: 

    mpiexec -n 5 python experiments.py

for the synthetic data experiments. MPI is used to speed up the experiments.  
For these tests to run several python libraries need to be installed. Check the dependencies inside the python files, the most important are `numpy`, `matplotlib` and `seaborn`.

After the experiments you can run `generate_tables.py` to view the results in a latex format.