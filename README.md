# larsenMRI
## Multivariate Larsen to process MRI images
The goal of this project is to implement a multivariate regressive model using the larsen algorithm to process MRI images.
This project is based on the below paper.
Inferring Brain Dyamics Using Granger Causality On FMRI Data by Guillermo A. Cecchi, Rahul Garg and A. Ravishankar Rao.
Link:http://www.cse.iitd.ernet.in/%7Erahulgarg/Publications/2008/CGR08.isbi.pdf
## Brief model of the Cuda Implementation
Based on the CPU implementation of the FARM code and larsen code from the github repo: https://github.com/DushyantSahoo/granger
We maintatin a buffer which stores all the current executing model‘s index. Each element in the buffer can be visualized as a worker which executes the larsen code for that respective model independently of other workers. 
After the worker completes the larsen algorithm for that respective model it removes the model saves it and adds a new incomplete model to this index and continues executing independently. 
## The model
The buffer size is determined by calculating remaining memory in the gpu and setting the buffer size such that 90% of the remaining memory is filled. Brief model flow is shown below:
N is the buffer size.
![img link](https://s19.postimg.io/kjwigt82b/larsen_MRIGithub_Repo1.png)
## Inside the worker
Inside each worker the standard larsen code from the earlier mentioned repo gets executed. Some of the important codes executing are:
Finding correlation and residuals.
Finding the gram matrix and its inverse parallely.
The gram inversion is done using the parallelized Gauss Jordan Matrix Inversion algorithm.
Moving towards OLS solution.
Adding and removing lasso variables.
Updating beta, residuals, etc.
For further information about the kernels the comments in the code explain more clearly.
## The code
The *.cu file is the main file and uses FileProc.h and larsProc.h.
FileProc.h cotains list of file processing functions and uses FileProcKernel.h.
LarsProc.h is the core larsen file and uses LarsProcKernel.h.
The *Kernel.h files are files containing the cuda kernels.
## Using the code
* Cuda Toolkit >= 6 necessary.
* Installing the cuda toolkit in linux: sudo apt-get install nvidia-cuda-toolkit.
* Ensure niftiread is installed with matlab and matlab command runs from the terminal.
* Create datanii, results and datatxt directories.
* Place all the nii or nii.gz files inside datanii. The datanii folder can have sub folders.
* Run: bash niiprocess.sh 6000 (6000 is mri read threshold)
* Each nii file will have generated txt files containing the flat matrix within the datatxt folder. The txt file will contain the shape of the matrix in the first line followed by the entire matrix data.
* Run: bash larsen.sh 512 8 0 0 0.25 0 0 (look inside the script for argument meaning)
* This will generate results and move it to results directories.
