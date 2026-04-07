## [Supplementary materials for manuscript ''Neural Control: Adjoint Learning Through Equilibrium Constraints'']()

This repository contains supplementary materials for the anonymous manuscript **“Neural Control: Adjoint Learning Through Equilibrium Constraints”**. 
The supplementary materials are organized as follows:
- **Validation on a learned DEQ model**, including data collection videos and result plots.
- **iCEM baseline results**, including comparison plots and tables.
- **Videos of the robotic experiments** shown in Fig. 5.

# Anonymous Supplementary Materials for ICML Rebuttal

This anonymous website provides supplementary materials referenced in the rebuttal for the submission **“Neural Control: Adjoint Learning Through Equilibrium Constraints.”**

The materials are organized to directly address reviewer questions regarding:
- validation on a learned DEQ-style equilibrium model,
- comparison with a stronger modern derivative-free baseline (iCEM),
- and access to supplementary videos and plots.

All contents are provided in anonymized form for review purposes only.

---

## 1. Validation on a learned DEQ-style equilibrium model

This section provides supplementary materials for the additional validation experiment based on a learned DEQ-style equilibrium model.

### Overview
We collect force–strain measurements from a slinky under one-end actuation and use these data to train a neural energy model \(E_\theta(\varepsilon)\). The resulting equilibrium state \(\varepsilon^\star\) under control input \(z\) is defined implicitly by

$$G(\varepsilon^\star, z; \theta) = F_\theta(\varepsilon^\star) - z = 0,
\qquad
F_\theta = \partial E_\theta / \partial \varepsilon.
$$

The forward equilibrium is solved to convergence, and training uses implicit differentiation / IFT without unrolling, in the same spirit as DEQ methods.

After training, this learned implicit model is frozen and used as the forward model for Neural Control, which optimizes a force trajectory \(z(\lambda)\) so that the resulting equilibrium strain trajectory tracks the target

$$
\varepsilon^*(\lambda) = 0.05\sin(2\pi\lambda) + 0.05, \qquad \lambda \in [0,1].
$$

This experiment is intended to provide a concrete validation on a learned implicit / DEQ-style model beyond the original mechanics simulator.

### Data collection
A video of the force–strain data collection process is shown below.

<p align="center">
  <img src="DEQ_relevant/video/data_collection.gif" alt="Training data collection for the slinky force-strain dataset">
  <br>
  <em>Figure 1. Training data collection for the force–strain dataset of a slinky through robotic manipulation.</em>
</p>

Original video: [DEQ_relevant/video/data_collection.mp4](DEQ_relevant/video/data_collection.mp4)

### DEQ model training
The training curve of the learned DEQ-style equilibrium model is shown below.

<p align="center">
  <img src="DEQ_relevant/plots/training_loss.png" alt="Training of DEQ model">
  <br>
  <em>Figure 2. Training curve of the DEQ-style equilibrium model.</em>
</p>

### DEQ model inference
The learned force–strain relation and its agreement with experimental data are shown below.

<p align="center">
  <img src="DEQ_relevant/plots/DEQ_model.png" alt="Inference of DEQ model">
  <br>
  <em>Figure 3. Inference results of the learned DEQ-style equilibrium model compared with experimental data.</em>
</p>

### Neural Control on top of the learned DEQ model
We then apply Neural Control to optimize the force input so that the equilibrium strain follows the sinusoidal target above.

<p align="center">
  <img src="DEQ_relevant/plots/training_plot.png" alt="Learning of neural control on DEQ model">
  <br>
  <em>Figure 4. Optimization process of Neural Control on the learned DEQ-style equilibrium model.</em>
</p>

The final result shows near-perfect sinusoidal strain tracking, with segment losses on the order of \(10^{-7}\)–\(10^{-8}\).

## 2. iCEM baseline results

This section provides additional baseline comparison results with [iCEM](https://proceedings.mlr.press/v155/pinneri21a).

The plots below compare iCEM with our Neural Control method (Adjoint + RHC) on all three tasks. The results show that iCEM struggles on these challenging deformable manipulation problems, while Neural Control achieves substantially better performance.

<p align="center">
  <img src="iCEM_relevant/plots/plot_task1.png" alt="iCEM comparison on task 1">
  <br>
  <em>Figure 4. Comparison between iCEM and Neural Control on Task 1.</em>
</p>

<p align="center">
  <img src="iCEM_relevant/plots/plot_task2.png" alt="iCEM comparison on task 2">
  <br>
  <em>Figure 5. Comparison between iCEM and Neural Control on Task 2.</em>
</p>

<p align="center">
  <img src="iCEM_relevant/plots/plot_task3.png" alt="iCEM comparison on task 3">
  <br>
  <em>Figure 6. Comparison between iCEM and Neural Control on Task 3.</em>
</p>

The quantitative results, together with the corresponding time complexity and memory efficiency, are summarized in the table below.


## Quantitative comparison
| Method | Time / update | Memory / update | Task 1 Time (s) ↓ | Task 1 Best loss ↓ | Task 2 Time (s) ↓ | Task 2 Best loss ↓ | Task 3 Time (s) ↓ | Task 3 Best loss ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iCEM | O(P K C_eq) | O(P n_Theta) | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` |
| **Adjoint + RHC** | O(H C_eq + H C_lin) ≈ O(H C_eq) | O(H (n_x + n_z) + n_Theta) | **`[...]`** | **`[...]`** | **`[...]`** | **`[...]`** | **`[...]`** | **`[...]`** |








Install the following C++ dependencies:
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
  - Eigen is used for various linear algebra operations.
  - IMC is built with Eigen version 3.4.0 which can be downloaded [here](https://gitlab.com/libeigen/eigen/-/releases/3.4.0). After downloading the source code, install through cmake as follows.
    ```bash
    cd eigen-3.4.0 && mkdir build && cd build
    cmake ..
    sudo make install
    ```
- [SymEngine](https://github.com/symengine/symengine)
  - SymEngine is used for symbolic differentiation and function generation.
  - Before installing SymEngine, LLVM is required which can be installed through apt.
    ```bash
    sudo apt-get install llvm
    ```
  - Afterwards, install SymEngine from source using the following commands.
    ```bash
    git clone https://github.com/symengine/symengine    
    cd symengine && mkdir build && cd build
    cmake -DWITH_LLVM=on ..
    make -j4
    sudo make install
    ```
- [Intel oneAPI Math Kernel Library (oneMKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&distributions=webdownload&options=online)
  - Necessary for access to Pardiso, which is used as a sparse matrix solver.
  - Intel MKL is also used as the BLAS / LAPACK backend for Eigen.
  - If you are using Linux, follow the below steps. Otherwise, click the link above for your OS.
    ```bash
    cd /tmp
    wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18483/l_onemkl_p_2022.0.2.136.sh
    
    # This runs an installer, simply follow the instructions.
    sudo sh ./l_onemkl_p_2022.0.2.136.sh
    ```
  - Add the following to your .bashrc. Change the directory accordingly if your MKL version is different.
    ```bash
    export MKLROOT=/opt/intel/oneapi/mkl/2022.0.2
    ```

- [OpenGL / GLUT](https://www.opengl.org/)
  - OpenGL / GLUT is used for rendering the knot through a simple graphic.
  - Simply install through apt package manager:
      ```bash
    sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
    ```
- Lapack (*usually preinstalled on your computer*)

***
### Compiling
After completing all the necessary above steps, clone the source repository of IMC and then build the project through cmake.
```bash
mkdir build && cd build
cmake ..
make -j4
```

***

### Setting Parameters

All simulation parameters are set through a parameter file ```option.txt```. A template file ```template_option.txt``` is provided that can be used to construct ```option.txt```.

```bash
cp template_option.txt option.txt   # create option.txt
```
Specifiable parameters are as follows (we use SI units):
- ```RodLength``` - Contour length of the rod.
- ```numVertices``` - Number of nodes on the rod.
- ```rodRadius``` - Cross-sectional radius of the rod.
- ```helixradius``` - Radius of the helix.
- ```helixpitch``` - Pitch of the helix.
- ```density``` - Mass per unit volume.
- ```youngM``` - Young's modulus.
- ```Poisson``` - Poisson ratio.
- ```tol``` and ```stol``` - Small numbers used in solving the linear system. Fraction of a percent, e.g. 1.0e-3, is often a good choice.
- ```maxIter``` - Maximum number of iterations allowed before the solver quits. 
- ```gVector``` - 3x1 vector specifying acceleration due to gravity.
- ```viscosity``` - Viscosity for applying damping forces.
- ```render (0 or 1) ```- Flag indicating whether OpenGL visualization should be rendered.
- ```saveData (0 or 1)``` - Flag indicating whether pull forces and rod end positions should be reocrded.
- ```recordNodes (0 or 1)``` - Flag indicating whether nodal positions will be recorded.
- ```dataResolution``` - Rate of data recording in seconds. Applies to both ```saveData``` and ```recordNodes```.
- ```waitTime``` - Initial wait period duration.
- ```pullTime``` - Duration to pull for (*starts after ```waitTime``` is done*).
- ```releaseTime``` - Duration to loosen for (*starts after ```waitTime``` + ```pullTime``` is done*).
- ```pullSpeed``` - Speed at which to pull and/or loosen each end.
- ```deltaTime``` - Time step size.
- ```colLimit``` - Distance limit for inclusion in contact candidate set (*colLimit must be > delta*).
- ```delta``` - Distance tolerance for contact.
- ```kScaler``` - Constant scaling factor for contact stiffness.
- ```mu``` - Friction coefficient. A value of zero turns friction off.
- ```nu``` - Slipping tolerance for friction.
- ```lineSearch (0 or 1)``` - Flag indicating whether line search will be used.
- ```knotConfig``` - File name for the initial knot configuration. Should be a txt file located in ```knot_configurations``` directory. Note that overhand knot configurations for ```n1, n2, n3, n4``` are provided with a discretization of 301 nodes.

***
### Running the Simulation
Once parameters are set to your liking, the simulation can be ran from the terminal by running the provided script:
```bash
./run.sh
```
If this doesn't work, execute ```chmod +x run.sh``` prior to running.

***

### Citation
If our work has helped your research, please cite the following paper.
```
@article{choi_imc_2021,
    author = {Choi, Andrew and Tong, Dezhong and Jawed, Mohammad K. and Joo, Jungseock},
    title = "{Implicit Contact Model for Discrete Elastic Rods in Knot Tying}",
    journal = {Journal of Applied Mechanics},
    volume = {88},
    number = {5},
    year = {2021},
    month = {03},
    issn = {0021-8936},
    doi = {10.1115/1.4050238},
    url = {https://doi.org/10.1115/1.4050238},
}

@article{tong_imc_2022,
    author = {Dezhong Tong and Andrew Choi and Jungseock Joo and M. Khalid Jawed},
    title = {A fully implicit method for robust frictional contact handling in elastic rods},
    journal = {Extreme Mechanics Letters},
    volume = {58},
    pages = {101924},
    year = {2023},
    issn = {2352-4316},
    doi = {https://doi.org/10.1016/j.eml.2022.101924},
    url = {https://www.sciencedirect.com/science/article/pii/S2352431622002000},
}
```



