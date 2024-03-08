# CausalFlow: Causal Discovery Methods for Time-Series Data

CausalFlow is a python library for causal analysis from time-series data. It comprises two causal discovery methods recently released in the literature:
* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15"> [F-PCMCI](https://github.com/lcastri/fpcmci) – Filtered-PCMCI
* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15"> CAnDOIT – CAusal Discovery with Observational and Interventional data from Time-series

## Useful links
Coming soon..
<!-- * [Documentation](https://lcastri.github.io/fpcmci/) -->
<!-- * [Tutorials](https://github.com/lcastri/fpcmci/tree/main/tutorials) -->

## <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="25"> F-PCMCI
Coming soon..

## <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="25"> CAnDOIT
Coming soon..

## Other Causal Discovery Algorithms
Coming soon..
<!-- To facilitate the use of -->

## Citation
Please consider citing the following papers depending on which method you use:
* F-PCMCI:<br>
    L. Castri, S. Mghames, M. Hanheide and N. Bellotto (2023). [Enhancing Causal Discovery from Robot Sensor Data in Dynamic Scenarios](https://arxiv.org/abs/2302.10135), Proceedings of the Conference on Causal Learning and Reasoning (CLeaR).<br>
    ```
    @inproceedings{castri2023fpcmci,
        title={Enhancing Causal Discovery from Robot Sensor Data in Dynamic Scenarios},
        author={Castri, Luca and Mghames, Sariah and Hanheide, Marc and Bellotto, Nicola},
        booktitle={Conference on Causal Learning and Reasoning (CLeaR)},
        year={2023},
    }
    ```
* CAnDOIT:<br>
    L. Castri, S. Mghames, M. Hanheide and N. Bellotto (2024).<br>
    CAnDOIT: Causal Discovery with Observational and Interventional Data from Time-Series,<br>
    Under review in Advanced Intelligent System.<br>


## Requirements
* pandas>=1.5.2
* netgraph>=4.10.2
* networkx>=2.8.6
* ruptures>=1.1.7
* scikit_learn>=1.1.3
* torch>=1.11.0
* gpytorch>=1.4
* dcor>=0.5.3
* h5py>=3.7.0   
* jpype1>=1.5.0
* mpmath>=1.3.0  
* causalnex>=0.12.1
* lingam>=1.8.2
* tigramite>=5.1.0.3


## Installation

Before installing the CausalFlow package, you need to install Java and the [IDTxl package](https://github.com/pwollstadt/IDTxl) used for the feature-selection process, following the guide described [here](https://github.com/pwollstadt/IDTxl/wiki/Installation-and-Requirements). Once complete, you can install the current release of `CausalFlow` with:
``` shell
# Coming soon: pip install causalflow
```

For a complete installation Java - IDTxl - CausalFlow, follow the following procedure.

### 1 - Java installation
Verify that you have not already installed Java:
```shell
java -version
```
if the latter returns `Command 'java' not found, ...`, you can install Java by the following commands, otherwise you can jump to IDTxl installation.
```shell
# Java
sudo apt-get update
sudo apt install default-jdk
```

Then, you need to add JAVA_HOME to the environment
```shell
sudo nano /etc/environment
JAVA_HOME="/lib/jvm/java-11-openjdk-amd64/bin/java" # Paste the JAVA_HOME assignment at the bottom of the file
source /etc/environment
```

### 2 - IDTxl installation
```shell
# IDTxl
git clone https://github.com/pwollstadt/IDTxl.git
cd IDTxl
pip install -e .
```

### 3 - F-PCMCI installation
```shell
# Coming soon: pip install causalflow
```


## Recent changes

| Version | Changes |
| :---: | ----------- |
| 4.0.0 | package published |