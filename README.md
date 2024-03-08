# CausalFlow: Causal Discovery Methods for Time-Series Data

CausalFlow is a python library for causal analysis from time-series data. It comprises two causal discovery methods recently released in the literature:
<!-- <div style="display: flex; align-items: center;">
    <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15" style="margin-right: 10px; margin-left: 10px;">
    [F-PCMCI](https://github.com/lcastri/fpcmci) – Filtered-PCMCI
</div>
<br>
<div style="display: flex; align-items: center;">
    <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15" style="margin-right: 10px; margin-left: 10px;">
    CAnDOIT – CAusal Discovery with Observational and Interventional data from Time-series
</div> -->
|   |  Acronym        |       Full-name             |
|:-:|-----------------|-----------------------------|
| <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15"> | [F-PCMCI](https://github.com/lcastri/fpcmci) | Filtered-PCMCI |
| <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15"> | CAnDOIT | CAusal Discovery with Observational and Interventional data from Time-series |

## Useful links
Coming soon..
<!-- * [Documentation](https://lcastri.github.io/fpcmci/) -->
<!-- * [Tutorials](https://github.com/lcastri/fpcmci/tree/main/tutorials) -->

## <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="18"> F-PCMCI
Extension of the state-of-the-art causal discovery method [PCMCI](https://github.com/jakobrunge/tigramite) augmented with a feature-selection method based on Transfer Entropy. The algorithm, starting from a prefixed set of variables, identifies the correct subset of features and a hypothetical causal model between them. Then, from the selected features and the hypothetical causal model, the causal discovery is executed. Running the latter with this refined set of variables, and with a list of potential causal links between them, contributes to achieve **faster** and **more accurate** causal discovery.

In the following it is presented an example showing the main functionality of F-PCMCI and a comparison between causal models obtained by PCMCI and F-PCMCI causal discovery algorithms on the same data. The latter have been created by defining a 7-variables system defined as follows:

$$
\begin{cases}
X_0(t) = 2X_1(t-1) + 3X_3(t-1) + \eta_0\\
X_1(t) = \eta_1\\
X_2(t) = 1.1(X_1(t-1))^2 + \eta_2\\
X_3(t) = X_3(t-1)X_2(t-1) + \eta_3\\
X_4(t) = X_4(t-1) + X_5(t-1)X_0(t-1) + \eta_4\\
X_5(t) = \eta_5\\
X_6(t) = \eta_6\\
\end{cases}
$$

```python
min_lag = 1
max_lag = 1
np.random.seed(1)
nsample = 1500
nfeature = 7

d = np.random.random(size = (nsample, feature))
for t in range(max_lag, nsample):
  d[t, 0] += 2 * d[t-1, 1] + 3 * d[t-1, 3]
  d[t, 2] += 1.1 * d[t-1, 1]**2
  d[t, 3] += d[t-1, 3] * d[t-1, 2]
  d[t, 4] += d[t-1, 4] + d[t-1, 5] * d[t-1, 0]
```

Causal Model by PCMCI       |  Causal Model by F-PCMCI 
:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/images/PCMCI_example_2.png "Causal model by PCMCI")  |  ![](https://github.com/lcastri/causalflow/raw/main/images/FPCMCI_example_2.png "Causal model by F-PCMCI")
Execution time ~ 8min 40sec | Execution time ~ 3min 00sec

In this case the F-PCMCI removes the $X_6$ variable from the causal graph leading to generate exactly the same causal model as in the previous example, with comparable executional time. Instead, the PCMCI suffers the presence of $X_6$ in terms of time and accuracy of the causal structure.
Indeed, a spurious link $X_6$ &rarr; $X_5$ appears in the causal graph derived by the PCMCI.


## <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="18"> CAnDOIT
Coming soon..

## Other Causal Discovery Algorithms
Although the main contribution of this repository is to present the CAnDOIT and F-PCMCI algorithms, other causal discovery methods have been included for benchmark purposes. As a consequence, CausalFLow provides a collection of causal discovery methods, beyond F-PCMCI and CAnDOIT, that output time-series DAGs (DAGs which comprises the lag specification for each link). They are listed as follows:

* [DYNOTEARS](https://arxiv.org/pdf/2002.00498.pdf) - from the [causalnex](https://github.com/mckinsey/causalnex) package;
* [PCMCI](http://proceedings.mlr.press/v124/runge20a/runge20a.pdf) - from the [tigramite](https://github.com/jakobrunge/tigramite) package;
* [TCDF](https://www.mdpi.com/2504-4990/1/1/19) - from the [causal_discovery_for_time_series](https://github.com/ckassaad/causal_discovery_for_time_series) package;
* [tsFCI](https://www.researchgate.net/publication/268324455_On_Causal_Discovery_from_Time_Series_Data_using_FCI) - from the [causal_discovery_for_time_series](https://github.com/ckassaad/causal_discovery_for_time_series) package;
* [VarLiNGAM](https://www.jmlr.org/papers/volume11/hyvarinen10a/hyvarinen10a.pdf) - from the [lingam](https://github.com/cdt15/lingam?tab=readme-ov-file) package;

Some algorithms are imported from other languages such as R and Java and are then wrapped in Python. Having the majority of causal discovery methods integrated into a single framework, which handles various types of inputs and outputs causal models, can facilitate the use of these algorithms. 

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
    L. Castri, S. Mghames, M. Hanheide and N. Bellotto (2024). CAnDOIT: Causal Discovery with Observational and Interventional Data from Time-Series, Under review in Advanced Intelligent System.<br>


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
# COMING SOON: pip install causalflow
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
# COMING SOON: pip install causalflow
```


## Recent changes

| Version | Changes |
| :---: | ----------- |
| 4.0.0 | package published |