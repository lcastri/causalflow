# <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/causalflow_icon.png" width="35"> CausalFlow: Causal Discovery Methods with Observational and Interventional Data from Time-series

CausalFlow is a python library for causal analysis from time-series data. It comprises:

* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15"> [F-PCMCI](https://github.com/lcastri/fpcmci) - Filtered-PCMCI
* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15"> CAnDOIT - CAusal Discovery with Observational and Interventional data from Time-series
* RandomGraph
* Other causal discovery methods all within the same framework

## Useful links
* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15"> F-PCMCI:<br>
  L. Castri, S. Mghames, M. Hanheide and N. Bellotto (2023).<br>
  [Enhancing Causal Discovery from Robot Sensor Data in Dynamic Scenarios](https://proceedings.mlr.press/v213/castri23a/castri23a.pdf),<br>
  Proceedings of the Conference on Causal Learning and Reasoning (CLeaR).<br>
  ```
  @inproceedings{castri2023enhancing,
    title={Enhancing Causal Discovery from Robot Sensor Data in Dynamic Scenarios},
    author={Castri, Luca and Mghames, Sariah and Hanheide, Marc and Bellotto, Nicola},
    booktitle={Conference on Causal Learning and Reasoning},
    pages={243--258},
    year={2023},
    organization={PMLR}
  }
  ```
* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15"> CAnDOIT:<br>
  L. Castri, S. Mghames, M. Hanheide and N. Bellotto (2024).<br>
  [CAnDOIT: Causal Discovery with Observational and Interventional Data from Time-Series](),<br>
  Advanced Intelligent System.<br>
  ```
  BibTex coming soon!
  ```
* Tutorials [Coming soon..]

## <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="18"> F-PCMCI
Extension of the state-of-the-art causal discovery method [PCMCI](https://github.com/jakobrunge/tigramite), augmented with a feature-selection method based on Transfer Entropy. The algorithm, starting from a prefixed set of variables, identifies the correct subset of features and a hypothetical causal model between them. Then, using the selected features and the hypothetical causal model, the causal discovery is executed. This refined set of variables and the list of potential causal links between them contribute to achieving **faster** and **more accurate** causal discovery.

In the following, an example demonstrating the main functionality of F-PCMCI is presented, along with a comparison between causal models obtained by PCMCI and F-PCMCI causal discovery algorithms using the same data. The dataset consists of a 7-variables system defined as follows:

$$
\begin{aligned}
X_0(t) &= 2X_1(t-1) + 3X_3(t-1) + \eta_0 \\
X_1(t) &= \eta_1 \\
X_2(t) &= 1.1(X_1(t-1))^2 + \eta_2 \\
X_3(t) &= X_3(t-1) \cdot X_2(t-1) + \eta_3 \\
X_4(t) &= X_4(t-1) + X_5(t-1) \cdot X_0(t-1) + \eta_4 \\
X_5(t) &= \eta_5 \\
X_6(t) &= \eta_6
\end{aligned}
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

<div align="center">

Causal Model by PCMCI       |  Causal Model by F-PCMCI 
:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/images/PCMCI_example_2.png "Causal model by PCMCI")  |  ![](https://github.com/lcastri/causalflow/raw/main/images/FPCMCI_example_2.png "Causal model by F-PCMCI")
Execution time ~ 8min 40sec | Execution time ~ 3min 00sec

</div>

F-PCMCI removes the variable $X_6$ from the causal graph (since isolated), and generate the correct causal model. In contrast, PCMCI retains $X_6$ leading to the wrong causal structure. Specifically, a spurious link $X_6$ &rarr; $X_5$ appears in the causal graph derived by PCMCI.

## <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="18"> CAnDOIT
CAnDOIT extends [LPCMCI](https://github.com/jakobrunge/tigramite), allowing the possibility of incorporating interventional data in the causal discovery process alongside the observational data. As its predecessor, CAnDOIT can deal with lagged and contemporaneous dependecies and latent variables.

In the following, an example is presented that demonstrates CAnDOIT's capability to incorporate and exploit interventional data. The dataset consists of a 5-variables system defined as follows:

$$
\begin{aligned}
X_0(t) &= \eta_0\\
X_1(t) &= 2.5X_0(t-1) + \eta_1\\
X_2(t) &= 0.5X_0(t-2) \cdot 0.75X_3(t-1) + \eta_2\\
X_3(t) &= 0.7X_3(t-1)X_4(t-2) + \eta_3\\
X_4(t) &= \eta_4\\
\end{aligned}
$$

This system of equation generates the time-series data in the observational case. For the interventional case instead, the equation $X_1(t) = 2.5X_0(t-1) + \eta_1$ was replaced by a hard intervention $X_1(t) = 15$.

```python
min_lag = 1
max_lag = 2
np.random.seed(1)
nsample_obs = 1000
nsample_int = 300
nfeature = 5
d = np.random.random(size = (nsample_obs, nfeature))
for t in range(max_lag, nsample_obs):
    d[t, 1] += 2.5 * d[t-1, 0]
    d[t, 2] += 0.5 * d[t-2, 0] * 0.75 * d[t-1, 3] 
    d[t, 3] += 0.7 * d[t-1, 3] * d[t-2, 4]


# hard intervention on X_1
d_int1 = np.random.random(size = (nsample_int, nfeature))
d_int1[:, 1] = 15 * np.ones(shape = (nsample_int,)) 
for t in range(max_lag, nsample_int):
    d_int1[t, 2] += 0.5 * d_int1[t-2, 0] * 0.75 * d_int1[t-2, 3] 
    d_int1[t, 3] += 0.7 * d_int1[t-1, 3] * d_int1[t-2, 4]
```

<div align="center">

Ground-truth Causal Model       |  Causal Model by LPCMCI |  Causal Model by CAnDOIT 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/images/GT_example_1.png)  |  ![](https://github.com/lcastri/causalflow/raw/main/images/FPCMCI_example_1.png) |  ![](https://github.com/lcastri/causalflow/raw/main/images/CAnDOIT_example_1.png)
| $X_0$ observable | $X_0$ hidden | $X_0$ hidden |
| observation samples 1000 | observation samples 1000 | observation samples 700 |
| intervention samples &cross; | intervention samples &cross; | observation samples 300 |

</div>

By using interventional data, CAnDOIT removes the spurious link $X_1$ &rarr; $X_2$ generated by the hidden confounder $X_0$.

## RandomGraph
RandomGraph is a random-model generator capable of creating random systems of equations. It can create systems of equation with different properties: linear, nonlinear, with lagged and/or contemporaneous dependecies and with hidden confounders.
This tool offers various adjustable parameters, listed as follows:
* time-series length;
* number of observable variables;
* number of observable parents per variable (link density);
* number of hidden confounders;
* number of confounded variables per hidden confounder;
* noise configuration, e.g. Gaussian noise $\mathcal{N}(\mu, \sigma^2)$;
* minimum $\tau_{min}$ and maximum $\tau_{max}$ time delay to consider in the equations;
* coefficient range of the equations' terms;
* functional forms applied to the equations' terms: $[-, \sin, \cos, \text{abs}, \text{pow}, \text{exp}]$, where $-$ stands for none;
* operators used to link various equations terms: $[+, -, *, /]$.

RandomGraph outputs a graph, the associated system of equations and observational data. Moreover, it offers the possibility to generate interventional data.

#### Example - Linear Random Graph

```python
noise_uniform = (NoiseType.Uniform, -0.5, 0.5)
noise_gaussian = (NoiseType.Gaussian, 0, 1)
noise_weibull = (NoiseType.Weibull, 2, 1)
RG = RandomGraph(nvars = 5, 
                 nsamples = 1000, 
                 link_density = 3, 
                 coeff_range = (0.1, 0.5), 
                 max_exp = 2, 
                 min_lag = 0, 
                 max_lag = 3, 
                 noise_config = random.choice([noise_uniform, noise_gaussian, noise_weibull]),
                 functions = [''], 
                 operators = ['+', '-'], 
                 n_hidden_confounders = 2)
RG.gen_equations()
RG.ts_dag(withHidden = True)
```

$$
\begin{aligned}
X_0(t)&=0.44X_1(t-1) - 0.15X_0(t-2) + 0.1X_4(t-3) + 0.33H_0(t-3) - 0.11H_1(t-2)\\
X_1(t)&=0.13X_2(t-2) + 0.19H_0(t-3) + 0.46H_1(t-3)\\
X_2(t)&=0.21X_4(t-3) + 0.37H_1(t)\\
X_3(t)&=0.23X_0(t-2) - 0.44H_0(t-3) - 0.17H_1(t-3)\\
X_4(t)&=0.47X_1(t-2) + 0.23X_0(t-3) + 0.49X_4(t-1) + 0.49H_0(t-3) - 0.27H_1(t-2)\\
H_0(t)&=0.1X_4(t-2)\\
H_1(t)&=0.44X_3(t)\\
\end{aligned}
$$  

#### Example - Nonlinear Random Graph

```python
noise_uniform = (NoiseType.Uniform, -0.5, 0.5)
noise_gaussian = (NoiseType.Gaussian, 0, 1)
noise_weibull = (NoiseType.Weibull, 2, 1)
RG = RandomGraph(nvars = 5, 
                 nsamples = 1000, 
                 link_density = 3, 
                 coeff_range = (0.1, 0.5), 
                 max_exp = 2, 
                 min_lag = 0, 
                 max_lag = 3, 
                 noise_config = random.choice([noise_uniform, noise_gaussian, noise_weibull]),
                 functions = ['','sin', 'cos', 'exp', 'abs', 'pow'], 
                 operators = ['+', '-', '*', '/'], 
                 n_hidden_confounders = 2)
RG.gen_equations()
RG.ts_dag(withHidden = True)
```

$$
\begin{aligned}
X_0(t)&=0.48\frac{\cos(X_4)(t-3)}{0.12\sin(H_1)(t-3)}\\
X_1(t)&=0.17\sin(X_4)(t-3) - 0.46\cos(X_3)(t) + 0.14|H_1|(t-3)\\
X_2(t)&=\frac{0.32X_4(t-1)}{0.2X_2(t-1)} + 0.23|X_3|(t-2) - 0.34e^{H_1}(t-3)\\
X_3(t)&=0.1|X_1|(t-1) \cdot 0.26\sin(X_2)(t) \cdot 0.4\cos(X_0)(t-2) - 0.2\cos(H_0)(t-2)\\
X_4(t)&=0.24|X_1|(t-3) - 0.43X_3(t) + 0.31\sin(H_0)(t-3) + 0.21H_1(t-3)\\
H_0(t)&=0.45|X_3|(t-2)\\
H_1(t)&=\frac{0.32H_0(t-1)}{0.35e^{H_1}(t-3)} \cdot 0.4X_4(t-3)\\
\end{aligned}
$$

<div align="center">

| Linear Random Graph | Nonlinear Random Graph |
:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/images/lin_hid_randomgraph.png)  |  ![](https://github.com/lcastri/causalflow/raw/main/images/nonlin_hid_randomgraph.png)
Linear model | Nonlinear model
Lagged dependencies | Lagged dependencies
Contemporaneous dependencies | Contemporaneous dependencies
2 hidden confounders | 2 hidden confounders

</div>

#### Example - Random Graph with Interventional Data

```python
noise_gaussian = (NoiseType.Gaussian, 0, 1)
RS = RandomGraph(nvars = 5, 
                 nsamples = 1500, 
                 link_density = 3, 
                 coeff_range = (0.1, 0.5), 
                 max_exp = 2, 
                 min_lag = 0, 
                 max_lag = 3, 
                 noise_config = noise_gaussian,
                 functions = ['','sin', 'cos', 'exp', 'abs', 'pow'], 
                 operators = ['+', '-', '*', '/'], 
                 n_hidden_confounders = 2)
RS.gen_equations()

d_obs_wH, d_obs = RS.gen_obs_ts()
d_obs.plot_timeseries()

d_int = RS.intervene(intvar, nsample_int, random.uniform(5, 10), d_obs.d)
d_int[intvar].plot_timeseries()
```

<div align="center">

| Observational Data | Interventional Data |
:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/images/obs_randomgraph.png)  |  ![](https://github.com/lcastri/causalflow/raw/main/images/int_randomgraph.png)

</div>


## Other Causal Discovery Algorithms
Although the main contribution of this repository is to present the CAnDOIT and F-PCMCI algorithms, other causal discovery methods have been included for benchmark purposes. As a consequence, CausalFLow provides a collection of causal discovery methods, beyond F-PCMCI and CAnDOIT, that output time-series graphs (graphs which comprises the lag specification for each link). They are listed as follows:

* [DYNOTEARS](https://arxiv.org/pdf/2002.00498.pdf) - from the [causalnex](https://github.com/mckinsey/causalnex) package;
* [PCMCI](http://proceedings.mlr.press/v124/runge20a/runge20a.pdf) - from the [tigramite](https://github.com/jakobrunge/tigramite) package;
* [PCMCI+](http://auai.org/uai2020/proceedings/579_main_paper.pdf) - from the [tigramite](https://github.com/jakobrunge/tigramite) package;
* [LPCMCI](https://proceedings.neurips.cc/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html) - from the [tigramite](https://github.com/jakobrunge/tigramite) package;
* [TCDF](https://www.mdpi.com/2504-4990/1/1/19) - from the [causal_discovery_for_time_series](https://github.com/ckassaad/causal_discovery_for_time_series) package;
* [tsFCI](https://www.researchgate.net/publication/268324455_On_Causal_Discovery_from_Time_Series_Data_using_FCI) - from the [causal_discovery_for_time_series](https://github.com/ckassaad/causal_discovery_for_time_series) package;
* [VarLiNGAM](https://www.jmlr.org/papers/volume11/hyvarinen10a/hyvarinen10a.pdf) - from the [lingam](https://github.com/cdt15/lingam?tab=readme-ov-file) package;

Some algorithms are imported from other languages such as R and Java and are then wrapped in Python. Having the majority of causal discovery methods integrated into a single framework, which handles various types of inputs and outputs causal models, can facilitate the use of these algorithms. 

<div align="center">

|   |  Algorithm        | Observations | Feature Selection | Interventions |
|:-:|:-----------------|:-----------------:|:------------:|:-------------:|
| | DYNOTEARS | ✅ | ❌ | ❌ |
| | PCMCI | ✅ | ❌ | ❌ |
| | PCMCI+ | ✅ | ❌ | ❌ |
| | LPCMCI | ✅ | ❌ | ❌ |
| | TCDF | ✅ | ❌ | ❌ |
| | tsFCI | ✅ | ❌ | ❌ |
| | VarLiNGAM | ✅ | ❌ | ❌ |
| <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15"> | F-PCMCI | ✅ | ✅ | ❌ |
| <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15"> | CAnDOIT | ✅ | ❌ | ✅ |

</div>


## Citation
Please consider citing the following papers depending on which method you use:

* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15"> F-PCMCI:<br>
  L. Castri, S. Mghames, M. Hanheide and N. Bellotto (2023).<br>
  [Enhancing Causal Discovery from Robot Sensor Data in Dynamic Scenarios](https://proceedings.mlr.press/v213/castri23a/castri23a.pdf),<br>
  Proceedings of the Conference on Causal Learning and Reasoning (CLeaR).<br>
  ```
  @inproceedings{castri2023enhancing,
    title={Enhancing Causal Discovery from Robot Sensor Data in Dynamic Scenarios},
    author={Castri, Luca and Mghames, Sariah and Hanheide, Marc and Bellotto, Nicola},
    booktitle={Conference on Causal Learning and Reasoning},
    pages={243--258},
    year={2023},
    organization={PMLR}
  }
  ```
* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15"> CAnDOIT:<br>
  L. Castri, S. Mghames, M. Hanheide and N. Bellotto (2024).<br>
  [CAnDOIT: Causal Discovery with Observational and Interventional Data from Time-Series](),<br>
  Advanced Intelligent System.<br>
  ```
  BibTex coming soon!
  ```


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

Before installing CausalFlow, you need to install Java and the [IDTxl package](https://github.com/pwollstadt/IDTxl) used for the feature-selection process, following the guide described [here](https://github.com/pwollstadt/IDTxl/wiki/Installation-and-Requirements). Once complete, you can install the current release of `CausalFlow` with:
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

### 3 - CausalFlow installation
```shell
# COMING SOON: pip install causalflow
```


## Recent changes

| Version | Changes |
| :---: | ----------- |
| 4.0.0 | package published |
