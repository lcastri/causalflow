## <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/causalflow_icon.png" width="25"> CausalFlow: a Collection of Methods for Causal Discovery from Time-series

CausalFlow is a python library for causal analysis from time-series data. It comprises:

* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15"> [F-PCMCI](https://github.com/lcastri/fpcmci) - Filtered-PCMCI
* <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15"> CAnDOIT - CAusal Discovery with Observational and Interventional data from Time-series
* RandomGraph
* Other causal discovery methods all within the same framework

### Useful links
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
  [CAnDOIT: Causal Discovery with Observational and Interventional Data from Time-Series](https://arxiv.org/pdf/2410.02844),<br>
  Advanced Intelligent Systems.<br>
  ```
  BibTex coming soon!
  ```
* Tutorials [Coming soon..]

### <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15"> F-PCMCI
Extension of the state-of-the-art causal discovery method [PCMCI](https://github.com/jakobrunge/tigramite), augmented with a feature-selection method based on Transfer Entropy. The algorithm, starting from a prefixed set of variables, identifies the correct subset of features and a hypothetical causal model between them. Then, using the selected features and the hypothetical causal model, the causal discovery is executed. This refined set of variables and the list of potential causal links between them contribute to achieving **faster** and **more accurate** causal discovery.

In the following, an example demonstrating the main functionality of F-PCMCI is presented, along with a comparison between causal models obtained by PCMCI and F-PCMCI causal discovery algorithms using the same data. The dataset consists of a 7-variables system defined as follows:

<p align="center">
  <img src="https://github.com/lcastri/causalflow/raw/main/PyPI/eq_1.png" alt="Equation 1">
</p>

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

F-PCMCI removes the variable $X_6$ from the causal graph (since isolated), and generate the correct causal model. In contrast, PCMCI retains $X_6$ leading to the wrong causal structure. Specifically, a spurious link $X_6$ -> $X_5$ appears in the causal graph derived by PCMCI.

### <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15"> CAnDOIT
CAnDOIT extends [LPCMCI](https://github.com/jakobrunge/tigramite), allowing the incorporation of interventional data into the causal discovery process alongside observational data. Like its predecessor, CAnDOIT can handle both lagged and contemporaneous dependencies, as well as latent variables.

#### Example
In the following example, taken from one of the tigramite tutorials ([this](https://github.com/jakobrunge/tigramite/blob/master/tutorials/causal_discovery/tigramite_tutorial_latent-pcmci.ipynb)), we demonstrate CAnDOIT's ability to incorporate and leverage interventional data to improve the accuracy of causal analysis. The example involves a system of equations with four variables:

<p align="center">
  <img src="https://github.com/lcastri/causalflow/raw/main/PyPI/eq_2.png" alt="Equation 2">
</p>

Note that $L_1$ is a latent confounder of $X_0$ and $X_2$. This system of equations generates the time-series data in the observational domain, which is then used by LPCMCI for causal discovery analysis.

```python
tau_max = 2
pc_alpha = 0.05
np.random.seed(19)
nsample_obs = 500
nfeature = 4

d = np.random.random(size = (nsample_obs, nfeature))
for t in range(tau_max, nsample_obs):
  d[t, 0] += 0.9 * d[t-1, 0] + 0.6 * d[t, 1]
  d[t, 2] += 0.9 * d[t-1, 2] + 0.4 * d[t-1, 1]
  d[t, 3] += 0.9 * d[t-1, 3] - 0.5 * d[t-2, 2]

# Remove the unobserved component time series
data_obs = d[:, [0, 2, 3]]

var_names = ['X_0', 'X_2', 'X_3']
d_obs = Data(data_obs, vars = var_names)
d_obs.plot_timeseries()

lpcmci = LPCMCI(d_obs,
                min_lag = 0,
                max_lag = tau_max,
                val_condtest = ParCorr(significance='analytic'),
                alpha = pc_alpha)

# Run LPCMCI
lpcmci_cm = lpcmci.run()
lpcmci_cm.ts_dag(node_size = 4, min_width = 1.5, max_width = 1.5, 
                 x_disp=0.5, y_disp=0.2, font_size=10)
```

<div align="center">

Observational Data       |  Causal Model by LPCMCI  
:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/images/LPCMCI_data.png)  |  ![](https://github.com/lcastri/causalflow/raw/main/images/LPCMCI.png)

</div>

As you can see from LPCMCI's result, the method correctly identifies the bidirected link (indicating the presence of a latent confounder) between $X_0$ and $X_2$. However, the final causal model presents uncertainty regarding the link $X_2$ o-> $X_3$. Specifically, the final causal model is a PAG that represents two MAGs: the first with $X_2$ <-> $X_3$, and the second with $X_2$ -> $X_3$.

Now, let's introduce interventional data and examine its benefits. In this case, we perform a hard intervention on the variable $X_2$, meaning we replace its equation with a constant value corresponding to the intervention (in this case, $X_2 = 3$).

```python
nsample_int = 150
int_data = dict()

# Intervention on X_2.
d_int = np.random.random(size = (nsample_int, nfeature))
d_int[0:tau_max, :] = d[len(d)-tau_max:,:]
d_int[:, 2] = 3 * np.ones(shape = (nsample_int,)) 
for t in range(tau_max, nsample_int):
    d_int[t, 0] += 0.9 * d_int[t-1, 0] + 0.6 * d_int[t, 1]
    d_int[t, 3] += 0.9 * d_int[t-1, 3] - 0.5 * d_int[t-2, 2]
        
data_int = d_int[:, [0, 2, 3]]
df_int = Data(data_int, vars = var_names)
int_data['X_2'] =  df_int

candoit = CAnDOIT(d_obs, 
                  int_data,
                  alpha = pc_alpha, 
                  min_lag = 0, 
                  max_lag = tau_max, 
                  val_condtest = ParCorr(significance='analytic'))
    
candoit_cm = candoit.run()
candoit_cm.ts_dag(node_size = 4, min_width = 1.5, max_width = 1.5, 
                  x_disp=0.5, y_disp=0.2, font_size=10)
```

<div align="center">

Observational & Interventional Data       |  Causal Model by CAnDOIT  
:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/images/CAnDOIT_data.png)  |  ![](https://github.com/lcastri/causalflow/raw/main/images/CAnDOIT.png) 

</div>

CAnDOIT, like LPCMCI, correctly detects the bidirected link $X_0$ <-> $X_2$. Additionally, by incorporating interventional data, CAnDOIT resolves the uncertainty regarding the link $X_2$ o-> $X_3$, resulting in a **reduction of the PAG size**. Specifically, the PAG found by CAnDOIT is the representaion of only one MAG.

#### Robotics application of CAnDOIT
In this section, we discuss an application of CAnDOIT in a robotic scenario. We designed an experiment to learn the causal model in a hypothetical robot arm application equipped with a camera. For this application, we utilised [Causal World](https://github.com/rr-learning/CausalWorld), which models a TriFinger robot, a floor, and a stage. 

In our case, we use only one finger of the robot, with the finger's end effector equipped with a camera. The scenario consists of a cube placed at the centre of the floor, surrounded by a white stage. 
The colour's brightness ($b$) of the cube and the floor is modelled as a function of the end-effector height ($H$), its absolute velocity ($v$), and the distance between the end-effector and the cube $d_c$. This model captures the shading and blurring effects on the cube. In contrast, the floor, being darker and larger than the cube, is only affected by the end effector's height.

Note that $H$, $v$, and $d_c$ are obtained directly from the simulator and not explicitly modelled, while the ground-truth structural causal model for the floor colour ($F_c$) and cube colour ($C_c$) is expressed as follows:

<p align="center">
  <img src="https://github.com/lcastri/causalflow/raw/main/PyPI/eq_3.png" alt="Equation 3">
</p>

This model is used to generate observational data, which is then used by LPCMCI and CAnDOIT to reconstruct the causal model. For the interventional domain instead, we substitute the equation modelling $F_c$ with a constant colour (green) and collect the data for the causal analysis conducted by CAnDOIT. Note that, for both the obervational and interventional domains, $H$ is considered as latent confounder between $F_c$ and $C_c$.

<div align="center">

Observational dataset       |  Interventional dataset  
:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/gifs/exp_obs.gif)  |  ![](https://github.com/lcastri/causalflow/raw/main/gifs/exp_int.gif) 

</div>

<div align="center">

Ground-truth Causal Model       | Causal Model by LPCMCI       |  Causal Model by CAnDOIT  
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/images/CW_complete.png)  |  ![](https://github.com/lcastri/causalflow/raw/main/images/CW_LPCMCI.png) |  ![](https://github.com/lcastri/causalflow/raw/main/images/CW_CAnDOIT.png) 

</div>

Also in this experiment, we can see the benefit of using intervention data alongside the observations. LPCMCI is unable to orient the contemporaneous (spurious) link between $F_c$ and $C_c$ due to the hidden confounder $H$. This results in the ambiguous link $F_c$ o-o $C_c$, which does not encode the correct link <->. Instead CAnDOIT, using interventional data, correctly identifies the bidirected link $F_c$ <-> $C_c$, decreasing once again the uncertainty level and increasing the accuracy of the reconstructed causal model.

### RandomGraph
RandomGraph is a random-model generator capable of creating random systems of equations with various properties: linear, nonlinear, lagged and/or contemporaneous dependencies, and hidden confounders. 
This tool offers several adjustable parameters, listed as follows:
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

RandomGraph outputs a graph, the associated system of equations, and observational data. Additionally, it provides the option to generate interventional data.

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

<p align="center">
  <img src="https://github.com/lcastri/causalflow/raw/main/PyPI/eq_4.png" alt="Equation 4">
</p>

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

<p align="center">
  <img src="https://github.com/lcastri/causalflow/raw/main/PyPI/eq_5.png" alt="Equation 5">
</p>

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

d_int = RS.intervene('X_4', 250, random.uniform(5, 10), d_obs.d)
d_int['X_4'].plot_timeseries()
```

<div align="center">

| Observational Data | Interventional Data |
:-------------------------:|:-------------------------:
![](https://github.com/lcastri/causalflow/raw/main/images/obs_randomgraph.png)  |  ![](https://github.com/lcastri/causalflow/raw/main/images/int_randomgraph.png)

</div>


### Other Causal Discovery Algorithms
Although the main contribution of this repository is to present the CAnDOIT and F-PCMCI algorithms, other causal discovery methods have been included for benchmarking purposes. Consequently, CausalFlow offers a collection of causal discovery methods, beyond F-PCMCI and CAnDOIT, that output time-series graphs (graphs that specify the lag for each link). These methods are listed as follows:

* [DYNOTEARS](https://arxiv.org/pdf/2002.00498.pdf) - from the [causalnex](https://github.com/mckinsey/causalnex) package;
* [PCMCI](http://proceedings.mlr.press/v124/runge20a/runge20a.pdf) - from the [tigramite](https://github.com/jakobrunge/tigramite) package;
* [PCMCI+](http://auai.org/uai2020/proceedings/579_main_paper.pdf) - from the [tigramite](https://github.com/jakobrunge/tigramite) package;
* [LPCMCI](https://proceedings.neurips.cc/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html) - from the [tigramite](https://github.com/jakobrunge/tigramite) package;
* [J-PCMCI+](https://proceedings.mlr.press/v216/gunther23a.html) - from the [tigramite](https://github.com/jakobrunge/tigramite) package;
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
| | J-PCMCI+ | ✅ | ❌ | ❌ |
| | TCDF | ✅ | ❌ | ❌ |
| | tsFCI | ✅ | ❌ | ❌ |
| | VarLiNGAM | ✅ | ❌ | ❌ |
| <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/fpcmci_icon.png" width="15"> | F-PCMCI | ✅ | ✅ | ❌ |
| <img src="https://github.com/lcastri/causalflow/raw/main/docs/assets/candoit_icon.png" width="15"> | CAnDOIT | ✅ | ❌ | ✅ |

</div>


### Citation
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
  [CAnDOIT: Causal Discovery with Observational and Interventional Data from Time-Series](https://arxiv.org/pdf/2410.02844),<br>
  Advanced Intelligent Systems.<br>
  ```
  BibTex coming soon!
  ```


### Requirements
* pandas>=1.5.2
* numba>=0.58.1
* scipy>=1.3.3
* networkx>=2.8.6
* ruptures>=1.1.7
* scikit_learn>=1.1.3
* torch>=1.11.0
* gpytorch>=1.4
* dcor>=0.5.3
* h5py>=3.7.0
* jpype1>=1.5.0
* mpmath>=1.3.0
* causalnex
* lingam
* pyopencl>=2024.1
* matplotlib>=3.7.0
* numpy
* pgmpy>=0.1.19
* tigramite>=5.1.0.3
* rectangle-packer
* grandalf


### Installation

Before installing CausalFlow, you need to install Java and the [IDTxl package](https://github.com/pwollstadt/IDTxl) used for the feature-selection process, following the guide described [here](https://github.com/pwollstadt/IDTxl/wiki/Installation-and-Requirements). Once complete, you can install the current release of `CausalFlow` with:
``` shell
pip install py-causalflow
```

For a complete installation Java - IDTxl - CausalFlow, follow the following procedure.

#### 1 - Java installation
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

#### 2 - IDTxl installation
```shell
# IDTxl
git clone -b v1.4 https://github.com/pwollstadt/IDTxl.git
cd IDTxl
pip install -e .
```

#### 3 - CausalFlow installation
```shell
pip install py-causalflow
```


### Recent changes

| Version | Changes |
| :---: | ----------- |
| 4.0.3 | numba version fix<br>DAG dag() fix<br>CAnDOIT fix: min_lag must be equal to 0|
| 4.0.2 | PyPI fixes<br>rectangle-packer and grandalf added to requirements<br>numba version fix<br>causal_discovery/baseline/pkgs fix|
| 4.0.1 | PyPI |
| 4.0.0 | package published |