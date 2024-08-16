# Anomalous Process Detection using TEP Data
Process monitoring using Autoencoder Neural Network

## Tennessee Eastman process 🏭

In this notebook we will illustrate the capabilities of some unsupervised learning techniques that perform dimensionality reduction applied to the task of process monitoring. The case-study that we are going to use is the called Tennessee Eastman process (TEP) [{cite}`downs1993plant`]. The data collected here corresponds to simulated data and it is widely used in the process monitoring community. Initially, TEP was created by the company [Eastman](https://www.eastman.com/en) to provide a realistic scenerario in which to test different process monitoring methods.

In the following we will provide a short explanation of the process. If you like to have a more detailed explanation overall read Chapter 8 of [{cite}`russell2000data`].

The process consist of five major units:

*   Reactor
*   Condenser
*   Compressor
*   Separator
*   Stripper

and it contains eight components: the gaseous reactants $A$, $C$, $D$ and $E$ that are fed into the reactor along the inert $B$ to produce the liquids $G$ and $H$. The product $F$ is an unwanted byproduct.

$$
A(g) + C(g) + D(g) → G(liq)
$$

$$
A(g) + C(g) + E(g) → H(liq)
$$

$$
A(g) + E(g) → F(liq)
$$

$$
3D(g) → 2F(liq)
$$

The reactor product stream is then cooled using the condenser and fed to a flash separator. A recycle is implemented via the compressor with the necessary purge to prevent accumulation of $B$ and $F$. Finally, the liquid outlet of the separator is fed into a stripper for further separation. See the figure below for a schematic reresentation of the flowsheet.

```{figure} TEP_flowsheet.png
:alt: cstr
:width: 80%
:align: center

Schematic representation of the Tennessee Eastman process flowsheet. Modified from [{cite}`russell2000data`].
```

The dataset contains 41 measured states and 11 manipulated variables. Some variables are sampled every 3 minutes, 6 minutes or 15 minutes. And all measurements include Gaussian noise.

The data contains 21 faults, out of which 16 are known (Faults 1-15 and 21). Some faults are caused by step changes in some process variables, while others are associated with a random variability increase. A slow drift in the reaction kinetics and sticking valves are other causes of the faults.


Let's import the data corresponding to the 21 faulty simulations (i.e., files 01 to 21) and the data for the normal operation (i.e., files with 00) using some for-loops and store it into a dictionary. 

### **Process faults**

File | Description                                            | Type
---- | ------------------------------------------------------ | ----
00   | Normal operation                                       |
01   | A/C Feed Ratio, B Composition Constant (Stream 4)      | Step
02  | B Composition, A/C Ratio Constant (Stream 4)            | Step
03  | D Feed Temperature (Stream 2)                           | Step
04  | Reactor Cooling Water Inlet Temperature                 | Step
05  | Condenser Cooling Water Inlet Temperature               | Step
06  | A Feed Loss (Stream 1)                                  | Step
07  | C Header Pressure Loss - Reduced Availability (Stream 4)| Step
08  | A, B, C Feed Composition (Stream 4)                     | Random Variation
09  | D Feed Temperature (Stream 2)                           | Random Variation
10 | C Feed Temperature (Stream 4)                            | Random Variation
11 | Reactor Cooling Water Inlet Temperature                  | Random Variation
12 | Condenser Cooling Water Inlet Temperature                | Random Variation
13 | Reaction Kinetics                                        | Slow Drift
14 | Reactor Cooling Water Valve                              | Valve Sticking
15 | Condenser Cooling Water Valve                            |Valve Sticking
16 | Unknown
17 | Unknown
18 | Unknown
19 | Unknown
20 | Unknown
21 | Unknown

As you can suspect from the code below, the data has already being splitted into training and test.
