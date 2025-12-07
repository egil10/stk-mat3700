# STK-MAT3700 - Introduction to Mathematical Finance and Investment Theory

This repository contains assignments and coursework for **STK-MAT3700 - Introduction to Mathematical Finance and Investment Theory** at the University of Oslo.

## Course Information

**Course:** STK-MAT3700 - Introduction to Mathematical Finance and Investment Theory  
**Credits:** 10  
**Level:** Bachelor  
**Institution:** University of Oslo  
**Course Page:** [https://www.uio.no/studier/emner/matnat/math/STK-MAT3700/index-eng.html](https://www.uio.no/studier/emner/matnat/math/STK-MAT3700/index-eng.html)

## Course Description

The course gives an introduction to the most important notions and problems in mathematical finance. The theory of arbitrage for pricing and hedging derivatives (options) will be studied in the context of discrete and continuous time stochastic models, with the famous Black-Scholes option pricing formula as a highlight. Moreover the course will focus on the theory of investments with special stress given to utility optimization and the Markowitz theory for optimal portfolio choice.

## Learning Outcomes

Upon completion of this course, students should:

- Develop hedging strategies for derivatives in tree-models
- Price using no-arbitrage and hedging principles in tree-models
- Know the concept of risk-neutral probability
- Develop the Black-Scholes option pricing formula as a limit of tree-models
- Apply the Black-Scholes formula to price and hedge plain-vanilla options in finance
- Create portfolios that balance profit and risk optimally

## Prerequisites

**Recommended previous knowledge:**
- MAT1100 – Calculus
- MAT1110 – Calculus and Linear Algebra
- MAT1120 – Linear Algebra / MAT1125 – Advanced linear algebra
- STK1100 – Probability and Statistical Modelling
- STK2130 – Modelling by Stochastic Processes (useful but not required)

## Repository Structure

```
stk-mat3700/
├── oblig1/              # Assignment 1 solutions
│   ├── 1a.R            # Problem 1a: Returns analysis
│   ├── 1b.R            # Problem 1b: Correlation and covariance
│   ├── 1c.R            # Problem 1c: Efficient frontier
│   ├── 1d.R            # Problem 1d: Portfolio optimization
│   ├── 2a.R            # Problem 2a: Black-Scholes implementation
│   ├── 2b.py           # Problem 2b: Implied volatility
│   ├── data/           # Financial data files
│   │   └── finance_data.csv
│   ├── plots/          # Generated plots and visualizations
│   └── oblig1.pdf      # Assignment description
├── rsc/                # Course resources
│   ├── stk-mat3700 eksamen 2024-2020.pdf
│   └── stk-mat3700 oblig 1.pdf
└── README.md           # This file
```

## Assignment 1 Overview

The mandatory assignment covers two main areas:

### Problem 1: Portfolio Optimization (Markowitz Theory)
- Analysis of financial returns and their distributions
- Calculation of correlation and covariance matrices
- Construction of efficient frontiers
- Minimum variance portfolio optimization
- Comparison of different portfolio configurations

### Problem 2: Black-Scholes Option Pricing
- Implementation of the Black-Scholes formula
- Option pricing for various scenarios
- Implied volatility calculations
- Analysis of volatility patterns

## Requirements

### R Packages
The R scripts require the following packages:
- `tidyverse` - Data manipulation and visualization
- Additional packages as specified in individual scripts

### Python Packages
The Python script (`2b.py`) may require:
- `numpy` - Numerical computations
- `scipy` - Scientific computing
- `matplotlib` - Plotting
- Additional packages as needed

## Usage

### Running R Scripts
```r
# Navigate to the oblig1 directory
cd oblig1

# Run individual problem scripts
Rscript 1a.R
Rscript 1b.R
Rscript 1c.R
Rscript 1d.R
Rscript 2a.R
```

### Running Python Scripts
```bash
# Navigate to the oblig1 directory
cd oblig1

# Run the Python script
python 2b.py
```

## Data

Financial data is stored in `oblig1/data/finance_data.csv`. The data includes trading day information and asset prices for various financial instruments.

## Output

Generated plots and visualizations are saved in the `oblig1/plots/` directory, including:
- Return distributions
- Correlation matrices
- Efficient frontiers
- Black-Scholes option pricing results
- Implied volatility analyses

## Course Resources

Additional course materials, including past exam papers and assignment descriptions, are available in the `rsc/` directory.

## Examination

This course has **1 mandatory assignment** that must be approved before sitting the final exam. The final written exam counts 100% towards the final grade.

## Academic Integrity

This repository is for educational purposes. Please ensure you understand all concepts and can explain the methodologies used. Follow your institution's academic integrity policies.

## License

This repository is for personal academic use as part of coursework at the University of Oslo.

---

**Last Updated:** 2025  
**Course:** STK-MAT3700 - Introduction to Mathematical Finance and Investment Theory  
**University of Oslo**
