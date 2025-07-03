## Model Parameters and Definitions

### Transition Probability
The probability of transitioning from state $i$ to state $j$ at a time $t$ that falls into time slice $k$.

$$
A_{ij}(k) = P(q_{t+1} = j \mid q_t = i, \text{ and time } t \text{ belongs to slice } k)
$$

The transition probability matrix for a given time slice $k$:

$$
A(k) = [a_{ij}(k)]_{N \times N}, \quad k = 1, 2, \ldots, K
$$

### Emission Probability
The probability of observing data $O_t$ given the system is in state $i$. This is modeled as a multivariate normal distribution with mean $\mu_i$ and covariance matrix $\Sigma_i$.

$$
B_i(O_t) = \mathcal{N}(O_t; \mu_i, \Sigma_i)
$$

---

## Forward Probability ($\alpha$)

The forward probability, $\alpha_t(i)$, is the probability of having observed the sequence $O_0, O_1, \ldots, O_t$ and being in state $i$ at time $t$, given the model parameters $\lambda$.

**Definition:**

$$
\alpha_t(i) = P(O_0, O_1, \ldots, O_t, q_t = i \mid \lambda)
$$

**Initialization (t=0):**

$$
\alpha_0(i) = \pi_i \cdot B_i(O_0)
$$

or equivalently:

$$
\alpha_0(j) = \pi_j \cdot \mathcal{N}(O_0; \mu_j, \Sigma_j)
$$

**Recursion (for t > 0):**

$$
\alpha_{t+1}(j) = \left[ \sum_{i=1}^{N} \alpha_t(i) \cdot A_{ij}(k(t)) \right] \cdot B_j(O_{t+1})
$$

or equivalently:

$$
\alpha_t(j) = \left[ \sum_{i=1}^{N} \alpha_{t-1}(i) \cdot a_{ij}(k(t-1)) \right] \cdot \mathcal{N}(O_t; \mu_j, \Sigma_j)
$$

---

## Backward Probability ($\beta$)

The backward probability, $\beta_t(i)$, is the probability of observing the future sequence of observations $O_{t+1}, O_{t+2}, \ldots, O_{T-1}$ given that the system is in state $i$ at time $t$.

**Definition:**

$$
\beta_t(i) = P(O_{t+1}, O_{t+2}, \ldots, O_{T-1} \mid q_t = i, \lambda)
$$

**Initialization (t=T-1):**

$$
\beta_{T-1}(i) = 1
$$

**Recursion (for t < T-1):**

$$
\beta_t(i) = \sum_{j=1}^{N} A_{ij}(k(t)) \cdot B_j(O_{t+1}) \cdot \beta_{t+1}(j)
$$

or equivalently:

$$
\beta_t(i) = \sum_{j=1}^{N} a_{ij}(k(t)) \cdot \mathcal{N}(O_{t+1}; \mu_j, \Sigma_j) \cdot \beta_{t+1}(j)
$$

---

## E-Step: Intermediate Variables

### State Occupancy Probability ($\gamma$)
The probability of being in state $i$ at time $t$ given the entire observation sequence.

$$
\gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^{N} \alpha_t(j) \beta_t(j)}
$$

An alternative formulation using the total probability of the observation sequence $P(O|\lambda)$:

$$
\gamma_t(i) = \frac{\alpha_t(i) \beta_t(i)}{P(O \mid \lambda)} \quad \text{where} \quad P(O \mid \lambda) = \sum_{i=1}^{N} \alpha_{T-1}(i)
$$

### State Transition Probability ($\xi$)
The probability of being in state $i$ at time $t$ and transitioning to state $j$ at time $t+1$, given the entire observation sequence.

$$
\xi_t(i, j) = \frac{\alpha_t(i) A_{ij}(k(t)) B_j(O_{t+1}) \beta_{t+1}(j)}{\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_t(i) A_{ij}(k(t)) B_j(O_{t+1}) \beta_{t+1}(j)}
$$

An alternative formulation:

$$
\xi_t(i, j) = \frac{\alpha_t(i) \cdot a_{ij}(k(t)) \cdot \mathcal{N}(O_{t+1}; \mu_j, \Sigma_j) \cdot \beta_{t+1}(j)}{\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_t(i) a_{ij}(k(t)) \mathcal{N}(O_{t+1}; \mu_j, \Sigma_j) \beta_{t+1}(j)}
$$

---

## M-Step: Parameter Re-estimation

### Initial State Probability ($\pi'$)

$$
\pi_i' = \gamma_0(i)
$$

### Transition Probability ($A'$)

$$
A_{ij}(k)' = \frac{\sum_{t \in T_k} \xi_t(i, j)}{\sum_{t \in T_k} \gamma_t(i)}
$$

This can be interpreted as:

$$
a_{ij}(k)' = \frac{\text{E}[\text{transitions } i \rightarrow j \text{ in } k]}{\text{E}[\text{visits to } i \text{ in } k]} = \frac{\sum_{t \in T_k} \xi_t(i, j)}{\sum_{t \in T_k} \gamma_t(i)}
$$

### Emission Mean ($\mu'$)

$$
\mu_i' = \frac{\sum_{t=0}^{T-1} \gamma_t(i) \cdot O_t}{\sum_{t=0}^{T-1} \gamma_t(i)}
$$

This is the solution to:

$$
\mu_i' = \arg\max_{\mu} \sum_{t=0}^{T-1} \gamma_t(i) \log \mathcal{N}(O_t; \mu, \Sigma_i)
$$

### Emission Covariance ($\Sigma'$)

$$
\Sigma_i' = \frac{\sum_{t=0}^{T-1} \gamma_t(i) \cdot (O_t - \mu_i')(O_t - \mu_i')^T}{\sum_{t=0}^{T-1} \gamma_t(i)}
$$

An alternative calculation:

$$
\Sigma_i' = \frac{\sum_{t} \gamma_t(i) O_t O_t^T}{\sum_{t} \gamma_t(i)} - \mu_i'(\mu_i')^T
$$

A regularization term can be added to prevent singularity:

$$
\Sigma_i' = \frac{\sum_{t} \gamma_t(i) (O_t - \mu_i')(O_t - \mu_i')^T}{\sum_{t} \gamma_t(i) + \epsilon I} \quad (\epsilon \to 0^+)
$$

---

## Likelihood and Optimization

### Log-Likelihood
The log-likelihood of observing sequence $O$ given the model $\lambda$.

$$
\log P(O \mid \lambda) = \log\left(\sum_{i=1}^{N} \alpha_{T-1}(i)\right)
$$
or

$$
\log P(O \mid \lambda) = \log\left(\sum_{i=1}^{N} \alpha_0(i) \beta_0(i)\right) = \log\left(\sum_{i=1}^{N} \pi_i B_i(O_0) \beta_0(i)\right)
$$

### EM Framework
The algorithm guarantees that the likelihood does not decrease with each iteration.

$$
L(\lambda^{(n+1)}) \geq L(\lambda^{(n)})
$$
This is achieved by maximizing the auxiliary Q-function:

$$
Q(\lambda, \lambda^{(n)}) = E_{q \mid O, \lambda^{(n)}}[\log P(O, q \mid \lambda)]
$$

### Gradients for Maximization
The gradients of the log-emission probability with respect to its parameters are used in the maximization step.

$$
\nabla_{\mu_i} \log B_i(O_t) = \Sigma_i^{-1} (O_t - \mu_i)
$$

$$
\nabla_{\Sigma_i} \log B_i(O_t) = -\frac{1}{2} \left[ \Sigma_i^{-1} - \Sigma_i^{-1} (O_t - \mu_i)(O_t - \mu_i)^T \Sigma_i^{-1} \right]
$$
