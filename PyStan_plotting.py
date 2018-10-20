import pystan
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()  # Nice plot aesthetic
np.random.seed(101)

# Nice plot parameters
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
matplotlib.rc('text', usetex=True)

# Workflow parameter
model_compile = False

## Stan Model ##################################################################

model = """
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(alpha + beta * x, sigma);
}
"""

## Data and Sampling ###########################################################

# Parameters to be inferred
alpha = 4.0
beta = 0.5
sigma = 1.0

# Generate and plot data
x = 10 * np.random.rand(100)
y = alpha + beta * x
y = np.random.normal(y, scale=sigma)
plt.scatter(x, y)
plt.show()

# Put our data in a dictionary
data = {'N': len(x), 'x': x, 'y': y}

if model_compile:
    # Compile the model
    sm = pystan.StanModel(model_code=model)
    # Save the model
    with open('regression_model.pkl', 'wb') as f:
        pickle.dump(sm, f)
else:
    sm = pickle.load(open('regression_model.pkl', 'rb'))

# Train the model and generate samples
fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101,
                  verbose=True)
print(fit)

## Diagnostics #################################################################

summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])

alpha_mean, beta_mean = df['mean']['alpha'], df['mean']['beta']
alpha_median, beta_median = df['50%']['alpha'], df['50%']['beta']
alpha_min, alpha_max = df['2.5%']['alpha'], df['97.5%']['alpha']
beta_min, beta_max = df['2.5%']['beta'], df['97.5%']['beta']

# Extracting traces
alpha = fit['alpha']
beta = fit['beta']
sigma = fit['sigma']
lp = fit['lp__']

#alpha_mean, beta_mean, sigma_mean = np.mean(alpha), np.mean(beta), np.mean(sigma)
#alpha_median, beta_median, sigma_median = np.median(alpha), np.median(beta), np.median(sigma)
range_x = max(x) - min(x)
x_min = min(x) - 0.05 * range_x
x_max = max(x) + 0.05 * range_x
x_plot = np.linspace(x_min, x_max, 100)

# np.random.seed(34)
# np.random.shuffle(alpha), np.random.shuffle(beta)

for i in range(len(alpha)):
  plt.plot(x_plot, alpha[i] + beta[i] * x_plot, color='lightsteelblue', alpha=0.005 )

plt.plot(x_plot, alpha_mean + beta_mean * x_plot)
#plt.plot(x_plot, alpha_median + beta_median * x_plot)
plt.scatter(x, y)

plt.xlim(x_min, x_max)
plt.show()

def plot_trace(param, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""
    plt.subplot(2,1,1)
    plt.plot(param)
    plt.xlabel('samples')
    plt.ylabel(param_name)
    plt.axhline(np.mean(param), color='r', lw=2, linestyle='--')
    plt.axhline(np.median(param), color='c', lw=2, linestyle='--')

    plt.subplot(2,1,2)
    plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
    plt.axvline(np.mean(param), color='r', lw=2, linestyle='--',label='mean')
    plt.axvline(np.median(param), color='c', lw=2, linestyle='--',label='median')
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.legend()

plot_trace(alpha, 'alpha') 
plt.show()