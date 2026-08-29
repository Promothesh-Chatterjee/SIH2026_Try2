"""
Thompson Sampling Multi-Armed Bandit Exploration Warmup.

During early training when the DRQN has no useful Q-values, Thompson Sampling 
over Beta(alpha, beta) posteriors provides principled exploration of frequency 
bands before handing control to the DRQN.
"""

import numpy as np


class ThompsonSampler:
    """
    Beta-Bernoulli Thompson Sampler for frequency band exploration.

    Each frequency band is modelled as an independent Bernoulli arm with a 
    Beta(alpha, beta) conjugate prior, initialized to Beta(1, 1) (uniform).
    On each step, a sample is drawn from each posterior and the arm with the 
    highest sample is selected.
    """

    def __init__(self, n_bands: int = 180, seed: int | None = None) -> None:
        """
        Initializes the Thompson Sampler.

        Args:
            n_bands: Number of frequency band arms.
            seed: Optional random seed for reproducibility.
        """
        self.n_bands = n_bands
        self.rng = np.random.default_rng(seed)

        # Beta(1, 1) == Uniform[0,1] — completely uninformed prior
        self.alpha = np.ones(n_bands, dtype=np.float64)  # successes + 1
        self.beta = np.ones(n_bands, dtype=np.float64)   # failures  + 1

    def sample_action(self) -> int:
        """
        Draws one sample from each arm's posterior and returns the argmax.

        Returns:
            Integer index of the selected frequency band.
        """
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, band_idx: int, hit: bool) -> None:
        """
        Updates the posterior of the chosen arm with the observed outcome.

        Args:
            band_idx: Index of the band that was scanned.
            hit: True if a pulse was intercepted (success), False otherwise.
        """
        if hit:
            self.alpha[band_idx] += 1.0
        else:
            self.beta[band_idx] += 1.0

    def reset(self) -> None:
        """
        Resets all posteriors to the uninformative Beta(1, 1) prior.
        """
        self.alpha.fill(1.0)
        self.beta.fill(1.0)

    def get_posterior_means(self) -> np.ndarray:
        """
        Returns the current posterior mean estimates for all bands.

        Returns:
            np.ndarray of shape (n_bands,) with values in (0, 1).
        """
        return self.alpha / (self.alpha + self.beta)
