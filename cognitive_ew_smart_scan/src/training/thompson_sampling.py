"""
Thompson Sampling Explorer for warmup exploration.

Beta(1,1) prior per band → principled exploration before DRQN takes over.
Also provides UCB1 alternative.

The explorer operates on bands (the arms) and convenience methods emit the
canonical time-frequency flat action ``band * n_modes + mode``.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ThompsonSamplingExplorer:
    """Beta-Bernoulli Thompson Sampler + UCB1.

    Each band is an independent Bernoulli arm with Beta(alpha,beta) posterior,
    initialised to Beta(1,1) (uniform). Also tracks counts for UCB1.

    Attributes:
        n_bands: Number of arms.
        alpha: Success counts +1.
        beta: Failure counts +1.
        counts: Total pulls per arm (for UCB).
        total_pulls: Global pull count.
    """

    def __init__(self, n_bands: int = 36, n_modes: int = 1, seed: int | None = None) -> None:
        """Initialise explorer.

        Args:
            n_bands: Number of frequency band arms.
            n_modes: Dwell modes per band for flat-action emission.
            seed: RNG seed.
        """
        self.n_bands = n_bands
        self.n_modes = int(n_modes)
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_bands, dtype=np.float64)
        self.beta = np.ones(n_bands, dtype=np.float64)
        self.counts = np.zeros(n_bands, dtype=np.int64)
        self.total_pulls: int = 0
        self._rewards = np.zeros(n_bands, dtype=np.float64)
        logger.info("ThompsonSamplingExplorer n_bands=%d Beta(1,1)", n_bands)

    def select_band(self) -> int:
        """Sample from each Beta posterior and return argmax.

        Returns:
            Selected band index.
        """
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def select_action(self) -> int:
        """Sample a full time-frequency action for the scheduler.

        Returns:
            Flat action = band * n_modes + NORMAL_DWELL (0).
        """
        return self.select_band() * self.n_modes

    def get_ucb_band(self, c: float = 2.0) -> int:
        """UCB1 alternative selection.

        UCB = mean + c * sqrt(log(total)/count). Unpulled arms prioritized.

        Args:
            c: Exploration constant.

        Returns:
            Band index with highest UCB.
        """
        # Prioritize never-pulled arms
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return int(self.rng.choice(untried))
        means = self._rewards / np.maximum(1, self.counts)
        ucb = means + c * np.sqrt(np.log(max(1, self.total_pulls)) / self.counts)
        return int(np.argmax(ucb))

    def update(self, band: int, reward: float) -> None:
        """Update posterior for chosen arm.

        Args:
            band: Band index scanned (a flat time-frequency action is decoded to
                its band before updating).
            reward: Scalar reward (>0 → success, ≤0 → failure).
        """
        if band >= self.n_modes and band < self.n_bands * self.n_modes:
            # Decode flat time-frequency action -> band arm.
            band = band // self.n_modes
        if not (0 <= band < self.n_bands):
            raise ValueError(f"band {band} out of range")
        self.counts[band] += 1
        self.total_pulls += 1
        self._rewards[band] += float(reward)
        if reward > 0:
            self.alpha[band] += 1.0
        else:
            self.beta[band] += 1.0
        logger.debug("Update band=%d reward=%.3f α=%.1f β=%.1f", band, reward, self.alpha[band], self.beta[band])

    def reset(self) -> None:
        """Reinitialise all posteriors to Beta(1,1) (call per new pulse train)."""
        self.alpha.fill(1.0)
        self.beta.fill(1.0)
        self.counts.fill(0)
        self._rewards.fill(0.0)
        self.total_pulls = 0
        logger.debug("Thompson posteriors reset to Beta(1,1)")

    def get_posterior_means(self) -> np.ndarray:
        """Return posterior means for all bands.

        Returns:
            Array (n_bands,) in (0,1).
        """
        return self.alpha / (self.alpha + self.beta)


# Alias for backward compatibility
ThompsonSampler = ThompsonSamplingExplorer
