"""
Provides a Bayesian Knowledge Tracing model.
"""

from hmm.hmm import HMM

import random

class BKT(HMM):

    """Bayesian Knowledge Tracing Model"""

    @classmethod
    def from_params(cls, init, trans, guess, slip):
        """Initializes a Bayesian Knowledge Tracing model.

        :init: The initial probability of mastery.
        :trans: The probability of transitioning to mastery over a single
                observation.
        :guess: The probability of guessing correctly when learning.
        :slip: The probability of slipping incorrectly when mastered.

        """
        return cls([1.0 - init, init],
                   [[1.0 - trans, trans],
                    [0.0, 1.0]],
                   [[1.0 - guess, guess],
                    [slip, 1.0 - slip]])

    @classmethod
    def randomStudent(cls):
        """Random student model with semantically meaningful values.

        :returns: Random BKT model
        """

        init = random.uniform(0.0, 1.0)
        trans = random.uniform(0.0, 1.0)
        guess = random.uniform(0.0, 0.5)
        slip = random.uniform(0.0, 0.5)
        return cls.from_params(init, trans, guess, slip)

    def getInit(self):
        """Getter for initial mastery probability.
        :returns: Initial mastery probability.

        """
        return self.init[1]

    def getTrans(self):
        """Getter for transition to mastery probability.
        :returns: Transition to mastery probability.

        """
        return self.trans[0][1]

    def getGuess(self):
        """Getter for guess probability.
        :returns: Guess probability.

        """
        return self.emit[0][1]

    def getSlip(self):
        """Getter for slip probability.
        :returns: Slip probability.

        """
        return self.emit[1][0]

    def predict_correct(self, observations):
        """For each observation in each observation sequence, return probability
        correct given previous observations.

        :observations: List of observation sequences.
        :returns: List of lists of probabilities.

        """
        return [[dist[1] for dist in sequence_dists]
                for sequence_dists in self.predict(observations)]

    def predict_next_correct(self, observation_sequence):
        """Return probability correct given previous observations.

        :observation_sequence: Observation sequence.
        :returns: Probability that correct.

        """
        return self.predict_next(observation_sequence)[1]

    def predict_mastery(self, observations):
        """For each observation in each observation sequence, return probability
        that the student has mastered the skill given previous observations.

        :observations: List of observation sequences.
        :returns: List of lists of probabilities.

        """
        return [[dist[1] for dist in sequence_dists]
                for sequence_dists in self.predict_state(observations)]

    def predict_next_mastery(self, observation_sequence):
        """Return probability correct given previous observations.

        :observation_sequence: Observation sequence.
        :returns: Probability that correct.

        """
        return self.predict_next_state(observation_sequence)[1]

    def __str__(self):
        """Returns string view of object."""
        return "Init:{} Trans:{} Guess:{} Slip:{}".format(self.getInit(),
                                                          self.getTrans(),
                                                          self.getGuess(),
                                                          self.getSlip())
