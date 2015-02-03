"""
Provides a Grouped Bayesian Knowledge Tracing Model.
"""

from hmm.hmm import HMM
from hmm.hmmRandom import randomDist

import random
from itertools import izip
class GroupedBKT(HMM):

    """A grouped Bayesian Knowledge Tracing model. """

    def __init__(self, init_ps, transs, guesses, slips, groupProbs):
        """
        :inits: Initial mastery probabilities.
        :transs: Probabilities of mastery over single observation.
        :guesses: Probabilities of correct guess while learning.
        :slips: Probabilities of incorrect slip when mastered.
        :groupProbs: Probability of each group.
        """
        assert (len(init_ps) ==
                len(transs) ==
                len(guesses) ==
                len(slips) ==
                len(groupProbs))

        nGroups = len(init_ps)
        nStates = 2 * nGroups
        nEvents = 2

        incorrect_i = 0
        correct_i = 1

        init = []
        for i_prob, group_prob in izip(init_ps, groupProbs):
            init_learning = (1.0 - i_prob) * group_prob
            init_mastered = i_prob * group_prob
            init.append(init_learning)
            init.append(init_mastered)

        trans = [[0.0 for _ in xrange(nStates)]
                 for _ in xrange(nStates)]
        for i, prob in enumerate(transs):
            learning_i = 2 * i
            mastered_i = (2 * i) + 1

            trans[learning_i][learning_i] = 1.0 - prob
            trans[learning_i][mastered_i] = prob
            trans[mastered_i][learning_i] = 0.0
            trans[mastered_i][mastered_i] = 1.0

        emit = [[0.0 for _ in xrange(nEvents)]
                for _ in xrange(nStates)]
        for i, guess, slip in izip(xrange(nGroups), guesses, slips):
            learning_i = 2 * i
            mastered_i = (2 * i) + 1

            emit[learning_i][incorrect_i] = 1.0 - guess
            emit[learning_i][correct_i] = guess
            emit[mastered_i][incorrect_i] = slip
            emit[mastered_i][correct_i] = 1.0 - slip

        HMM.__init__(self, init, trans, emit)

    @classmethod
    def randomGroups(cls, nGroups):
        """Creates nGroups random groups of students.

        :nGroups: The number of groups of students
        :returns: A random GroupedBKT

        """

        inits = [random.uniform(0.0, 1.0) for _ in xrange(nGroups)]
        transs = [random.uniform(0.0, 1.0) for _ in xrange(nGroups)]
        guesses = [random.uniform(0.0, 0.5) for _ in xrange(nGroups)]
        slips = [random.uniform(0.0, 0.5) for _ in xrange(nGroups)]
        groupProbs = randomDist(nGroups)

        return GroupedBKT(inits, transs, guesses, slips, groupProbs)

    def predict_correct(self, observations):
        """For each observation in each observation sequence, return probability
        correct given previous observations.

        :observations: List of observation sequences.
        :returns: List of lists of probabilities.

        """

        return [[dist[1] for dist in sequence_dists]
                for sequence_dists in self.predict(observations)]

    def predict_mastery(self, observations):
        """For each observation in each observation sequence, return probability
        that the student has mastered the skill given previous observations.

        :observations: List of observation sequences.
        :returns: List of lists of probabilities.

        """
        odd_sum = lambda xs: sum(x for i, x in enumerate(x) if i % 2 == 1)

        return [[odd_sum(dist) for dist in sequence_dists]
                for sequence_dists in self.predict_state(observations)]

    def predict_group(self, observations):
        """For each observation in each observation sequence, return probability
        that the student has mastered the skill given previous observations.

        :observations: List of observation sequences.
        :returns: List of lists of probabilities.

        """

        state_predictions = self.predict_state(observations)

        group_predictions = []
        n_groups = len(self.init) / 2
        for group in xrange(n_groups):
            group_predictions[group] = (state_predictions[2 * group] +
                                        state_predictions[(2 * group) + 1])
        return group_predictions
