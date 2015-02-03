"""
An container for predicting observations.
"""

from itertools import izip
import scipy.sparse as sparse


class EduLogreg(object):

    """Educational Logistic Regression Model container.

    """

    # Methods that this class requires from children.

    def get_logreg(self):
        """Returns contained logistic regression model.
        :returns: Logistic Regression Model

        """
        raise NotImplementedError

    def obs_to_features(self, students, skills, observations):
        """Turns students, skills, observations into features arrays and values
        arrays.

        :students: An iterator of students for each observation sequence.
        :skills: An iterator of skills for each observation sequence.
        :observations: An iterator over observation sequence iterators.
        :returns: (features, values)

        """
        raise NotImplementedError


    # Provided Methods


    def predict_observations(self, students, skills, observations):
        """Returns the probability of correct observation given previous
        observations.

        :students: The student of the observation.
        :skills: The skill of each observation.
        :observations: Iterator of observation sequences.
        :returns: Prediction array per observation sequence.

        """
        predictions = []
        for student, skill, obs_seq in izip(students, skills, observations):
            features, _ = self.obs_to_features([student], [skill], [obs_seq])
            predictions.append(self.get_logreg().featureProb(features))
        return predictions


    def train(self, students, skills, observations, stopping, learning_rate):
        """Trains the logistic regression model.

        :students: An iterator of students for each observation sequence.
        :skills: An iterator of skills for each observation sequence.
        :observations: An iterator over observation sequence iterators.

        """
        print "Begun Train"
        features, values = self.obs_to_features(students, skills,
                                                observations)
        print "Got feature array"
        return self.get_logreg().train(features, values, stopping,
                                       learning_rate)
