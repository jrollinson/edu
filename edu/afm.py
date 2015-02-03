"""
An Additive Factors Model implementation.
"""

from itertools import izip
import numpy as np
from StringIO import StringIO
import scipy.sparse as sparse
import json

from .edulogreg import EduLogreg
from logreg import LogisticRegressionModel


class AFM(EduLogreg):

    """Additive Factors Model"""

    # Initializers

    def __init__(self, student_idxs, skill_idxs, learning_rate_idxs,
                 logreg, lops_divisor=1):
        """Initializes values

        :student_idxs: Dictionary of students to indicies.
        :skill_idxs: Dictionary of skills to indicies.
        :learning_rate_idxs: Dictionary of skills to learning rate indicies.
        :logreg: Logistic Regression Model

        """
        EduLogreg.__init__(self)

        self._student_idx = student_idxs
        self._skill_idx = skill_idxs
        self._skill_learn_rate = learning_rate_idxs
        self._logreg = logreg
        self._lops_divisor = lops_divisor
        n_students = len(self._student_idx)
        n_skills = len(self._skill_idx)
        self._n_features = n_students + (2 * n_skills)


    @classmethod
    def from_lists(cls, students, skills, lops_divisor=1):
        """Initializes the model with students and skills

        :students: An iterator over students.
        :skills: An iterator over skills

        """
        students = set(students)
        skills = set(skills)

        student_idx = {student: idx
                       for idx, student in enumerate(students)}

        n_students = len(student_idx)
        skill_start = n_students
        skill_idx = {skill: idx + skill_start
                     for idx, skill in enumerate(skills)}

        n_skills = len(skill_idx)
        learn_rate_start = n_skills + skill_start
        skill_learn_rate = {skill: idx + learn_rate_start
                            for idx, skill in enumerate(skills)}


        n_features = n_students + (2 * n_skills)

        logreg = LogisticRegressionModel.fromZeros(n_features)

        return cls(student_idx, skill_idx, skill_learn_rate, logreg,
                   lops_divisor=lops_divisor)


    # Getters


    def get_student_idx(self, student):
        """Returns feature index.

        :student: The student.
        :returns: The index in the weight array.

        """
        return self._student_idx[student]


    def get_skill_idx(self, skill):
        """Returns index of skill parameter.

        :skill: The skill.
        :returns: The index in the weight array of the skill parameter.

        """
        return self._skill_idx[skill]


    def get_learning_rate_idx(self, skill):
        """Returns index of learning rate.

        :skill: The skill.
        :returns: The index in the weight array of the learning rate parameter.

        """
        return self._skill_learn_rate[skill]

    def get_student_weight(self, student):
        """Returns student weight.

        :student: Student whose weight to return.
        :returns: Weight

        """
        idx = self.get_student_idx(student)
        return self._logreg.featureWeights[idx]

    def get_skill_weight(self, skill):
        """Returns skill weight.

        :skill: Skill
        :returns: Weight

        """
        idx = self.get_skill_idx(skill)
        return self._logreg.featureWeights[idx]

    def get_learning_rate_weight(self, skill):
        """Returns learning rate weight for the skill.

        :skill: Skill
        :returns: Weight

        """
        idx = self.get_learning_rate_idx(skill)
        return self._logreg.featureWeights[idx]


    # Methods

    def get_feature_array(self, student, skill, n_lops, features=None):
        """Returns the feature array.

        :student: The student
        :skill: The skill
        :n_lops: The number of previous learning opportunities.
        :features: Array to fill in.
        :returns: A numpy array of features.
        """
        return self.get_feature_array_multskills(
                student, [skill], [n_lops],
                features=features)

    def get_feature_array_multskills(self, student, skills, n_lops_list,
                                     features=None):
        """Returns the feature array.

        :student: The student
        :skills: Skills to add
        :n_lops_list: The number of previous learning opportunities.
        :features: Array to fill in.
        :returns: A numpy array of features.
        """
        if features is None:
            features = np.zeros(self._n_features)

        features[self.get_student_idx(student)] = 1.0
        for skill, n_lops in izip(skills, n_lops_list):
            features[self.get_skill_idx(skill)] = 1.0

            lr_value = 1.0 * n_lops / self._lops_divisor
            features[self.get_learning_rate_idx(skill)] = lr_value
        return features

    def predict_observation(self, student, skill, n_lops):
        """Returns probability of correct observation.

        :student: The student.
        :skill: The skill.
        :n_lops: The number of previous learning opportunities.
        :returns: The probability of the next observation being correct.

        """
        features = self.get_feature_array(student, skill, n_lops)
        return self._logreg.featureProb(features)


    # EduLogreg necessary methods


    def get_logreg(self):
        """Returns contained logistic regression model.
        :returns: Logistic Regression Model

        """
        return self._logreg


    def obs_to_features(self, students, skills, observations):
        """Turns students, skills, observations into features arrays and values
        arrays.

        :students: An iterator of students for each observation sequence.
        :skills: An iterator of skills for each observation sequence.
        :observations: An iterator over observation sequence iterators.
        :returns: (features, values)

        """
        n_obs = 0
        for obs_seq in observations:
            n_obs += len(obs_seq)

        features = sparse.lil_matrix((n_obs, self._n_features))
        print features.shape
        values = np.zeros(n_obs)

        i = 0
        for student, skill, obs_seq in izip(students, skills, observations):
            for previous_lops, obs in enumerate(obs_seq):

                student_idx = self.get_student_idx(student)
                features[i, student_idx] = 1.0

                skill_idx = self.get_skill_idx(skill)
                features[i, skill_idx] = 1.0

                learn_idx = self.get_learning_rate_idx(skill)
                learn_val = 1.0 * previous_lops / self._lops_divisor
                features[i, learn_idx] = learn_val

                values[i] = obs

                i += 1

        assert i == n_obs

        features = sparse.csr_matrix(features)

        return features, values


    # I/O Methods


    def save(self, f):
        """Saves model to file

        :f: File.

        """
        string_f = StringIO()
        self._logreg.save(string_f)
        logreg_s = string_f.getvalue()
        string_f.close()

        json.dump({'student dict' : self._student_idx,
                   'skill dict' : self._skill_idx,
                   'learning dict' : self._skill_learn_rate,
                   'logreg' : logreg_s,
                   'lops divisor' : self._lops_divisor},
                  f)


    @classmethod
    def load(cls, f):
        """Loads model from file

        :f: File.
        :returns: model

        """
        data = json.load(f)
        student_dict = data['student dict']
        skill_dict = data['skill dict']
        learning_dict = data['learning dict']
        logreg_s = data['logreg']
        lops_divisor = data['lops divisor']

        logreg_f = StringIO(logreg_s)
        logreg = LogisticRegressionModel.load(logreg_f)
        logreg_f.close()

        return cls(student_dict, skill_dict, learning_dict, logreg,
                   lops_divisor=lops_divisor)
