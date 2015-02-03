"""
A Performance Factors Model implementation.
"""

from .edulogreg import EduLogreg
from logreg import LogisticRegressionModel
import scipy.sparse as sparse

from itertools import izip
from StringIO import StringIO
import json
import numpy as np

class PFM(EduLogreg):
    """Performance Factors Model"""

    # Initializers


    def __init__(self, student_idxs, skill_idxs, success_idxs, failure_idxs,
                 logreg, lops_divisor=1):
        EduLogreg.__init__(self)

        self._student_idx = student_idxs
        self._skill_idx = skill_idxs
        self._success_idx = success_idxs
        self._failure_idx = failure_idxs
        self._logreg = logreg
        self._lops_divisor = lops_divisor

        n_students = len(self._student_idx)
        n_skills = len(self._skill_idx)
        self._n_features = n_students + (3 * n_skills)

    @classmethod
    def from_lists(cls, students, skills, lops_divisor=1):
        """Creates PFM from lists of students and skills.

        :students: Iterator of students.
        :skills: Iterator of skills.
        :returns: PFM

        """

        student_idxs = {student: idx
                        for idx, student in enumerate(students)}
        n_students = len(student_idxs)

        skill_start = n_students
        skill_idxs = {skill: idx + skill_start
                      for idx, skill in enumerate(skills)}
        n_skills = len(skill_idxs)

        success_start = n_skills + skill_start
        success_idxs = {skill: idx + success_start
                        for idx, skill in enumerate(skills)}

        failure_start = n_skills + success_start
        failure_idxs = {skill: idx + failure_start
                        for idx, skill in enumerate(skills)}

        n_features = n_students + (3 * n_skills)

        logreg = LogisticRegressionModel.fromZeros(n_features)

        return cls(student_idxs, skill_idxs, success_idxs, failure_idxs,
                   logreg, lops_divisor=lops_divisor)

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


    def get_success_idx(self, skill):
        """Returns index of success parameter

        :skill: The skill.
        :returns: Success parameter

        """
        return self._success_idx[skill]


    def get_failure_idx(self, skill):
        """Returns index of failure parameter

        :skill: The skill.
        :returns: Failure parameter

        """
        return self._failure_idx[skill]


    # Methods

    def feature_array(self, student, skill, n_success, n_failure):
        """Returns the feature array

        :student: The student performing the question.
        :skill: The skill being tested.
        :n_success: The number of previous successes.
        :n_failure: The number of previous failures.
        :returns: A feature array for the logistic regression model.

        """
        features = np.zeros(self._n_features)
        features[self.get_student_idx(student)] = 1.0
        features[self.get_skill_idx(skill)] = 1.0

        success_feature = 1.0 * n_success / self._lops_divisor
        failure_feature = 1.0 * n_failure / self._lops_divisor
        features[self.get_success_idx(skill)] = success_feature
        features[self.get_failure_idx(skill)] = failure_feature
        return features

    # EduLogreg methods


    def get_logreg(self):
        """Returns contained logistic regression model.
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
        values = np.zeros(n_obs)

        i = 0
        for student, skill, obs_seq in izip(students, skills, observations):

            n_success = 0
            n_failure = 0

            for obs in obs_seq:

                values[i] = obs

                student_idx = self.get_student_idx(student)
                features[i, student_idx] = 1.0

                skill_idx = self.get_skill_idx(skill)
                features[i, skill_idx] = 1.0

                success_idx = self.get_success_idx(skill)
                success_val = 1.0 * n_success / self._lops_divisor
                features[i, success_idx] = success_val

                failure_idx = self.get_failure_idx(skill)
                failure_val = 1.0 * n_failure / self._lops_divisor
                features[i, failure_idx] = failure_val

                i += 1

                if obs == 0:
                    n_failure += 1
                else:
                    assert obs == 1
                    n_success += 1

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
                   'success dict' : self._success_idx,
                   'failure dict' : self._failure_idx,
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
        success_dict = data['success dict']
        failure_dict = data['failure dict']
        logreg_s = data['logreg']
        lops_divisor = data['lops divisor']

        logreg_f = StringIO(logreg_s)
        logreg = LogisticRegressionModel.load(logreg_f)
        logreg_f.close()

        return cls(student_dict, skill_dict, success_dict, failure_dict,
                   logreg, lops_divisor=lops_divisor)
