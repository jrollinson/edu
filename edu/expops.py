"""
File: expops.py
Author: Joseph Rollinson
Email: jtrollinson@gmail.com
Description: Expected Learning Opportunity functions.
"""

def expops(predict_correct, stop_f, start_state, update_state, path_threshold,
           max_len=100):
    """A higher order function that returns the expected number of learning
    opportunities required to before stop_f(path) is true.

    :predict_correct: Function that takes in observation sequence and returns
    the probability that the next observation is correct.
    :stop_f: Function with argument path that returns whether to stop
    or not.
    :start_state: Starting state of the world for the predictor.
    :updated_state: Function for updating the state of the world.
    :path_threshold: Probability threshold before stopping down a path.
    :returns: Approximate expected number of learning opportunities required.

    """
    def inner_expops(state, p_path, length):
        """Recursive inner function of expops."""

        # We do p_path threshold first so that we don't attempt to deal with
        # state when p_path = 0
        if (p_path < path_threshold) or (length >= max_len) or stop_f(state):
            return 0.0
        else:

            p_correct = predict_correct(state)
            assert 0.0 <= p_correct <= 1.0, "P(correct) + {}".format(p_correct)

            if p_correct > 0.0:
                p_path_and_c = p_path * p_correct
                state_after_correct = update_state(state, 1)
                expops_given_c = inner_expops(state_after_correct, p_path_and_c,
                                              length + 1)
            else:
                expops_given_c = 0.0

            if p_correct < 1.0:
                p_path_and_w = p_path * (1 - p_correct)
                state_after_wrong = update_state(state, 0)
                expops_given_w = inner_expops(state_after_wrong, p_path_and_w,
                                              length + 1)
            else:
                expops_given_w = 0.0

            return (1 +
                    (p_correct * expops_given_c) +
                    ((1 - p_correct) * expops_given_w))

    return inner_expops(start_state, 1.0, 0)


def expops_mastery(predict_correct, predict_mastery, start_state, update_state,
                   mastery_threshold, path_threshold):
    """Calculates the expected number of learning opportunities required to
    mastery a subject.

    :predict_correct: Function that takes in state and returns
    the probability that the next observation is correct.
    :predict_mastery: Function that predicts mastery given state.
    :start_state: Start state for predictor.
    :updated_state: Function for updating state of the world.
    :mastery_threshold: Threshold to consider mastery.
    :path_threshold: Threshold before stopping path.
    :returns: Expected number of learning opportunities.

    """
    stop_f = lambda s: predict_mastery(s) >= mastery_threshold
    return expops(predict_correct, stop_f, start_state, update_state,
                  path_threshold)

def similarity_stop(predict_correct, state, update_state, similarity_threshold,
                    confidence_threshold):
    """Returns True if should stop, False otherwise.

    :predict_correct: Function that takes in an observation sequence and returns
    probability that the next observation will be 'correct'.
    :state: State of the world.
    :update_state: Function that returns updated state of the world.
    :similarity_threshold: Threshold on similarity.
    :confidence_threshold: Threshold on the confidence that too similar.
    :returns: True if should stop, False otherwise.

    """
    current_p_c = predict_correct(state)
    current_p_w = 1 - current_p_c

    p_too_close = 0.0

    if current_p_c > 0.0:
        state_after_correct = update_state(state, 1)
        p_c_after_c = predict_correct(state_after_correct)

        if abs(current_p_c - p_c_after_c) < similarity_threshold:
            p_too_close += current_p_c

    if current_p_w > 0.0:
        state_after_wrong = update_state(state, 0)
        p_c_after_w = predict_correct(state_after_wrong)
        if abs(current_p_c - p_c_after_w) < similarity_threshold:
            p_too_close += current_p_w


    return p_too_close > confidence_threshold

def expops_similarity(predict_correct, start_state, update_state,
                      similarity_threshold, confidence_threshold,
                      path_threshold):
    """Returns the expected number of learning opportunities required for the
    similarity threshold.

    :predict_correct: Function that takes in observation sequence and returns
    the probability that the next observation is correct.
    :similarity_threshold: Similarity Threshold.
    :confidence_threshold: Confidence Threshold.
    :path_threshold: Path Threshold.
    :returns: Expected number of learning opportunities.

    """

    stop_f = lambda state: similarity_stop(predict_correct, state, update_state,
                                           similarity_threshold,
                                           confidence_threshold)

    return expops(predict_correct, stop_f, start_state, update_state,
                  path_threshold)


def expops_expsim(predict_correct, start_state, update_state,
                similarity_threshold, path_threshold):
    """
    Calculates the expected number of learning opportunities using expected
    similarity as the stopping policy.
    """
    def stop_f(state):
        """
        Returns true if should stop in given state.
        """
        p_correct = predict_correct(state)

        expected_p_correct = 0.0

        if p_correct > 0.0:
            state_after_correct = update_state(state, 1)
            p_correct_after_c = predict_correct(state_after_correct)

            expected_p_correct += p_correct * p_correct_after_c

        if p_correct < 1.0:
            state_after_wrong = update_state(state, 0)
            p_correct_after_w = predict_correct(state_after_wrong)

            expected_p_correct += (1 - p_correct) * p_correct_after_w

        expected_similarity = abs(p_correct - expected_p_correct)
        return expected_similarity < similarity_threshold

    return expops(predict_correct, stop_f, start_state, update_state,
                  path_threshold)




