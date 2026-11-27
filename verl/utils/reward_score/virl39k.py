# CVGA_VIRL39K_REWARD_V1
"""Answer-only reward for ViRL39K step-tagged, boxed-answer outputs.

Used by VERL reward loop for validation/telemetry. No task reward or
policy gradient is added to the CVGA-OPD objective by this module.
"""
from mathruler.grader import extract_boxed_content, grade_answer


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Return 1.0 for a correct boxed answer, 0.0 otherwise.

    Do not use Geo3K's old <think> format bonus: ViRL39K uses <step>.
    """
    predicted = extract_boxed_content(str(solution_str))
    if predicted is None or predicted == "":
        return 0.0
    return float(bool(grade_answer(predicted, str(ground_truth))))
