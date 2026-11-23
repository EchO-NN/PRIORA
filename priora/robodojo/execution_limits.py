"""Execution budgets for fresh proposals; native episode limits still apply."""

STUDENT_MAX_STEPS = 30
EEF_MAX_STEPS = 30
EDIT_MAX_STEPS = 5
DIRECT_EEF_MAX_STEPS = 5
HYBRID_STEP_LIMITS = dict(student=STUDENT_MAX_STEPS, eef=EEF_MAX_STEPS,
                          edit=EDIT_MAX_STEPS)
