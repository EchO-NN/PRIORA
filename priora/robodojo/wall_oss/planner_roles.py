"""Enable EEF IK for policy arms without changing native support-arm roles."""


def configure_planner_roles(robots, student_backend):
    for robot in robots:
        support = robot.get('type') == 'support'
        # Support arms replay native joint demonstrations unless the task
        # explicitly requests a planner. Do not construct unused IK solvers.
        enabled = student_backend == 'wall_oss' and (
            not support or bool(robot.get('need_planner', False)))
        robot['need_planner'] = enabled
