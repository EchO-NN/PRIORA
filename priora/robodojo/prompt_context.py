"""Version of teacher-visible evaluation guidance, independent of model/source."""

CONTEXT_VERSION = 'v4'

# Public task semantics only: no layout truth, score queries or action script.
TASK_NOTES = {
    'organize_table': (
        'Place the alarm clock upright on top of the drawer cabinet: its upright axis must remain '
        'less than 15 degrees from world vertical. Keep its base down; a clock lying with its dial '
        'facing upward or downward is not complete even when it is on the cabinet. '
        'Place the figurine upright and centered on its stand, the mouse on its pad, and the keyboard '
        'flat and aligned over the marked rectangular target. After releasing and withdrawing, '
        'visually verify that the placements remain stable and upright. All four placement conditions '
        'must hold together when both arms return to their initial poses. '
        'The drawer is static; opening it or storing additional objects inside is not part of this task.'
    ),
    'classify_objects_by_language': 'Follow this episode\'s language-specified category-to-basket mapping for every object.',
    'imitate_sorting_sequence': 'Observe and remember the demonstration before placing the corresponding objects in the same order.',
    'arrange_largest_number': 'Arrange the digits left to right as the largest number, correctly oriented and placed on their pads.',
    'pack_objects_into_box': 'Place every object inside the box with its front facing left, as instructed.',
    'classify_objects': 'Group all objects by category into the three baskets.',
    'build_tower': 'Build the supported tower layers using the wooden blocks and boards.',
    'make_kong': 'Wait for the discard, expose the matching tiles, and finish the tile-handling sequence without disturbing the other tiles.',
    'fold_clothes': 'Fold the sleeves inward and the lower garment upward into a neat folded shape.',
    'put_bottles_into_dustbin': 'Put all bottles inside the dustbin.',
}


def task_context(task):
    base = task.removesuffix('_random')
    if base not in TASK_NOTES:
        return {}
    return dict(process=TASK_NOTES[base], requires_arm_return=base != 'make_kong')
