"""Hard output constraints are local and independent of any paid model."""
import pytest
from pipelines.ffmpeg.contracts import StageError
from pipelines.ffmpeg.goal_parser import parse_goal


@pytest.mark.parametrize('goal,target', [
    ('Family trip highlights', 45000), ('quick teaser', 30000), ('short teaser', 30000),
    ('longer teaser', 60000), ('extended teaser', 60000), ('5 seconds', 5000),
    ('120s', 120000), ('12.5 sec', 12500), ('1 minute', 60000), ('.5 minutes', 30000),
    ('30-60s', 45000), ('0.5–1 minutes', 45000), ('30 seconds to 1 minute', 45000),
    ('quick 50 seconds', 50000), ('30 seconds then 60 seconds', 60000),
    ('16:9 seconds', 45000), ('9:16 teaser', 45000), ('short then extended', 60000),
])
def test_duration(goal, target):
    assert parse_goal(goal).target_duration_ms == target


@pytest.mark.parametrize('goal,aspect', [('trip', '9:16'), ('portrait', '9:16'),
    ('vertical', '9:16'), ('16:9', '16:9'), ('landscape', '16:9'),
    ('horizontal', '16:9'), ('9:16 and 16:9', '16:9'), ('16:9 then vertical', '16:9')])
def test_aspects(goal, aspect):
    assert parse_goal(goal).aspect_ratio == aspect


@pytest.mark.parametrize('goal', ['4 seconds', '121s', '3 minutes', '1 hour',
    '-5 seconds', '30-130s', '60-30s', '4:3 teaser', 'square teaser', '5.0001s'])
def test_unsupported(goal):
    with pytest.raises(StageError) as error:
        parse_goal(goal)
    assert error.value.code == 'GOAL_CONSTRAINT_UNSUPPORTED'
    assert error.value.stage == 'planner'


def test_conflicts_and_named_requests():
    intent = parse_goal('30s then 60s birthday in Paris vertical and landscape')
    assert intent.requested_terms == ['birthday', 'paris']
    assert any('ignored' in a for a in intent.assumptions)
    assert any('Conflicting aspect' in a for a in intent.assumptions)
    assert intent == parse_goal('30s then 60s birthday in Paris vertical and landscape')


@pytest.mark.parametrize('goal', ['30--60s', '30-foo seconds'])
def test_malformed_numeric_ranges_fail(goal):
    with pytest.raises(StageError) as error:
        parse_goal(goal)
    assert error.value.code == 'GOAL_CONSTRAINT_UNSUPPORTED'


def test_hyphenated_duration():
    assert parse_goal('30-second teaser').target_duration_ms == 30000
