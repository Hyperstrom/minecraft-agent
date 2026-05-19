import sys
sys.path.insert(0, '.')
from main import _safety_filter, ActionResponse
from prompt_builder import build_messages

# Test safety filter
a1 = ActionResponse(action='MINE', params={'block':'grass_block'}, source='llm')
r1 = _safety_filter(a1, 'collect 10 wood blocks')
print('Safety MINE(grass)->SEEK:', r1.action == 'SEEK', '| action:', r1.action, r1.params)

a2 = ActionResponse(action='ATTACK', params={}, source='llm')
r2 = _safety_filter(a2, 'survive')
print('Safety ATTACK->IDLE:     ', r2.action == 'IDLE', '| action:', r2.action)

a3 = ActionResponse(action='MINE', params={'block':'oak_log'}, source='llm')
r3 = _safety_filter(a3, 'collect 10 wood blocks')
print('Safety MINE(oak)->pass:  ', r3.action == 'MINE', '| action:', r3.action)

# Test prompt has progress/recent_actions fields
state = {
    'player': {'health':20,'food':20,'position':{'x':0,'y':64,'z':0}},
    'inventory': {'oak_log': 3},
    'nearby_blocks': [{'name':'oak_log','distance':2,'position':{}}],
    'nearby_entities': [],
    'environment': {'time_of_day':'noon','weather':'clear'},
    'goal_progress': '3/10 oak_log [XXXXOOOOO] 30%',
    'recent_actions': ['SEEK(target=oak_log)', 'MINE(block=oak_log)'],
}
msgs = build_messages(state, 'collect 10 wood blocks')
user_content = msgs[1]['content']
print('Progress in prompt:      ', '3/10' in user_content)
print('Recent actions in prompt:', 'SEEK' in user_content)
print('\nALL PHASE 2 CHECKS PASSED' if all([
    r1.action == 'SEEK', r2.action == 'IDLE', r3.action == 'MINE',
    '3/10' in user_content, 'SEEK' in user_content
]) else 'SOME CHECKS FAILED')
