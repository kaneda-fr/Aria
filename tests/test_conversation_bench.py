from scripts.e2e.conversation_bench import check_case


def events(commands=(), tools=()):
    return ([{'event': 'Live.Request'}, {'event': 'Live.Audio'}] +
            [{'event': 'Live.Tool', 'name': tool} for tool in tools] +
            [{'event': 'Live.Command', 'cmd_id': command, 'value': None, 'payload': '{}',
              'result': {'success': True, 'dry_run': True, 'executed': False}} for command in commands])


def test_wrong_opposite_or_duplicate_command_cannot_pass():
    case = {'kind': 'on', 'allow_search': True, 'answer': None,
            'expected_commands': [{'cmd_id': 'on-id', 'value': None, 'payload': {}}]}
    result = {'success': True, 'spoken': True, 'response': 'Recorded for testing.'}
    assert not check_case(case, result, events(['on-id']))
    for attempted in ([], ['off-id'], ['on-id','on-id']):
        assert 'command_trace_mismatch' in check_case(case, result, events(attempted))


def test_right_words_do_not_hide_unnecessary_tools_or_missing_audio():
    case = {'kind': 'fact', 'allow_search': False, 'answer': 'Rome', 'expected_commands': []}
    result = {'success': True, 'spoken': True, 'response': 'Rome.'}
    assert not check_case(case, result, events())
    assert 'unexpected_tool_call' in check_case(case, result, events(tools=['find_devices']))
    assert 'audio_missing' in check_case(case, result, [{'event':'Live.Request'}])


def test_transport_success_does_not_hide_wrong_answer_or_token_cutoff():
    case = {'kind': 'arithmetic', 'allow_search': False, 'answer': '84', 'expected_commands': []}
    result = {'success': True, 'spoken': True, 'response': 'I think it is 84 but'}
    checks = check_case(case, result, events()+[{'event':'Live.Model','done_reason':'length'}])
    assert 'answer_mismatch' in checks and 'generation_truncated' in checks


def test_spoken_tool_json_is_not_a_capability_answer():
    case = {'kind': 'capabilities', 'allow_search': False, 'answer': None, 'expected_commands': []}
    result = {'success': True, 'spoken': True,
              'response': '{"name":"find_devices","arguments":{}}>{"name":"execute_command"}'}
    assert 'capability_reply_is_tool_json' in check_case(case, result, events())
    result['response'] = 'find_devices and execute_command. Commands are only recorded in test mode.'
    assert not check_case(case, result, events())


def test_clock_must_match_request_interval_not_nearby_minutes():
    case = {'kind': 'time', 'allow_search': False, 'answer': None,
            'expected_commands': [], 'accepted_times': ['13:59', '14:00']}
    result = {'success': True, 'spoken': True, 'response': '14:01'}
    assert 'local_time_incorrect' in check_case(case, result, events())
    result['response'] = '14:00'
    assert not check_case(case, result, events())
