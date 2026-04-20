_WINDOW_ATTACK_EVENTS = []


def record_attack_event(attack_type, satellite_name, metadata=None):
    event = {
        'attack_type': str(attack_type),
        'satellite_name': str(satellite_name) if satellite_name is not None else 'unknown_satellite',
        'metadata': dict(metadata or {}),
    }
    _WINDOW_ATTACK_EVENTS.append(event)


def consume_window_attack_events():
    events = list(_WINDOW_ATTACK_EVENTS)
    _WINDOW_ATTACK_EVENTS.clear()
    return events


def reset_window_attack_events():
    _WINDOW_ATTACK_EVENTS.clear()
