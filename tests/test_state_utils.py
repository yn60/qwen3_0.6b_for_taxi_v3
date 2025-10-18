from backend.taxi.state_utils import decode_state, describe_state_for_llm, get_prompt


def encode_state(taxi_row, taxi_col, passenger_location, destination_index):
    return (((taxi_row * 5) + taxi_col) * 5 + passenger_location) * 4 + destination_index


def test_decode_state():
    scenarios = [
        (0, 0, 0, 0),
        (4, 0, 2, 2),
        (2, 3, 4, 1),  # passenger already in taxi
    ]

    for taxi_row, taxi_col, passenger_location, destination_index in scenarios:
        state = encode_state(taxi_row, taxi_col, passenger_location, destination_index)
        assert decode_state(state) == {
            "taxi_row": taxi_row,
            "taxi_col": taxi_col,
            "passenger_location": passenger_location,
            "destination_index": destination_index,
        }


def test_describe_state_for_llm():
    decoded_state = {
        "taxi_row": 0,
        "taxi_col": 1,
        "passenger_location": 4,
        "destination_index": 0,
    }
    description = describe_state_for_llm(decoded_state)
    assert "The taxi is currently at position (0, 1)." in description
    assert "already in the taxi" in description

    decoded_state["passenger_location"] = 1
    description = describe_state_for_llm(decoded_state)
    assert "waiting at Green" in description
    assert "final destination is Red" in description


def test_get_prompt():
    state_description = "The taxi is currently at position (0, 0). The passenger is waiting at Green at location (0, 4)."
    prompt = get_prompt(state_description)
    assert "You are an expert Taxi-v3 agent." in prompt
    assert "# Environment Overview" in prompt
    assert state_description in prompt
    assert '"action": <one integer' in prompt