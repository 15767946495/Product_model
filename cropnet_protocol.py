"""Shared CropNet data protocol constants and validation helpers."""

ALLOWED_STATES = {
    "minnesota", "wisconsin", "michigan", "illinois",
    "indiana", "ohio", "missouri", "kentucky",
}
START_MONTH = 4
END_MONTH = 9
DAYS_PER_MONTH = 28
PROTOCOL_MAX_STEPS = (END_MONTH - START_MONTH + 1) * DAYS_PER_MONTH
TIME_WINDOW = "04-01--09-28"

# Explicit protocol-prefixed aliases keep consumer APIs descriptive while the
# values remain defined only once above.
PROTOCOL_START_MONTH = START_MONTH
PROTOCOL_END_MONTH = END_MONTH
PROTOCOL_DAYS_PER_MONTH = DAYS_PER_MONTH


def validate_calendar_fields(month, day, l_enc):
    """Validate aligned calendar fields and the protocol time-step limit."""
    if len(month) != len(day) or len(month) != l_enc:
        raise ValueError("month, day, and l_enc must have the same length")
    if l_enc > PROTOCOL_MAX_STEPS:
        raise ValueError("l_enc must be at most 168")
    if any(not START_MONTH <= value <= END_MONTH for value in month):
        raise ValueError("month values must be within the protocol window")
    if any(not 1 <= value <= DAYS_PER_MONTH for value in day):
        raise ValueError("day values must be within the protocol window")
