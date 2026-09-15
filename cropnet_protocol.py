"""Shared CropNet data protocol constants and validation helpers."""

ALLOWED_STATES = {
    "minnesota", "wisconsin", "michigan", "illinois",
    "indiana", "ohio", "missouri", "kentucky",
}
CROPNET_FIVE_STATES = {
    "illinois", "iowa", "louisiana", "mississippi", "new_york",
}
START_MONTH = 4
END_MONTH = 9
DAYS_PER_MONTH = 28
PROTOCOL_MAX_STEPS = (END_MONTH - START_MONTH + 1) * DAYS_PER_MONTH
TIME_WINDOW = "04-01--09-28"
PROTOCOL_VERSION = 4

EXPECTED_CALENDAR = tuple(
    (month, day)
    for month in range(START_MONTH, END_MONTH + 1)
    for day in range(1, DAYS_PER_MONTH + 1)
)

# Explicit protocol-prefixed aliases keep consumer APIs descriptive while the
# values remain defined only once above.
PROTOCOL_START_MONTH = START_MONTH
PROTOCOL_END_MONTH = END_MONTH
PROTOCOL_DAYS_PER_MONTH = DAYS_PER_MONTH


def validate_grid_entry(entry, index=None):
    """Validate one version-4 grid cache entry against the short-season protocol."""
    prefix = f"grid cache entry {index}" if index is not None else "grid cache entry"
    if not isinstance(entry, dict):
        raise ValueError(f"{prefix} must be a dict")

    try:
        feats = entry["feats"]
        month = entry["month"]
        day = entry["day"]
        l_enc = int(entry["l_enc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{prefix} missing or invalid required field: {error}") from error

    if not hasattr(feats, "shape") or len(feats.shape) < 2:
        raise ValueError(f"{prefix} feats must have at least 2 dimensions")
    if feats.shape[1] != PROTOCOL_MAX_STEPS:
        raise ValueError(f"{prefix} feats time dimension must be exactly {PROTOCOL_MAX_STEPS}")
    if l_enc != PROTOCOL_MAX_STEPS:
        raise ValueError(f"{prefix} l_enc must be exactly {PROTOCOL_MAX_STEPS}")

    for name, values, low, high in (("month", month, START_MONTH, END_MONTH), ("day", day, 1, DAYS_PER_MONTH)):
        if not hasattr(values, "shape") or len(values.shape) != 1:
            raise ValueError(f"{prefix} {name} shape must be one-dimensional")
        if values.shape[0] != PROTOCOL_MAX_STEPS:
            raise ValueError(
                f"{prefix} {name} length must be exactly {PROTOCOL_MAX_STEPS}"
            )
        values_list = values.tolist() if hasattr(values, "tolist") else list(values)
        if any(isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high for value in values_list):
            raise ValueError(f"{prefix} {name} values must be within {low}..{high}")
    validate_expected_calendar(month.tolist(), day.tolist(), prefix=prefix)


def validate_expected_calendar(month, day, prefix="calendar"):
    """Require the exact ordered 4/1--9/28, 168-step calendar."""
    if len(month) != len(day):
        raise ValueError(f"{prefix} month/day lengths must match")
    if len(month) != PROTOCOL_MAX_STEPS:
        raise ValueError(f"{prefix} must contain exactly {PROTOCOL_MAX_STEPS} steps")
    pairs = tuple(zip(month, day))
    if pairs != EXPECTED_CALENDAR:
        raise ValueError(f"{prefix} must equal the exact 4/1--9/28 calendar")


def protocol_metadata() -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "time_window": TIME_WINDOW,
        "days_per_month": DAYS_PER_MONTH,
        "max_steps": PROTOCOL_MAX_STEPS,
    }


def validate_protocol_metadata(payload, prefix="cache"):
    for field, expected in protocol_metadata().items():
        if payload.get(field) != expected:
            label = {"version": "version 4", "time_window": "time window"}.get(field, field)
            raise ValueError(f"{prefix} {label} mismatch: expected {expected!r}")


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
