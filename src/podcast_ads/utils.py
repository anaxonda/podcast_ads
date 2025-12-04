def parse_timestamp(timestamp_str: str) -> float:
    """
    Converts HH:MM:SS or MM:SS string to seconds (float).
    Examples:
        "00:01:30" -> 90.0
        "1:30" -> 90.0
        "90" -> 90.0
    """
    if not timestamp_str:
        return 0.0
    
    parts = timestamp_str.strip().split(':')
    parts = [float(p) for p in parts]
    
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 1:
        return parts[0]
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")

def seconds_to_timestamp(seconds: float) -> str:
    """Converts seconds to HH:MM:SS string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{float(s):06.3f}"

def offset_timestamp(timestamp_str: str, offset_seconds: float) -> str:
    """Parses a timestamp, adds seconds, and returns the new string."""
    secs = parse_timestamp(timestamp_str)
    return seconds_to_timestamp(secs + offset_seconds)
