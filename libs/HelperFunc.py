import re

def encode_float_filename(number: float) -> str:
    """
    Encodes a float into a filename-safe string by combining zero-padding and scientific notation.
    - If number is between 0.01 and 9999, use fixed-point format with zero-padding.
    - Otherwise, use scientific notation with a safe separator.
    """
    if 1 <= abs(number) < 0.1:
        formatted = f"{number:08.2f}".replace('.', '_')  # Zero-padding, 2 decimal places
    else:
        formatted = f"{number:.2e}".replace('.', '_').replace('+', '')  # Scientific notation
    
    return f"{formatted}.txt"

def decode_float_filename(filename: str) -> float:
    """
    Decodes a filename back into a float by identifying the encoding method used.
    - Detects scientific notation and fixed-point encoding.
    """
    match = re.search(r"([0-9eE_\-]+)\.txt", filename)
    if not match:
        raise ValueError("Invalid filename format")

    encoded_number = match.group(1)
    
    # Check if it's scientific notation (contains 'e' or 'E')
    if 'e' in encoded_number or 'E' in encoded_number:
        decoded = float(encoded_number.replace('_', '.'))
    else:
        decoded = float(encoded_number.replace('_', '.'))  # Convert back to float
    
    return decoded
