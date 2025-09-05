import secrets
import string

def generate_secure_password(length=12):
    """
    Generate a secure password of a given length.

    Args:
    length (int): Length of the password to generate. Default is 12.

    Returns:
    str: A secure password.
    """
    if length < 8:
        raise ValueError("Password length should be at least 8 characters.")

    # Characters to use in the password
    characters = string.ascii_letters + string.digits + string.punctuation

    # Securely generate a random password
    password = ''.join(secrets.choice(characters) for i in range(length))

    return password

# Example usage
password = generate_secure_password(12)
print(password)
