"""
auth.py — Signup and login helpers using bcrypt password hashing.
"""
import bcrypt
import re
from database import create_user, find_user_by_email


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(plain: str, hashed: str) -> bool:
    """Return True if the plain password matches the stored hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def _is_valid_email(email: str) -> bool:
    """Basic email format check."""
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", email.strip()))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def signup(full_name: str, email: str, password: str) -> tuple[bool, str, dict | None]:
    """
    Register a new user.
    Returns (success: bool, message: str, user_doc: dict | None).
    """
    full_name = full_name.strip()
    email     = email.strip()

    # Validation
    if not full_name:
        return False, "Full name is required.", None
    if not _is_valid_email(email):
        return False, "Please enter a valid email address.", None
    if len(password) < 8:
        return False, "Password must be at least 8 characters.", None

    hashed = _hash_password(password)
    user   = create_user(full_name, email, hashed)

    if user is None:
        return False, "An account with this email already exists.", None

    return True, f"Welcome, {full_name}! Your account has been created.", user


def login(email: str, password: str) -> tuple[bool, str, dict | None]:
    """
    Authenticate a user.
    Returns (success: bool, message: str, user_doc: dict | None).
    """
    user = find_user_by_email(email.strip())

    if user is None:
        return False, "No account found with that email.", None

    if not _check_password(password, user["password_hash"]):
        return False, "Incorrect password.", None

    return True, f"Welcome back, {user['full_name']}!", user
