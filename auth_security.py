"""Authentication, authorization, and encryption helpers for the API."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from fastapi import Request
from fastapi.responses import JSONResponse

from monitoring_store import (
    create_user,
    get_user_by_username,
    get_user_by_id,
    list_users,
)

_current_user: ContextVar[dict[str, Any] | None] = ContextVar("current_user", default=None)
_ROLES = {"admin", "analyst", "viewer"}
_PUBLIC_PATHS = {"/", "/api/health", "/api/auth/login", "/docs", "/openapi.json", "/redoc"}
_ADMIN_ONLY_PATHS = (
    "/api/auth/users",
    "/api/configurations",
    "/api/alerts/rules",
    "/api/alerts/history",
    "/api/audit",
)
_READ_ONLY_POST_PATHS = {"/api/assistant/ask"}
_WRITE_PATHS = {
    "/api/monitoring/analyze",
    "/api/sources/analyze",
    "/api/notifications/send",
    "/api/datasets",
    "/api/datasets/compare",
    "/api/datasets/upload",
    "/api/schedules",
    "/api/data-quality/analyze",
    "/api/data-quality/analyze-csv",
    "/api/drift/analyze",
    "/api/drift/analyze-csv",
    "/api/analytics/trends",
    "/api/analytics/forecast",
    "/api/analytics/explain",
    "/api/analytics/advanced-ml",
}
MAX_PASSWORD_LENGTH = 1024


class AuthenticationError(ValueError):
    """Raised when stored user credentials or token data are invalid."""


class RequestBodyLimitMiddleware:
    def __init__(self, app, max_body_bytes: int = 10 * 1024 * 1024):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": "Request body exceeds 10 MiB"},
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                response = JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
                await response(scope, receive, send)
                return

        messages = []
        received_bytes = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    break
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        if received_bytes > self.max_body_bytes:
            response = JSONResponse(
                status_code=413,
                content={"detail": "Request body exceeds 10 MiB"},
            )
            await response(scope, receive, send)
            return

        message_index = 0

        async def replay_receive():
            nonlocal message_index
            if message_index < len(messages):
                message = messages[message_index]
                message_index += 1
                return message
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


def current_user() -> dict[str, Any]:
    user = _current_user.get()
    if user is None:
        raise AuthenticationError("An authenticated user is required")
    return user


def current_user_id() -> str:
    return str(current_user()["id"])


def current_user_scope() -> str | None:
    user = current_user()
    return None if user["role"] == "admin" else str(user["id"])


def login_attempt_key(client_ip: str) -> str:
    return hmac.new(_secret(), client_ip.encode("utf-8"), hashlib.sha256).hexdigest()


def _secret() -> bytes:
    value = os.getenv("AUTH_SECRET_KEY", "")
    if len(value.encode("utf-8")) < 32:
        raise AuthenticationError(
            "AUTH_SECRET_KEY must be configured with at least 32 characters"
        )
    return value.encode("utf-8")


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt, expected = encoded.split("$", 2)
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=_b64decode(salt), n=2**14, r=8, p=1
        )
        return hmac.compare_digest(actual, _b64decode(expected))
    except (ValueError, TypeError):
        return False


@lru_cache(maxsize=1)
def _dummy_password_hash() -> str:
    return hash_password(secrets.token_urlsafe(32))


def issue_token(user: dict[str, Any], lifetime_seconds: int = 3600) -> str:
    payload = {
        "sub": user["id"],
        "exp": int(time.time()) + lifetime_seconds,
        "jti": secrets.token_urlsafe(12),
    }
    body = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64encode(hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{signature}"


def _decode_token(token: str) -> dict[str, Any]:
    try:
        body, signature = token.split(".", 1)
        expected = hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64decode(signature)):
            raise AuthenticationError("Invalid bearer token")
        payload = json.loads(_b64decode(body))
        if not isinstance(payload, dict) or int(payload.get("exp", 0)) <= time.time():
            raise AuthenticationError("Bearer token has expired")
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        if isinstance(error, AuthenticationError):
            raise
        raise AuthenticationError("Invalid bearer token") from error


def encrypt_data(value: Any) -> str:
    from monitoring_store import _json_default

    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, default=_json_default
    ).encode("utf-8")
    return _fernet().encrypt(encoded).decode("ascii")


def _fernet() -> Fernet:
    return _derive_fernet(_secret())


@lru_cache(maxsize=1)
def _derive_fernet(secret: bytes) -> Fernet:
    key = base64.urlsafe_b64encode(
        hashlib.pbkdf2_hmac(
            "sha256", secret, b"monitoring-data-v1", 100_000, dklen=32
        )
    )
    return Fernet(key)


def decrypt_data(value: str) -> Any:
    try:
        decoded = _fernet().decrypt(value.encode("ascii"))
    except InvalidToken:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        raise AuthenticationError(
            "Stored dataset cannot be decrypted; verify AUTH_SECRET_KEY"
        ) from None
    return json.loads(decoded)


def encrypt_legacy_value(value: str) -> str:
    try:
        _fernet().decrypt(value.encode("ascii"))
    except (InvalidToken, UnicodeEncodeError):
        try:
            legacy_value = json.loads(value)
        except json.JSONDecodeError as error:
            raise AuthenticationError("Legacy stored data is not valid JSON") from error
        return encrypt_data(legacy_value)
    return value


def ensure_bootstrap_admin() -> None:
    if list_users(limit=1):
        return
    username = os.getenv("ADMIN_USERNAME", "").strip()
    password = os.getenv("ADMIN_PASSWORD", "")
    if username and password:
        if len(password) < 12 or len(password) > MAX_PASSWORD_LENGTH:
            raise AuthenticationError(
                f"ADMIN_PASSWORD must contain 12 to {MAX_PASSWORD_LENGTH} characters"
            )
        try:
            create_user(username.lower(), hash_password(password), "admin")
        except ValueError:
            if get_user_by_username(username) is None:
                raise


async def authentication_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 10 * 1024 * 1024:
                return JSONResponse(status_code=413, content={"detail": "Request body exceeds 10 MiB"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    try:
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthenticationError("A bearer access token is required")
        claims = _decode_token(token)
        user = get_user_by_id(str(claims.get("sub", "")))
        if user is None or not user["is_active"]:
            raise AuthenticationError("User is unavailable")

        path = request.url.path
        method = request.method.upper()
        if path == "/api/auth/me":
            pass
        elif path == "/api/auth/users" and method == "GET":
            if user["role"] != "admin":
                return JSONResponse(status_code=403, content={"detail": "Admin role required"})
        elif any(path == prefix or path.startswith(prefix + "/") for prefix in _ADMIN_ONLY_PATHS):
            if user["role"] != "admin":
                return JSONResponse(status_code=403, content={"detail": "Admin role required"})
        elif (
            method in {"PUT", "PATCH", "DELETE"}
            and path.startswith("/api/schedules/")
        ):
            pass
        elif (
            method == "POST"
            and path.startswith("/api/datasets/")
            and path.endswith("/versions")
            and user["role"] == "viewer"
        ):
            return JSONResponse(status_code=403, content={"detail": "Analyst role required"})
        elif method in {"PUT", "PATCH", "DELETE"} and user["role"] != "admin":
            return JSONResponse(status_code=403, content={"detail": "Admin role required"})
        elif method == "POST" and user["role"] == "viewer" and path not in _READ_ONLY_POST_PATHS:
            return JSONResponse(status_code=403, content={"detail": "Analyst role required"})
        elif (
            method == "POST"
            and user["role"] == "analyst"
            and path not in _WRITE_PATHS | _READ_ONLY_POST_PATHS | {"/api/auth/users"}
            and not (
                path.startswith("/api/datasets/")
                and path.endswith("/versions")
            )
        ):
            return JSONResponse(status_code=403, content={"detail": "Admin role required"})

        token_context = _current_user.set(user)
        try:
            return await call_next(request)
        finally:
            _current_user.reset(token_context)
    except AuthenticationError as error:
        status_code = 503 if "AUTH_SECRET_KEY" in str(error) else 401
        headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
        return JSONResponse(status_code=status_code, content={"detail": str(error)}, headers=headers)


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    ensure_bootstrap_admin()
    user = get_user_by_username(username)
    if user is None:
        verify_password(password, _dummy_password_hash())
        return None
    valid_password = verify_password(password, user["password_hash"])
    if not user["is_active"] or not valid_password:
        return None
    return user


def create_account(username: str, password: str, role: str) -> dict[str, Any]:
    normalized_username = username.strip().lower()
    if not normalized_username or len(normalized_username) > 128:
        raise ValueError("username must contain 1 to 128 characters")
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"password must not exceed {MAX_PASSWORD_LENGTH} characters")
    if role not in _ROLES - {"admin"}:
        raise ValueError("role must be analyst or viewer")
    return create_user(normalized_username, hash_password(password), role)
