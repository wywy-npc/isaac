"""Credential store — unified secret storage for connectors.

One encrypted file (~/.isaac/credentials.yaml) holds all API keys, tokens, and secrets.
No more scattered env vars. Backward compatible — env vars still work as fallback.

Encryption: Fernet symmetric encryption with a machine-derived key.
Not military-grade but prevents plaintext secrets on disk. Upgradeable to keychain.
"""
from __future__ import annotations

import base64
import hashlib
import os
import platform
from pathlib import Path
from typing import Any

import yaml

from isaac.core.config import ISAAC_HOME

CREDENTIALS_FILE = ISAAC_HOME / "credentials.yaml"


def _derive_key() -> bytes:
    """Derive a machine-specific encryption key.

    Uses ISAAC_HOME path + machine node as seed. Same machine, same key.
    Different machine, different key (credentials don't accidentally transfer).
    """
    seed = f"{ISAAC_HOME}:{platform.node()}:{os.getuid()}"
    key_bytes = hashlib.sha256(seed.encode()).digest()
    return base64.urlsafe_b64encode(key_bytes)


def _encrypt(value: str) -> str:
    """Encrypt a credential value."""
    try:
        from cryptography.fernet import Fernet
        f = Fernet(_derive_key())
        return f.encrypt(value.encode()).decode()
    except ImportError:
        # Fallback: base64 encoding (not secure, but better than plaintext)
        return "b64:" + base64.b64encode(value.encode()).decode()


def _decrypt(encrypted: str) -> str:
    """Decrypt a credential value."""
    if encrypted.startswith("b64:"):
        return base64.b64decode(encrypted[4:]).decode()
    try:
        from cryptography.fernet import Fernet
        f = Fernet(_derive_key())
        return f.decrypt(encrypted.encode()).decode()
    except (ImportError, Exception):
        return encrypted  # Return as-is if we can't decrypt


class CredentialStore:
    """Encrypted credential storage. One file, all secrets."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or CREDENTIALS_FILE
        self._cache: dict[str, dict[str, str]] | None = None

    def _load(self) -> dict[str, dict[str, str]]:
        """Load credentials from disk."""
        if self._cache is not None:
            return self._cache

        if not self._path.exists():
            self._cache = {}
            return self._cache

        with open(self._path) as f:
            raw = yaml.safe_load(f) or {}

        self._cache = raw.get("credentials", {})
        return self._cache

    def _save(self) -> None:
        """Write credentials to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"credentials": self._cache or {}}
        self._path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        # Restrict permissions (owner read/write only)
        self._path.chmod(0o600)

    def save(self, service: str, key: str, value: str) -> None:
        """Store a credential (encrypted)."""
        creds = self._load()
        if service not in creds:
            creds[service] = {}
        creds[service][key] = _encrypt(value)
        self._save()

    def save_many(self, service: str, credentials: dict[str, str]) -> None:
        """Store multiple credentials for a service."""
        for key, value in credentials.items():
            self.save(service, key, value)

    def get(self, service: str, key: str) -> str | None:
        """Get a decrypted credential. Returns None if not found."""
        creds = self._load()
        service_creds = creds.get(service, {})
        encrypted = service_creds.get(key)
        if encrypted:
            return _decrypt(encrypted)
        return None

    def get_all(self, service: str) -> dict[str, str]:
        """Get all decrypted credentials for a service."""
        creds = self._load()
        service_creds = creds.get(service, {})
        return {k: _decrypt(v) for k, v in service_creds.items()}

    def delete(self, service: str) -> bool:
        """Remove all credentials for a service."""
        creds = self._load()
        if service in creds:
            del creds[service]
            self._save()
            return True
        return False

    def list_services(self) -> list[str]:
        """List services that have stored credentials."""
        return list(self._load().keys())

    def has_credentials(self, service: str, required_keys: list[str]) -> bool:
        """Check if all required credentials exist for a service."""
        creds = self.get_all(service)
        return all(k in creds and creds[k] for k in required_keys)


def resolve_credential(service: str, key: str, store: CredentialStore | None = None) -> str:
    """Resolve a credential value. Checks store first, then env var.

    Resolution order:
    1. CredentialStore (encrypted on disk)
    2. Environment variable
    3. Empty string (missing)
    """
    # 1. Check credential store
    if store:
        value = store.get(service, key)
        if value:
            return value

    # 2. Check environment variable
    env_value = os.environ.get(key, "")
    if env_value:
        return env_value

    return ""
