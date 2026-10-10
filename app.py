"""
PosterProof — Evidence-First Event Poster Audit.
Compare a printed poster photo against an approved master to surface evidence of changes.
"""

import base64
import cgi
import html
import io
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class AuditError(Exception):
    """Raised when audit input is invalid or evidence is insufficient to compare."""
    pass
