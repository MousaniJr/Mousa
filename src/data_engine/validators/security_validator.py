"""
Security Validator - Scans code for security vulnerabilities and secrets

Checks for:
- Hardcoded credentials (API keys, passwords, tokens)
- SQL injection patterns
- XSS vulnerabilities
- Insecure cryptography
- Common security anti-patterns
"""

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """Security issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIssue:
    """Represents a detected security issue"""
    severity: Severity
    category: str
    description: str
    line_number: int
    code_snippet: str
    recommendation: str


class SecurityValidator:
    """
    Validates code for security vulnerabilities
    """

    # Patterns for detecting hardcoded secrets
    SECRET_PATTERNS = [
        # API Keys
        (r'api[_-]?key\s*=\s*["\']([a-zA-Z0-9_\-]{20,})["\']',
         "Hardcoded API key", Severity.CRITICAL),

        # AWS Keys
        (r'AKIA[0-9A-Z]{16}',
         "AWS Access Key ID", Severity.CRITICAL),

        # Generic secrets
        (r'secret\s*=\s*["\']([^"\']{10,})["\']',
         "Hardcoded secret", Severity.HIGH),

        # Passwords
        (r'password\s*=\s*["\']([^"\']{1,})["\']',
         "Hardcoded password", Severity.CRITICAL),

        # Private keys
        (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
         "Private key in code", Severity.CRITICAL),

        # Database connection strings with passwords
        (r'(mysql|postgresql|mongodb)://[^:]+:([^@]+)@',
         "Database password in connection string", Severity.CRITICAL),

        # Bearer tokens
        (r'bearer\s+[a-zA-Z0-9_\-\.]{20,}',
         "Bearer token", Severity.HIGH),
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        (r'execute\s*\(\s*f?["\'].*?\{.*?\}.*?["\']',
         "Potential SQL injection via f-string", Severity.HIGH),

        (r'execute\s*\(\s*["\'].*?%s.*?["\'].*?%',
         "Potential SQL injection via string formatting", Severity.HIGH),

        (r'(SELECT|INSERT|UPDATE|DELETE).*?\+.*?[\w_]+',
         "SQL query concatenation", Severity.HIGH),

        (r'execute\s*\(\s*.*?\+\s*[\w_]+',
         "SQL query string concatenation", Severity.CRITICAL),
    ]

    # XSS patterns
    XSS_PATTERNS = [
        (r'innerHTML\s*=\s*[\w_]+',
         "Potential XSS via innerHTML", Severity.HIGH),

        (r'document\.write\s*\([\w_]+',
         "Potential XSS via document.write", Severity.MEDIUM),

        (r'eval\s*\(',
         "Use of eval() - potential code injection", Severity.HIGH),

        (r'exec\s*\([\w_]+\)',
         "Use of exec() - potential code injection", Severity.HIGH),
    ]

    # Insecure cryptography
    CRYPTO_PATTERNS = [
        (r'hashlib\.(md5|sha1)\(',
         "Weak hash function for passwords", Severity.MEDIUM),

        (r'Cipher\.(DES|RC4)',
         "Weak encryption algorithm", Severity.HIGH),

        (r'random\.random\(\)',
         "Insecure random for security", Severity.MEDIUM),
    ]

    def __init__(self, strict_mode: bool = False):
        """
        Initialize security validator

        Args:
            strict_mode: If True, fail on any security issue
        """
        self.strict_mode = strict_mode
        self.issues_found: List[SecurityIssue] = []

    def scan_file(self, code: str, language: str) -> Tuple[bool, List[SecurityIssue]]:
        """
        Scan code for security issues

        Args:
            code: Code content
            language: Programming language

        Returns:
            Tuple of (is_safe, list of issues)
        """
        self.issues_found = []
        lines = code.split('\n')

        # Scan for secrets
        self._scan_patterns(
            code, lines, self.SECRET_PATTERNS, "Secrets Detection"
        )

        # Scan for SQL injection (for relevant languages)
        if language in ["Python", "JavaScript", "PHP", "Java", "C#"]:
            self._scan_patterns(
                code, lines, self.SQL_INJECTION_PATTERNS, "SQL Injection"
            )

        # Scan for XSS (for web languages)
        if language in ["JavaScript", "TypeScript", "PHP"]:
            self._scan_patterns(
                code, lines, self.XSS_PATTERNS, "XSS"
            )

        # Scan for weak cryptography
        if language in ["Python", "JavaScript", "Java", "C#"]:
            self._scan_patterns(
                code, lines, self.CRYPTO_PATTERNS, "Weak Cryptography"
            )

        # Determine if file passes validation
        is_safe = self._is_safe()

        return is_safe, self.issues_found

    def _scan_patterns(
        self,
        code: str,
        lines: List[str],
        patterns: List[Tuple],
        category: str
    ):
        """
        Scan code against a list of patterns

        Args:
            code: Full code content
            lines: Code split into lines
            patterns: List of (pattern, description, severity) tuples
            category: Issue category
        """
        for pattern, description, severity in patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                # Find line number
                line_number = code[:match.start()].count('\n') + 1

                # Get code snippet (3 lines of context)
                start_line = max(0, line_number - 2)
                end_line = min(len(lines), line_number + 1)
                snippet = '\n'.join(lines[start_line:end_line])

                # Generate recommendation
                recommendation = self._get_recommendation(description, severity)

                issue = SecurityIssue(
                    severity=severity,
                    category=category,
                    description=description,
                    line_number=line_number,
                    code_snippet=snippet,
                    recommendation=recommendation
                )

                self.issues_found.append(issue)

    def _get_recommendation(self, description: str, severity: Severity) -> str:
        """
        Get recommendation for fixing a security issue

        Args:
            description: Issue description
            severity: Issue severity

        Returns:
            Recommendation string
        """
        recommendations = {
            "Hardcoded API key": "Store API keys in environment variables or a secure vault",
            "AWS Access Key ID": "Use AWS IAM roles instead of hardcoded credentials",
            "Hardcoded password": "Use environment variables or a password manager",
            "Hardcoded secret": "Store secrets in environment variables or a secrets manager",
            "Private key in code": "Store private keys in secure key storage, never in code",
            "Database password in connection string": "Use environment variables for DB credentials",
            "SQL injection": "Use parameterized queries or an ORM",
            "Potential XSS": "Sanitize user input and use safe DOM manipulation methods",
            "Weak hash function": "Use bcrypt, scrypt, or Argon2 for password hashing",
            "Weak encryption": "Use AES-256 or ChaCha20 for encryption",
            "Insecure random": "Use secrets.SystemRandom() for security-critical randomness",
        }

        for key, recommendation in recommendations.items():
            if key.lower() in description.lower():
                return recommendation

        return "Review this code for security best practices"

    def _is_safe(self) -> bool:
        """
        Determine if code is safe based on issues found

        Returns:
            True if safe, False otherwise
        """
        if not self.issues_found:
            return True

        # In strict mode, any issue fails validation
        if self.strict_mode:
            return False

        # Otherwise, only critical/high severity issues fail validation
        critical_issues = [
            issue for issue in self.issues_found
            if issue.severity in [Severity.CRITICAL, Severity.HIGH]
        ]

        return len(critical_issues) == 0

    def get_report(self) -> str:
        """
        Get a formatted security report

        Returns:
            Formatted report string
        """
        if not self.issues_found:
            return "✅ No security issues found"

        report = ["\n🔒 Security Scan Report\n" + "=" * 50]

        # Group issues by severity
        by_severity = {}
        for issue in self.issues_found:
            severity = issue.severity.value
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(issue)

        # Report each severity level
        severity_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        for severity in severity_order:
            issues = by_severity.get(severity.value, [])
            if not issues:
                continue

            report.append(f"\n{severity.value.upper()}: {len(issues)} issue(s)")
            for i, issue in enumerate(issues, 1):
                report.append(f"\n{i}. {issue.description}")
                report.append(f"   Line: {issue.line_number}")
                report.append(f"   Category: {issue.category}")
                report.append(f"   Recommendation: {issue.recommendation}")

        return '\n'.join(report)


def validate_code(code: str, language: str, strict: bool = False) -> bool:
    """
    Convenience function to validate code security

    Args:
        code: Code content
        language: Programming language
        strict: Strict mode flag

    Returns:
        True if code is safe, False otherwise
    """
    validator = SecurityValidator(strict_mode=strict)
    is_safe, issues = validator.scan_file(code, language)

    if not is_safe:
        logger.warning(f"Security issues found:\n{validator.get_report()}")

    return is_safe


def main():
    """Example usage"""
    # Example vulnerable code
    vulnerable_code = '''
import mysql.connector

def get_user(username):
    # SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE username = '{username}'"
    db.execute(query)

    # Hardcoded credentials
    api_key = "sk-1234567890abcdefghijklmnop"
    password = "my_secret_password"

    return db.fetchone()
'''

    validator = SecurityValidator(strict_mode=False)
    is_safe, issues = validator.scan_file(vulnerable_code, "Python")

    print(f"Code is safe: {is_safe}")
    print(validator.get_report())


if __name__ == "__main__":
    main()
