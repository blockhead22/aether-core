"""Mutual-exclusion detection for canonical class-valued facts.

The structural tension meter is fact-vs-fact via slot extraction. It
catches "I live in Seattle" vs "I live in Portland" because the
location slot extractor produces overlapping slots. It misses cases
where two facts are about the same domain but no slot extractor
recognizes the domain — e.g. "we deploy to AWS" vs "we deploy to GCP",
or "the team uses Postgres" vs "we run on MySQL".

This module ships a small registry of canonical mutually-exclusive
value classes (cloud providers, package managers, databases, etc.)
plus a `detect_mutex_conflict` function that checks whether two texts
mention different values from the same class. It's a precision tool —
high false-negative rate, near-zero false-positive rate. Adding a new
class is a one-liner.

Usage:
    from aether.contradiction.mutex import detect_mutex_conflict

    hit = detect_mutex_conflict(
        "We deploy to AWS us-east-1",
        "We deploy to GCP us-central1",
    )
    if hit:
        print(hit.class_name, hit.value_a, hit.value_b)
        # cloud_provider AWS GCP
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MutexClass:
    """A class of mutually-exclusive canonical values.

    `aliases` maps a canonical value to a list of strings that resolve
    to it. Matching is case-insensitive whole-word.
    """
    name: str
    aliases: Dict[str, List[str]]
    # Optional cue terms that have to appear in BOTH texts for the
    # detector to fire. Without this, "the AWS docs are great" and
    # "the GCP outage was bad" would be flagged. With cue=["deploy",
    # "host", "run on", "use", "production"], we limit to deployment
    # context.
    require_shared_cue: List[str] = field(default_factory=list)


@dataclass
class MutexConflict:
    """Result of a mutex detection."""
    class_name: str
    value_a: str
    value_b: str
    cue_used: Optional[str] = None
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

DEFAULT_CLASSES: List[MutexClass] = [
    MutexClass(
        name="cloud_provider",
        aliases={
            "AWS": ["aws", "amazon web services", "ec2", "amazon cloud"],
            "GCP": ["gcp", "google cloud", "google cloud platform", "gce"],
            "Azure": ["azure", "microsoft azure"],
            "DigitalOcean": ["digitalocean", "digital ocean"],
            "Vercel": ["vercel"],
            "Netlify": ["netlify"],
            "Fly.io": ["fly.io", "fly io"],
            "Cloudflare": ["cloudflare workers", "cloudflare pages"],
            "Heroku": ["heroku"],
            "Render": ["render.com"],
        },
        require_shared_cue=[
            "deploy", "host", "run on", "running on", "production",
            "infra", "infrastructure", "cluster", "region", "stack",
            "platform", "we use",
        ],
    ),
    MutexClass(
        name="package_manager_js",
        aliases={
            "npm": ["npm"],
            "pnpm": ["pnpm"],
            "yarn": ["yarn"],
            "bun": ["bun install", "bun pm"],
        },
        require_shared_cue=[
            "package manager", "install", "lockfile", "node_modules",
            "we use", "team uses", "the project uses", "scripts",
        ],
    ),
    MutexClass(
        name="package_manager_py",
        aliases={
            "pip": ["pip install", "pip "],
            "poetry": ["poetry"],
            "uv": ["uv pip", "uv add", "uv "],
            "pdm": ["pdm "],
            "pipenv": ["pipenv"],
            "conda": ["conda install", "conda env"],
        },
        require_shared_cue=[
            "dependency", "dependencies", "install", "we use", "the project",
            "lockfile", "venv", "environment",
        ],
    ),
    MutexClass(
        name="database",
        aliases={
            "Postgres": ["postgres", "postgresql", "psql"],
            "MySQL": ["mysql", "mariadb"],
            "SQLite": ["sqlite"],
            "MongoDB": ["mongo", "mongodb"],
            "Redis": ["redis"],
            "DynamoDB": ["dynamodb", "dynamo"],
            "Cassandra": ["cassandra"],
            "Snowflake": ["snowflake"],
            "BigQuery": ["bigquery", "big query"],
        },
        require_shared_cue=[
            "database", "db", "primary store", "data store", "we use",
            "running on", "schema", "migration", "migrations",
        ],
    ),
    MutexClass(
        name="frontend_framework",
        aliases={
            "React": ["react", "reactjs"],
            "Vue": ["vue", "vue.js", "vuejs"],
            "Svelte": ["svelte", "sveltekit"],
            "Angular": ["angular"],
            "Solid": ["solidjs", "solid.js"],
            "Next.js": ["next.js", "nextjs", "next "],
            "Remix": ["remix"],
            "Astro": ["astro"],
        },
        require_shared_cue=[
            "frontend", "front end", "client", "ui", "we use", "the app",
            "spa", "framework",
        ],
    ),
    MutexClass(
        name="backend_runtime",
        aliases={
            "Node.js": ["node.js", "nodejs", "node "],
            "Deno": ["deno"],
            "Bun": ["bun runtime", "bun.sh", " bun "],
            "Python": ["python", "fastapi", "django", "flask"],
            "Go": [" go ", "golang"],
            "Rust": ["rust", "actix", "axum"],
            "Ruby": ["ruby", "rails"],
            "Java": [" java ", "spring boot"],
            "C#": ["c#", ".net"],
            "Elixir": ["elixir", "phoenix"],
        },
        require_shared_cue=[
            "backend", "back end", "server", "api", "we use",
            "the service", "the app", "runtime",
        ],
    ),
    MutexClass(
        name="vcs_host",
        aliases={
            "GitHub": ["github"],
            "GitLab": ["gitlab"],
            "Bitbucket": ["bitbucket"],
            "Gitea": ["gitea"],
        },
        require_shared_cue=[
            "repo", "repository", "host", "ci", "we use", "code lives",
            "actions", "pipelines",
        ],
    ),
    MutexClass(
        name="container_orchestrator",
        aliases={
            "Kubernetes": ["kubernetes", "k8s", "kube "],
            "Docker Swarm": ["docker swarm"],
            "Nomad": ["nomad"],
            "ECS": [" ecs "],
            "Fargate": ["fargate"],
        },
        require_shared_cue=[
            "orchestrator", "deploy", "container", "cluster",
            "production", "we use",
        ],
    ),
    MutexClass(
        name="auth_provider",
        aliases={
            "Auth0": ["auth0"],
            "Clerk": ["clerk"],
            "Supabase Auth": ["supabase auth"],
            "Firebase Auth": ["firebase auth"],
            "Cognito": ["cognito"],
            "Okta": ["okta"],
            "Keycloak": ["keycloak"],
            "WorkOS": ["workos"],
        },
        require_shared_cue=[
            "auth", "login", "sign in", "sso", "we use", "identity",
        ],
    ),
    MutexClass(
        name="payment_processor",
        aliases={
            "Stripe": ["stripe"],
            "Square": ["square payments", "square pos"],
            "Braintree": ["braintree"],
            "PayPal": ["paypal"],
            "Adyen": ["adyen"],
            "LemonSqueezy": ["lemonsqueezy", "lemon squeezy"],
        },
        require_shared_cue=[
            "payment", "checkout", "billing", "we use", "process",
            "subscription", "subscriptions",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _find_canonical_value(
    text: str,
    cls: MutexClass,
) -> Optional[str]:
    """Return the canonical value mentioned in `text` for this class, or None.

    If multiple are found, returns None (ambiguous — don't fire).
    """
    t = text.lower()
    matched: List[str] = []
    for canonical, aliases in cls.aliases.items():
        for alias in aliases:
            # Word-boundary match. We pad with spaces and match to handle
            # alias-with-trailing-space cases like " bun ".
            pattern = r"(?:^|\W)" + re.escape(alias.strip()) + r"(?:$|\W)"
            if re.search(pattern, t):
                matched.append(canonical)
                break
    matched = list(dict.fromkeys(matched))  # dedupe preserving order
    if len(matched) == 1:
        return matched[0]
    return None


def _shared_cue(text_a: str, text_b: str, cues: List[str]) -> Optional[str]:
    if not cues:
        return ""  # No cue requirement
    a = text_a.lower()
    b = text_b.lower()
    for cue in cues:
        if cue in a and cue in b:
            return cue
    return None


def detect_mutex_conflict(
    text_a: str,
    text_b: str,
    classes: Optional[List[MutexClass]] = None,
) -> Optional[MutexConflict]:
    """Return a MutexConflict if the two texts mention different canonical
    values from the same class (and share a context cue when required).

    Otherwise returns None.
    """
    classes = classes if classes is not None else DEFAULT_CLASSES

    for cls in classes:
        v_a = _find_canonical_value(text_a, cls)
        v_b = _find_canonical_value(text_b, cls)
        if v_a is None or v_b is None:
            continue
        if v_a == v_b:
            continue

        cue = _shared_cue(text_a, text_b, cls.require_shared_cue)
        if cue is None:
            # Required cue not shared — skip
            continue

        return MutexConflict(
            class_name=cls.name,
            value_a=v_a,
            value_b=v_b,
            cue_used=cue or None,
            confidence=0.9,
        )

    return None


def all_classes() -> List[MutexClass]:
    """Return a fresh copy of the default registry."""
    return list(DEFAULT_CLASSES)
