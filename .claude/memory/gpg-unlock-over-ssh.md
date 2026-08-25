---
name: gpg-unlock-over-ssh
description: "How to unlock gpg-agent for pxy/pass when Saiful is on SSH (no desktop), and the SIGHUP cache-flush trap"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4694d38b-d7cd-4225-8a9e-236e24670a41
  modified: 2026-08-25T04:45:37.617Z
---

pxy resolves every secret via `pass show`, so a locked gpg-agent blocks the whole
router after a restart. When Saiful is on SSH, the configured pinentry-qt pops on
the desktop display (`:0`) where nobody can answer it, and it does NOT fall back
to the SSH tty even with `GPG_TTY` set.

**Why:** `~/.gnupg/gpg-agent.conf` pins `pinentry-program /usr/bin/pinentry-qt`
(chezmoi-written; a future `chezmoi apply` rewrites that line to pinentry-qt).

**How to apply (working procedure, verified 2026-08-25):**
1. Kill any stale `pinentry-qt` process (it blocks the agent's prompt queue).
2. `sed` the conf line to `/usr/bin/pinentry-curses`, then `gpgconf --reload gpg-agent`.
3. Saiful runs `env GPG_TTY=(tty) pass show AI/zai/main` in his terminal (fish syntax)
   and types the passphrase — curses prompt draws in the SSH tty.
4. Revert the conf line to pinentry-qt on disk **WITHOUT reloading** — the running
   agent keeps the curses setting + cache in memory; disk is correct for next boot.

**The trap:** `gpgconf --reload gpg-agent` sends SIGHUP, and **SIGHUP flushes the
passphrase cache**. Reloading right after the unlock silently re-locks everything
(made that mistake once). Cache ttl is 400 days ("until next boot").

Check cache state: `gpg-connect-agent 'keyinfo --list' /bye` — a `1` in the field
after the three dashes means cached. Related: [[pxy-project]].
