"""
ARIA — Backup & Restore
========================
Protects months of learned knowledge from disk failure.

What gets backed up (data/ folder contents):
  - chroma_db/          ChromaDB vector embeddings (your knowledge base)
  - world_model.db      Living knowledge graph
  - conversations.db    Full conversation history
  - tasks.db            Alarms, reminders, price alerts
  - system_events.db    Behaviour analytics
  - training/           Fine-tuning dataset + examples
  - adapters/           Trained model adapters
  - user_profile.json   Behaviour profile
  - user_interests.json Proactive monitoring interests
  - code_patterns.json  Learned code patterns
  - fix_rules.json      Self-correction rules
  - auth.json           PIN hash (not the PIN itself)
  - devices.json        Paired device tokens

What does NOT get backed up:
  - .jwt_secret         Regenerated on restore (forces re-login, by design)
  - *.pyc               Compiled Python bytecode
  - __pycache__/        

Backup destinations:
  1. Local path (default: data/backups/)
  2. Custom path — set BACKUP_PATH in .env
  3. Network drive — any path your OS can write to

Schedule: nightly at 2am, keeps last 14 backups (2 weeks).
Manual:   POST /api/backup/run
Restore:  POST /api/backup/restore {"backup_id": "..."}
"""

import os
import json
import shutil
import time
import zipfile
import threading
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()

# Files/dirs to back up
BACKUP_INCLUDE = [
    "data/chroma_db",
    "data/world_model.db",
    "data/conversations.db",
    "data/tasks.db",
    "data/system_events.db",
    "data/training",
    "data/adapters",
    "data/user_profile.json",
    "data/user_interests.json",
    "data/code_patterns.json",
    "data/fix_rules.json",
    "data/auth.json",
    "data/devices.json",
    "data/plans.json",
    "data/proactive_alerts.json",
    "logs/self_improvement.jsonl",
]

# Never back up these
BACKUP_EXCLUDE = {
    ".jwt_secret", "__pycache__", ".pyc", ".pyo",
    "aria_complete.zip", "node_modules",
}

BACKUP_DIR = PROJECT_ROOT / "data" / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


class BackupManager:
    """Handles creating, listing, and restoring ARIA backups."""

    def __init__(self):
        self._running   = False
        self._last_run: str | None = None
        self._load_state()

    def _state_path(self) -> Path:
        return BACKUP_DIR / "backup_state.json"

    def _load_state(self):
        try:
            if self._state_path().exists():
                data = json.loads(self._state_path().read_text())
                self._last_run = data.get("last_run")
        except Exception:
            pass

    def _save_state(self):
        try:
            self._state_path().write_text(json.dumps({
                "last_run": self._last_run,
            }, indent=2))
        except Exception:
            pass

    # ── Create backup ─────────────────────────────────────────────────────────

    def create_backup(self, label: str = "auto") -> dict:
        """
        Create a timestamped zip backup of all critical data.
        Returns info about the backup including path and size.
        """
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aria_backup_{label}_{ts}.zip"
        out_path = BACKUP_DIR / filename

        console.print(f"  [dim]Creating backup: {filename}[/]")
        included = 0
        skipped  = 0
        total_bytes = 0

        try:
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as zf:
                for pattern in BACKUP_INCLUDE:
                    src = PROJECT_ROOT / pattern
                    if not src.exists():
                        continue

                    if src.is_file():
                        if self._should_include(src):
                            zf.write(src, pattern)
                            included    += 1
                            total_bytes += src.stat().st_size
                    elif src.is_dir():
                        for root, dirs, files in os.walk(src):
                            # Skip excluded dirs
                            dirs[:] = [d for d in dirs
                                       if not any(ex in d for ex in BACKUP_EXCLUDE)]
                            for file in files:
                                fpath = Path(root) / file
                                if self._should_include(fpath):
                                    arcname = str(fpath.relative_to(PROJECT_ROOT))
                                    try:
                                        zf.write(fpath, arcname)
                                        included    += 1
                                        total_bytes += fpath.stat().st_size
                                    except Exception:
                                        skipped += 1
                                else:
                                    skipped += 1

            size_mb = out_path.stat().st_size / 1024 / 1024
            self._last_run = datetime.now().isoformat()
            self._save_state()

            console.print(
                f"  [green]Backup complete:[/] {filename} "
                f"({size_mb:.1f}MB, {included} files)"
            )

            # Prune old backups (keep last 14)
            self._prune(keep=14)

            return {
                "success":      True,
                "backup_id":    filename,
                "path":         str(out_path),
                "size_mb":      round(size_mb, 2),
                "files":        included,
                "skipped":      skipped,
                "ts":           self._last_run,
            }

        except Exception as e:
            if out_path.exists():
                out_path.unlink()
            console.print(f"  [red]Backup failed: {e}[/]")
            return {"success": False, "error": str(e)}

    def _should_include(self, path: Path) -> bool:
        name = path.name
        for ex in BACKUP_EXCLUDE:
            if ex in name:
                return False
        return True

    # ── List backups ──────────────────────────────────────────────────────────

    def list_backups(self) -> list[dict]:
        """List all available backups, newest first."""
        backups = []
        for f in sorted(BACKUP_DIR.glob("aria_backup_*.zip"), reverse=True):
            try:
                stat = f.stat()
                # Count files in zip
                with zipfile.ZipFile(f) as zf:
                    n_files = len(zf.namelist())
                backups.append({
                    "backup_id": f.name,
                    "path":      str(f),
                    "size_mb":   round(stat.st_size / 1024 / 1024, 2),
                    "files":     n_files,
                    "created":   datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            except Exception:
                pass
        return backups

    # ── Restore ───────────────────────────────────────────────────────────────

    def restore(self, backup_id: str, confirm: bool = False) -> dict:
        """
        Restore from a backup. Overwrites current data.
        This is destructive — current data is moved to data/backups/pre_restore/
        before restoring, so you can undo.
        """
        backup_path = BACKUP_DIR / backup_id
        if not backup_path.exists():
            return {"success": False, "error": f"Backup not found: {backup_id}"}

        if not confirm:
            return {
                "success":      False,
                "needs_confirm": True,
                "message":      (
                    f"This will overwrite current data with backup from "
                    f"{backup_id}. Current data will be saved to "
                    f"data/backups/pre_restore/ before overwriting. "
                    "Call again with confirm=true to proceed."
                ),
            }

        console.print(f"  [dim]Restoring from: {backup_id}[/]")

        # Safety: back up current state first
        pre_restore = self.create_backup("pre_restore")
        if not pre_restore.get("success"):
            return {"success": False, "error": "Could not create pre-restore backup"}

        try:
            with zipfile.ZipFile(backup_path) as zf:
                for member in zf.namelist():
                    dest = PROJECT_ROOT / member
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)

            console.print(f"  [green]Restore complete:[/] {backup_id}")
            return {
                "success":       True,
                "restored_from": backup_id,
                "pre_restore":   pre_restore["backup_id"],
                "note":          "Restart ARIA server to apply restored data.",
            }
        except Exception as e:
            console.print(f"  [red]Restore failed: {e}[/]")
            return {"success": False, "error": str(e)}

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune(self, keep: int = 14):
        """Delete old backups, keeping the N most recent."""
        backups = sorted(
            BACKUP_DIR.glob("aria_backup_auto_*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in backups[keep:]:
            try:
                old.unlink()
                console.print(f"  [dim]Pruned old backup: {old.name}[/]")
            except Exception:
                pass

    # ── Scheduler ─────────────────────────────────────────────────────────────

    def start_scheduler(self):
        """Run nightly backup at 2am in a background thread."""
        import time

        def loop():
            while True:
                now = datetime.now()
                # Next 2am
                next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                wait_s = (next_run - now).total_seconds()
                console.print(
                    f"  [dim]Backup scheduler: next run in {wait_s/3600:.1f}h[/]"
                )
                time.sleep(wait_s)
                self.create_backup("auto")

        self._running = True
        threading.Thread(target=loop, daemon=True, name="backup-scheduler").start()
        console.print("  [green]Backup scheduler:[/] nightly at 2am")

    def status(self) -> dict:
        backups = self.list_backups()
        total_mb = sum(b["size_mb"] for b in backups)
        return {
            "last_run":       self._last_run,
            "backup_count":   len(backups),
            "total_size_mb":  round(total_mb, 2),
            "backup_dir":     str(BACKUP_DIR),
            "latest_backup":  backups[0] if backups else None,
            "schedule":       "nightly at 2am",
        }


# Global instance
backup_manager = BackupManager()
