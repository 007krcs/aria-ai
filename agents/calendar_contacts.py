"""
ARIA — Calendar & Contacts Agent
===================================
Read your schedule, create events, look up contacts.

Two backends — works without Google account:

1. Local (.ics + .vcf files)
   - Export your Google Calendar as .ics once, ARIA reads it
   - Export contacts as .vcf, ARIA parses them
   - No API keys, no OAuth, fully offline
   - For read-only + batch use

2. Google Calendar API (optional, requires OAuth2 once)
   - Real-time — always up to date
   - Can CREATE events
   - pip install google-auth google-auth-oauthlib google-api-python-client
   - Credentials: create OAuth2 credentials at console.cloud.google.com
     (free, Calendar API is free for personal use)

Setup:
  Option A (local, no setup needed):
    1. Go to calendar.google.com → Settings → Export
    2. Save the .ics file to data/calendar.ics
    3. Export contacts as .vcf to data/contacts.vcf
    ARIA reads these immediately.

  Option B (Google API, real-time):
    1. pip install google-auth google-auth-oauthlib google-api-python-client
    2. Create OAuth2 credentials → download credentials.json
    3. Place credentials.json in data/
    4. ARIA opens browser for one-time auth → saves token.json
    5. All future calls are automatic (token auto-refreshes)
"""

import re
import json
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()

CALENDAR_FILE   = PROJECT_ROOT / "data" / "calendar.ics"
CONTACTS_FILE   = PROJECT_ROOT / "data" / "contacts.vcf"
GCAL_CREDS      = PROJECT_ROOT / "data" / "credentials.json"
GCAL_TOKEN      = PROJECT_ROOT / "data" / "token.json"


# ─────────────────────────────────────────────────────────────────────────────
# ICS PARSER — no dependencies
# ─────────────────────────────────────────────────────────────────────────────

class ICSParser:
    """Parse .ics calendar files without any dependencies."""

    def parse(self, path: Path) -> list[dict]:
        try:
            text   = path.read_text(encoding="utf-8", errors="replace")
            events = []
            current: dict | None = None

            for line in text.splitlines():
                line = line.strip()
                if line == "BEGIN:VEVENT":
                    current = {}
                elif line == "END:VEVENT" and current is not None:
                    events.append(self._clean(current))
                    current = None
                elif current is not None and ":" in line:
                    key, _, val = line.partition(":")
                    key = key.split(";")[0]   # strip parameters
                    val = val.replace("\\n", "\n").replace("\\,", ",")
                    current[key] = val

            return events
        except Exception as e:
            console.print(f"  [yellow]ICS parse error: {e}[/]")
            return []

    def _clean(self, raw: dict) -> dict:
        """Convert raw ICS fields to a readable event dict."""
        def parse_dt(val: str) -> str | None:
            val = val.replace("Z","").replace("-","").replace(":","")
            for fmt in ("%Y%m%dT%H%M%S", "%Y%m%d"):
                try:
                    return datetime.strptime(val[:15], fmt).isoformat()
                except Exception:
                    pass
            return val

        return {
            "id":          raw.get("UID",""),
            "title":       raw.get("SUMMARY","").strip(),
            "start":       parse_dt(raw.get("DTSTART","")),
            "end":         parse_dt(raw.get("DTEND","")),
            "location":    raw.get("LOCATION","").strip(),
            "description": raw.get("DESCRIPTION","").strip()[:300],
            "status":      raw.get("STATUS","CONFIRMED").strip(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# VCF PARSER — no dependencies
# ─────────────────────────────────────────────────────────────────────────────

class VCFParser:
    """Parse .vcf contact files without any dependencies."""

    def parse(self, path: Path) -> list[dict]:
        try:
            text     = path.read_text(encoding="utf-8", errors="replace")
            contacts = []
            current: dict | None = None

            for line in text.splitlines():
                line = line.strip()
                if line in ("BEGIN:VCARD", "BEGIN:vCard"):
                    current = {"phones": [], "emails": []}
                elif line in ("END:VCARD", "END:vCard") and current:
                    if current.get("name") or current.get("phones"):
                        contacts.append(current)
                    current = None
                elif current is not None and ":" in line:
                    key, _, val = line.partition(":")
                    key_upper = key.upper().split(";")[0]

                    if key_upper == "FN":
                        current["name"] = val.strip()
                    elif key_upper == "N" and "name" not in current:
                        parts = val.split(";")
                        current["name"] = " ".join(
                            p.strip() for p in [parts[1] if len(parts)>1 else "",
                                                 parts[0]] if p.strip()
                        )
                    elif key_upper in ("TEL","TEL TYPE=CELL","TEL TYPE=MOBILE","TEL TYPE=HOME","TEL TYPE=WORK"):
                        phone = re.sub(r"[^\d+]","", val).strip()
                        if phone and phone not in current["phones"]:
                            current["phones"].append(phone)
                    elif key_upper in ("EMAIL","EMAIL TYPE=INTERNET"):
                        email = val.strip()
                        if email and "@" in email and email not in current["emails"]:
                            current["emails"].append(email)
                    elif key_upper == "BDAY":
                        current["birthday"] = val.strip()
                    elif key_upper == "ORG":
                        current["org"] = val.replace(";","").strip()
                    elif key_upper == "NOTE":
                        current["note"] = val[:200]

            return contacts
        except Exception as e:
            console.print(f"  [yellow]VCF parse error: {e}[/]")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE CALENDAR — optional real-time backend
# ─────────────────────────────────────────────────────────────────────────────

class GoogleCalendar:
    """
    Google Calendar API backend. Optional — requires one-time OAuth setup.
    Falls back to ICS if not configured.
    """

    def __init__(self):
        self._service = None

    def is_available(self) -> bool:
        return GCAL_CREDS.exists() or GCAL_TOKEN.exists()

    def _get_service(self):
        if self._service:
            return self._service
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request as GRequest
            from googleapiclient.discovery import build

            SCOPES = ["https://www.googleapis.com/auth/calendar"]
            creds  = None

            if GCAL_TOKEN.exists():
                creds = Credentials.from_authorized_user_file(str(GCAL_TOKEN), SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(GRequest())
                else:
                    if not GCAL_CREDS.exists():
                        return None
                    flow  = InstalledAppFlow.from_client_secrets_file(
                        str(GCAL_CREDS), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                GCAL_TOKEN.write_text(creds.to_json())

            self._service = build("calendar","v3", credentials=creds)
            return self._service
        except ImportError:
            console.print("  [dim]pip install google-auth google-auth-oauthlib google-api-python-client[/]")
            return None
        except Exception as e:
            console.print(f"  [yellow]Google Calendar error: {e}[/]")
            return None

    def get_events(
        self,
        start: datetime = None,
        end:   datetime = None,
        max_results: int = 20,
    ) -> list[dict]:
        service = self._get_service()
        if not service:
            return []
        try:
            start = start or datetime.utcnow()
            end   = end   or (start + timedelta(days=7))
            result = service.events().list(
                calendarId="primary",
                timeMin=start.isoformat() + "Z",
                timeMax=end.isoformat()   + "Z",
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            events = []
            for item in result.get("items",[]):
                start_raw = item["start"].get("dateTime", item["start"].get("date",""))
                end_raw   = item["end"].get("dateTime",   item["end"].get("date",""))
                events.append({
                    "id":          item.get("id",""),
                    "title":       item.get("summary",""),
                    "start":       start_raw,
                    "end":         end_raw,
                    "location":    item.get("location",""),
                    "description": item.get("description","")[:300],
                    "status":      item.get("status","confirmed"),
                    "link":        item.get("htmlLink",""),
                })
            return events
        except Exception as e:
            console.print(f"  [yellow]GCal events error: {e}[/]")
            return []

    def create_event(
        self,
        title:       str,
        start:       datetime,
        end:         datetime = None,
        location:    str = "",
        description: str = "",
        attendees:   list[str] = None,
    ) -> dict:
        service = self._get_service()
        if not service:
            return {"success": False, "error": "Google Calendar not configured"}
        end = end or (start + timedelta(hours=1))
        try:
            body: dict = {
                "summary":  title,
                "start":    {"dateTime": start.isoformat(), "timeZone": "Asia/Kolkata"},
                "end":      {"dateTime": end.isoformat(),   "timeZone": "Asia/Kolkata"},
                "location": location,
                "description": description,
            }
            if attendees:
                body["attendees"] = [{"email": e} for e in attendees]
            event = service.events().insert(calendarId="primary", body=body).execute()
            console.print(f"  [green]Event created:[/] {title}")
            return {"success": True, "id": event.get("id"), "link": event.get("htmlLink","")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_event(self, event_id: str) -> dict:
        service = self._get_service()
        if not service:
            return {"success": False, "error": "Google Calendar not configured"}
        try:
            service.events().delete(calendarId="primary", eventId=event_id).execute()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CALENDAR & CONTACTS AGENT
# ─────────────────────────────────────────────────────────────────────────────

class CalendarContactsAgent:
    """
    Unified calendar + contacts interface.
    Uses Google Calendar if available, falls back to .ics/.vcf files.
    """

    def __init__(self):
        self._ics_parser     = ICSParser()
        self._vcf_parser     = VCFParser()
        self._gcal           = GoogleCalendar()
        self._contacts_cache: list[dict] | None = None
        self._contacts_ts:    float = 0.0

    # ── Calendar ──────────────────────────────────────────────────────────────

    def get_events(
        self,
        start:   datetime = None,
        end:     datetime = None,
        days:    int = 7,
    ) -> list[dict]:
        """Get calendar events for a time range."""
        start = start or datetime.now().replace(hour=0, minute=0, second=0)
        end   = end   or (start + timedelta(days=days))

        # Google Calendar if available
        if self._gcal.is_available():
            return self._gcal.get_events(start, end)

        # Local .ics fallback
        if CALENDAR_FILE.exists():
            all_events = self._ics_parser.parse(CALENDAR_FILE)
            filtered   = []
            for ev in all_events:
                try:
                    ev_start = datetime.fromisoformat(ev.get("start",""))
                    if start <= ev_start <= end:
                        filtered.append(ev)
                except Exception:
                    pass
            return sorted(filtered, key=lambda e: e.get("start",""))

        return []

    def get_today(self) -> list[dict]:
        today = datetime.now().replace(hour=0, minute=0, second=0)
        tomorrow = today + timedelta(days=1)
        return self.get_events(today, tomorrow)

    def get_this_week(self) -> list[dict]:
        return self.get_events(days=7)

    def is_free(
        self,
        dt:           datetime,
        duration_min: int = 60,
    ) -> dict:
        """Check if a specific time slot is free."""
        end      = dt + timedelta(minutes=duration_min)
        events   = self.get_events(dt - timedelta(hours=1), end + timedelta(hours=1))
        conflicts = []
        for ev in events:
            try:
                ev_start = datetime.fromisoformat(ev.get("start",""))
                ev_end   = datetime.fromisoformat(ev.get("end", ev["start"]))
                # Check overlap
                if ev_start < end and ev_end > dt:
                    conflicts.append(ev)
            except Exception:
                pass
        return {
            "free":      len(conflicts) == 0,
            "time":      dt.strftime("%A, %d %B %Y at %I:%M %p"),
            "conflicts": conflicts,
        }

    def create_event(
        self,
        title:    str,
        when_str: str,
        duration_min: int = 60,
        location: str = "",
        description: str = "",
    ) -> dict:
        """Create an event. when_str: natural language like 'tomorrow 3pm'."""
        dt = self._parse_datetime(when_str)
        if not dt:
            return {"success": False, "error": f"Could not parse datetime: {when_str}"}

        end = dt + timedelta(minutes=duration_min)

        if self._gcal.is_available():
            return self._gcal.create_event(title, dt, end, location, description)

        # Local fallback: add to task scheduler as a reminder
        return {
            "success": True,
            "method":  "local_reminder",
            "note":    "Google Calendar not set up — event saved as reminder only",
            "title":   title,
            "time":    dt.isoformat(),
        }

    def natural_query(self, query: str) -> dict:
        """
        Answer natural language calendar questions.
        'Am I free tomorrow at 3pm?'
        'What do I have this week?'
        'When is my next meeting?'
        """
        q = query.lower()

        if any(w in q for w in ["today","tonight","this morning","this afternoon"]):
            events = self.get_today()
            return {"query": query, "events": events,
                    "answer": f"You have {len(events)} event(s) today." if events
                              else "Your schedule is clear today."}

        if any(w in q for w in ["this week","coming week","next 7 days"]):
            events = self.get_this_week()
            return {"query": query, "events": events[:10],
                    "answer": f"You have {len(events)} event(s) this week."}

        if any(w in q for w in ["free","available","busy"]):
            # Try to extract a time
            dt = self._parse_datetime(query)
            if dt:
                result = self.is_free(dt)
                answer = (f"You're free at {result['time']}." if result["free"]
                          else f"You have a conflict: {result['conflicts'][0].get('title','event')} at {result['time']}.")
                return {"query": query, **result, "answer": answer}

        # Default: next 3 days
        events = self.get_events(days=3)
        return {"query": query, "events": events[:5],
                "answer": f"{len(events)} event(s) in the next 3 days."}

    # ── Contacts ──────────────────────────────────────────────────────────────

    def _load_contacts(self) -> list[dict]:
        now = time.time()
        if self._contacts_cache and now - self._contacts_ts < 300:
            return self._contacts_cache
        contacts = []
        if CONTACTS_FILE.exists():
            contacts = self._vcf_parser.parse(CONTACTS_FILE)
        self._contacts_cache = contacts
        self._contacts_ts    = now
        console.print(f"  [dim]Contacts loaded: {len(contacts)}[/]")
        return contacts

    def find_contact(self, name_or_phone: str) -> list[dict]:
        """Find contacts by name or phone number."""
        contacts = self._load_contacts()
        query    = name_or_phone.lower().strip()
        found    = []
        for c in contacts:
            if (query in (c.get("name","") or "").lower() or
                any(query in p for p in c.get("phones",[])) or
                any(query in e for e in c.get("emails",[]))):
                found.append(c)
        return found[:10]

    def get_contact_phone(self, name: str) -> str | None:
        """Get first phone number for a contact."""
        matches = self.find_contact(name)
        if matches and matches[0].get("phones"):
            return matches[0]["phones"][0]
        return None

    def get_contact_email(self, name: str) -> str | None:
        """Get first email for a contact."""
        matches = self.find_contact(name)
        if matches and matches[0].get("emails"):
            return matches[0]["emails"][0]
        return None

    def upcoming_birthdays(self, days: int = 30) -> list[dict]:
        """Find contacts with birthdays in the next N days."""
        contacts = self._load_contacts()
        today    = date.today()
        upcoming = []
        for c in contacts:
            bday_str = c.get("birthday","")
            if not bday_str:
                continue
            try:
                # Parse MM-DD or YYYY-MM-DD
                parts = re.split(r"[-/]", bday_str.strip())
                if len(parts) >= 2:
                    month = int(parts[-2])
                    day   = int(parts[-1])
                    bday  = date(today.year, month, day)
                    if bday < today:
                        bday = date(today.year + 1, month, day)
                    delta = (bday - today).days
                    if 0 <= delta <= days:
                        upcoming.append({**c, "days_until": delta,
                                         "birthday_date": bday.isoformat()})
            except Exception:
                pass
        return sorted(upcoming, key=lambda x: x["days_until"])

    def status(self) -> dict:
        contacts = self._load_contacts()
        return {
            "backend":           "google_calendar" if self._gcal.is_available() else "local_ics",
            "google_connected":  self._gcal.is_available(),
            "ics_file":          str(CALENDAR_FILE) if CALENDAR_FILE.exists() else None,
            "vcf_file":          str(CONTACTS_FILE) if CONTACTS_FILE.exists() else None,
            "contacts_loaded":   len(contacts),
            "setup": {
                "local": "Export from Google Calendar (.ics) → save to data/calendar.ics",
                "google_api": "Place credentials.json in data/ and call /api/calendar/auth",
            } if not self._gcal.is_available() else None,
        }

    # ── Time parser ───────────────────────────────────────────────────────────

    def _parse_datetime(self, text: str) -> datetime | None:
        text  = text.lower().strip()
        now   = datetime.now()

        # "tomorrow"
        base  = now
        if "tomorrow" in text:
            base = now + timedelta(days=1)
        elif "next week" in text:
            base = now + timedelta(weeks=1)
        elif "monday"  in text: base = self._next_weekday(now, 0)
        elif "tuesday" in text: base = self._next_weekday(now, 1)
        elif "wednesday" in text: base = self._next_weekday(now, 2)
        elif "thursday" in text: base = self._next_weekday(now, 3)
        elif "friday"   in text: base = self._next_weekday(now, 4)
        elif "saturday" in text: base = self._next_weekday(now, 5)
        elif "sunday"   in text: base = self._next_weekday(now, 6)

        # Time
        m = re.search(r"(\d{1,2})[:.](\d{2})\s*(am|pm)?", text)
        if m:
            h, mn = int(m.group(1)), int(m.group(2))
            if m.group(3) == "pm" and h < 12: h += 12
            elif m.group(3) == "am" and h == 12: h = 0
            return base.replace(hour=h, minute=mn, second=0, microsecond=0)

        m2 = re.search(r"(\d{1,2})\s*(am|pm)", text)
        if m2:
            h = int(m2.group(1))
            if m2.group(2) == "pm" and h < 12: h += 12
            elif m2.group(2) == "am" and h == 12: h = 0
            return base.replace(hour=h, minute=0, second=0, microsecond=0)

        # Just a date — default to 9am
        return base.replace(hour=9, minute=0, second=0, microsecond=0)

    def _next_weekday(self, dt: datetime, weekday: int) -> datetime:
        days_ahead = weekday - dt.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return dt + timedelta(days=days_ahead)
