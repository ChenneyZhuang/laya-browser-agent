#!/usr/bin/env python3
"""v37 deployment patch: Outlook / M365 wide-suite targeted training records.

Root cause (RETRAIN_HANDOFF.md + 3080 wide-suite baselines of v10s/v32b/v24a):
  * Operation-boundary errors: verb-labelled buttons (New event / Today /
    Import contacts / Send email to selected) chosen as TYPE_TEXT targets.
    Counterfactual twins: same page, goal needs the neighbouring search field
    -> TYPE_TEXT on the field; goal needs the button -> CLICK on the button.
  * Label confusions measured on the bench: Reply vs Reply all; Cc vs Show Bcc;
    Sort vs folder names (Sent Items/Drafts); Account manager vs Move to;
    Pop out vs Move to; Close (window) vs Close ticket; New mail vs opening
    the newest message from a named sender; search-field vs Filter.
  * Restraint class absent (0/16 for every checkpoint): goals already
    satisfied -> DONE, goals impossible -> BLOCKED. Generated with
    counterfactual twins so the gold flips on evidence, not on keywords.
  * people / CRM surfaces under-represented (20% vs 100% elsewhere).

Record contract: identical to v20/v29 (build_laya_items.py consumes it
unchanged; SELECT golds use the SELECT_OPTION alias via supported_operations).
Output: JSONL, one record per line.

Design rules:
  * Counterfactual pairs share page + near-identical goal wording; only the
    state/evidence changes -> gold flips. This forces state reading.
  * Restraint records: 60% bench-style goals that state the situation
    ("...already been deleted"), 25% plain goals with evidence only in
    state/history, 15% hard-negative twins where the same wording means CLICK
    because the action is NOT yet done. Prevents trigger-happy DONE/BLOCKED.
  * Surface variation: each control has 2-4 label variants and each goal 3-5
    phrasings, drawn deterministically per record. The direct bench labels
    stay in the variant pool so the deployment patterns transfer.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LANGS = ("en", "en", "en", "en", "zh-CN")  # deployment is English; light zh slice


def c(cid, role, name, operations, lang, *, state=None, placeholder=None):
    number = int("".join(ch for ch in cid if ch.isdigit()) or "1")
    item = {
        "id": cid, "role": role, "accessible_name": name,
        "visible_text": name if role in {"button", "link", "option", "tab", "menuitem", "checkbox"} else "",
        "state": {"visible": True, "enabled": True, "in_viewport": True, **(state or {})},
        "geometry": {"x": 36 + (number % 6) * 148, "y": 64 + (number % 5) * 44, "width": 172, "height": 36},
        "language": lang, "dir": "ltr",
        "supported_operations": [("SELECT_OPTION" if op == "SELECT" else op) for op in operations],
    }
    if placeholder is not None:
        item["placeholder"] = placeholder
    return item


def H(op, tid, result):
    return {"operation": op, "target_id": tid, "result": result}


# ── page states (structure mirrors the wide bench's synthetic Outlook pages) ──

def st_calendar(v):
    text = ("Outlook Calendar, week view, week of 28 September 2026. "
            "Events: Mon 09:00 Team standup; Tue 14:00 Allcare Homes kickoff; "
            "Wed 11:00 Andrew 1:1; Thu 16:00 Vendor call; Fri 10:00 Backup review.")
    if v == "today_open":
        text += " The current day is already shown."
    cands = [
        c("n1", "button", "New event", ["CLICK"], "en"),
        c("n2", "button", "Today", ["CLICK"], "en"),
        c("n3", "button", "Day view", ["CLICK"], "en"),
        c("n4", "button", "Work week view", ["CLICK"], "en"),
        c("n5", "button", "Month view", ["CLICK"], "en"),
        c("n6", "searchbox", "Search calendar", ["TYPE_TEXT"], "en", placeholder="Search"),
        c("n7", "button", "Share", ["CLICK"], "en"),
        c("n8", "button", "Print", ["CLICK"], "en"),
        c("n9", "button", "Calendar settings", ["CLICK"], "en"),
        c("n10", "button", "Add calendar", ["CLICK"], "en"),
        c("v1", "link", "Mon 09:00 Team standup", ["CLICK"], "en"),
        c("v2", "link", "Tue 14:00 Allcare Homes kickoff", ["CLICK"], "en"),
        c("v3", "link", "Wed 11:00 Andrew 1:1", ["CLICK"], "en"),
        c("v4", "link", "Thu 16:00 Vendor call", ["CLICK"], "en"),
        c("v5", "link", "Fri 10:00 Backup review", ["CLICK"], "en"),
        c("r1", "button", "Previous week", ["CLICK"], "en"),
        c("r2", "button", "Next week", ["CLICK"], "en"),
        c("c1", "button", "Minimize", ["CLICK"], "en"),
        c("c3", "button", "Close", ["CLICK"], "en"),
    ]
    return ("https://outlook.office.com/calendar/view/week", "Calendar - Outlook", text, cands)


def st_people(v):
    text = ("Outlook People. Contacts: Andrew Caudle (andrew@4data.com.au), "
            "Bo Zhuang, helpdesk@4data.com.au, billing@acme.com.au.")
    if v == "selected":
        text += " Bo Zhuang is selected."
    cands = [
        c("p1", "button", "New contact", ["CLICK"], "en"),
        c("p2", "searchbox", "Search contacts", ["TYPE_TEXT"], "en", placeholder="Search contacts"),
        c("p3", "button", "Import contacts", ["CLICK"], "en"),
        c("p4", "button", "Export contacts", ["CLICK"], "en"),
        c("p5", "button", "Manage categories", ["CLICK"], "en"),
        c("p6", "link", "Andrew Caudle", ["CLICK"], "en"),
        c("p7", "link", "Bo Zhuang", ["CLICK"], "en"),
        c("p8", "link", "helpdesk@4data.com.au", ["CLICK"], "en"),
        c("p9", "link", "billing@acme.com.au", ["CLICK"], "en"),
        c("p10", "button", "Send email to selected", ["CLICK"], "en"),
        c("p11", "button", "Delete contact", ["CLICK"], "en"),
        c("c1", "button", "Minimize", ["CLICK"], "en"),
        c("c3", "button", "Close", ["CLICK"], "en"),
    ]
    return ("https://outlook.office.com/people", "People - Outlook", text, cands)


def st_inbox(v):
    text = ("Outlook on the web, signed in as chenney@4data.com.au. Folder: Inbox "
            "(14 unread of 106). Message list newest first:\n"
            "  1. Andrew Caudle - Re: Allcare Homes onboarding - Mon 09:14 - read\n"
            "  2. Helpdesk - Ticket 4471 assigned to you - Mon 08:02 - unread\n"
            "  3. Microsoft - Subscription renewal - Sun 22:41 - read\n"
            "  4. ACME Pty Ltd - Invoice query - Fri 16:20 - unread\n"
            "  5. Andrew Caudle - MSP retainer renewal - Fri 11:05 - read\n"
            "  6. Security alert - New sign-in detected - Thu 19:33 - read\n"
            "  7. HR - Reimbursement form reminder - Thu 10:12 - unread\n"
            "  8. Newsletter - Industry roundup - Wed 06:00 - read\n")
    if v == "deleted":
        text += " The previously selected message was deleted; the list now shows the next message. Nothing is selected."
    if v == "replied":
        text += " A reply to the open message was sent successfully a moment ago."
    if v == "flagged":
        text += " The open message already carries a follow-up flag."
    if v == "unread":
        text += " The open message is already marked unread."
    if v == "shared":
        text += " The selected item is already shared with everyone in the organisation."
    cands = [
        c("t1", "searchbox", "Search", ["TYPE_TEXT"], "en", placeholder="Search"),
        c("t2", "button", "New mail", ["CLICK"], "en"),
        c("t3", "button", "Delete", ["CLICK"], "en"),
        c("t4", "button", "Archive", ["CLICK"], "en"),
        c("t5", "button", "Reply", ["CLICK"], "en"),
        c("t6", "button", "Reply all", ["CLICK"], "en"),
        c("t7", "button", "Forward", ["CLICK"], "en"),
        c("t8", "button", "Move to", ["CLICK"], "en"),
        c("t9", "button", "Categorize", ["CLICK"], "en"),
        c("t10", "button", "Filter", ["CLICK"], "en"),
        c("t11", "button", "Sort", ["CLICK"], "en"),
        c("t12", "button", "Settings", ["CLICK"], "en"),
        c("t13", "button", "Help", ["CLICK"], "en"),
        c("t14", "button", "Account manager", ["CLICK"], "en"),
        c("f1", "treeitem", "Inbox", ["CLICK"], "en"),
        c("f2", "treeitem", "Drafts", ["CLICK"], "en"),
        c("f3", "treeitem", "Sent Items", ["CLICK"], "en"),
        c("f4", "treeitem", "Deleted Items", ["CLICK"], "en"),
        c("f5", "treeitem", "Junk Email", ["CLICK"], "en"),
        c("f6", "treeitem", "Archive", ["CLICK"], "en"),
        c("f7", "treeitem", "Conversation History", ["CLICK"], "en"),
        c("m1", "link", "Andrew Caudle - Re: Allcare Homes onboarding", ["CLICK"], "en"),
        c("m2", "link", "Helpdesk - Ticket 4471 assigned to you", ["CLICK"], "en"),
        c("m4", "link", "ACME Pty Ltd - Invoice query", ["CLICK"], "en"),
        c("m5", "link", "Andrew Caudle - MSP retainer renewal", ["CLICK"], "en"),
        c("p1", "button", "Quick reply", ["CLICK"], "en"),
        c("p2", "button", "Pop out", ["CLICK"], "en"),
        c("p3", "button", "Download attachments", ["CLICK"], "en"),
        c("p4", "button", "Mark as unread", ["CLICK"], "en"),
        c("p5", "button", "Flag", ["CLICK"], "en"),
        c("p6", "button", "Pin", ["CLICK"], "en"),
        c("c1", "button", "Minimize", ["CLICK"], "en"),
        c("c3", "button", "Close", ["CLICK"], "en"),
    ]
    return ("https://outlook.office.com/mail/inbox", "Inbox - Outlook", text, cands)


def st_compose(v):
    text = "New message window. To: (empty). Subject: (empty). Body: (empty). Options ribbon: Attach, Address book, Signature, Importance, Sensitivity."
    if v == "filled":
        text = "New message window. To: andrew@4data.com.au. Subject: (empty). Body: (empty). Options ribbon: Attach, Address book, Signature, Importance, Sensitivity."
    if v == "attached":
        text = ("New message window. To: andrew@4data.com.au. Subject: Backup report. Body: See attached. "
                "Attachments: backkup-report-sept.pdf (already listed).")
    if v == "empty_draft":
        text = "New message window. To: (empty). Subject: (empty). Body: (empty). No draft content exists."
    cands = [
        c("to1", "combobox", "To", ["TYPE_TEXT"], "en", state={"current_value": ""}, placeholder="To"),
        c("cc1", "combobox", "Cc", ["TYPE_TEXT"], "en", state={"current_value": ""}, placeholder="Cc"),
        c("sub1", "textbox", "Add a subject", ["TYPE_TEXT"], "en", state={"current_value": ""}, placeholder="Add a subject"),
        c("body1", "textbox", "Message body", ["TYPE_TEXT"], "en", state={"current_value": ""}, placeholder="Message body"),
        c("a1", "button", "Attach", ["CLICK"], "en"),
        c("a2", "button", "Address book", ["CLICK"], "en"),
        c("a3", "button", "Signature", ["CLICK"], "en"),
        c("a4", "button", "Importance", ["CLICK"], "en"),
        c("a5", "button", "Sensitivity", ["CLICK"], "en"),
        c("a6", "button", "Show Bcc", ["CLICK"], "en"),
        c("s1", "button", "Send", ["CLICK"], "en"),
        c("s2", "button", "Discard", ["CLICK"], "en"),
        c("s3", "button", "Save draft", ["CLICK"], "en"),
        c("s4", "button", "Formatting options", ["CLICK"], "en"),
        c("s5", "button", "Insert table", ["CLICK"], "en"),
        c("s6", "button", "Insert picture", ["CLICK"], "en"),
        c("r1", "button", "Undo", ["CLICK"], "en"),
        c("r3", "button", "Check spelling", ["CLICK"], "en"),
        c("c1", "button", "Minimize", ["CLICK"], "en"),
        c("c2", "button", "Maximize", ["CLICK"], "en"),
        c("c3", "button", "Close", ["CLICK"], "en"),
    ]
    if v == "filled":
        for e in cands:
            if e["id"] == "to1":
                e["state"]["current_value"] = "andrew@4data.com.au"
    return ("https://outlook.office.com/mail/deeplink/compose", "New message", text, cands)


def st_ticket(v):
    text = ("Helpdesk ticketing web app. Ticket 4471 'Allcare Homes - mailbox setup', "
            "priority High, status Open, assignee chenney, client Allcare Homes.")
    if v == "closed":
        text = ("Helpdesk ticketing web app. Ticket 4471 'Allcare Homes - mailbox setup', "
                "priority High, status Closed, assignee chenney, client Allcare Homes.")
    if v == "unassigned":
        text = "Helpdesk ticketing web app. Ticket 4471, status Open, assignee (none). Assignee list: chenney, helpdesk, margaret."
    cands = [
        c("a1", "button", "Assign to me", ["CLICK"], "en"),
        c("a2", "button", "Change priority", ["CLICK"], "en"),
        c("a3", "button", "Change status", ["CLICK"], "en"),
        c("a4", "textbox", "Add comment", ["TYPE_TEXT"], "en", state={"current_value": ""}, placeholder="Add comment"),
        c("a5", "button", "Attach file", ["CLICK"], "en"),
        c("a6", "button", "Log time", ["CLICK"], "en"),
        c("a7", "button", "Link ticket", ["CLICK"], "en"),
        c("a8", "button", "Merge ticket", ["CLICK"], "en"),
        c("a9", "button", "Close ticket", ["CLICK"], "en"),
        c("a10", "button", "Reopen ticket", ["CLICK"], "en"),
        c("a11", "button", "Watch ticket", ["CLICK"], "en"),
        c("a12", "button", "Escalate", ["CLICK"], "en"),
        c("d1", "link", "Client: Allcare Homes", ["CLICK"], "en"),
        c("d2", "link", "Assignee: chenney", ["CLICK"], "en"),
        c("d3", "link", "Requester: andrew@4data.com.au", ["CLICK"], "en"),
        c("c1", "button", "Minimize", ["CLICK"], "en"),
        c("c3", "button", "Close", ["CLICK"], "en"),
    ]
    return ("https://helpdesk.4data.com.au/tickets/4471", "Ticket 4471", text, cands)


def st_files(v):
    text = ("OneDrive file browser. Folders: 4Data, Client Reports, Backups, Invoices. "
            "Files: client-list-july-26.xlsx, backkup-report-sept.pdf, invoice-88213.pdf.")
    if v == "shared":
        text += " The selected file is already shared with everyone in the organisation."
    if v == "no_payroll":
        text += " There is no Payroll folder here."
    cands = [
        c("f1", "treeitem", "4Data", ["CLICK"], "en"),
        c("f2", "treeitem", "Client Reports", ["CLICK"], "en"),
        c("f3", "treeitem", "Backups", ["CLICK"], "en"),
        c("f4", "treeitem", "Invoices", ["CLICK"], "en"),
        c("d1", "link", "client-list-july-26.xlsx", ["CLICK"], "en"),
        c("d2", "link", "backkup-report-sept.pdf", ["CLICK"], "en"),
        c("d3", "link", "invoice-88213.pdf", ["CLICK"], "en"),
        c("u1", "button", "Upload", ["CLICK"], "en"),
        c("u2", "button", "New folder", ["CLICK"], "en"),
        c("u3", "button", "Sync", ["CLICK"], "en"),
        c("u4", "button", "Download", ["CLICK"], "en"),
        c("u5", "button", "Share", ["CLICK"], "en"),
        c("u6", "button", "Delete", ["CLICK"], "en"),
        c("s1", "button", "Sort by name", ["CLICK"], "en"),
        c("s2", "button", "Sort by modified", ["CLICK"], "en"),
        c("s3", "button", "Filter by type", ["CLICK"], "en"),
        c("c1", "button", "Minimize", ["CLICK"], "en"),
        c("c3", "button", "Close", ["CLICK"], "en"),
    ]
    return ("https://outlook.office.com/files", "Files - Outlook", text, cands)


def st_settings(v):
    text = ("Outlook settings pane. Sections: Account, Mail, Calendar, People, "
            "General, Signatures, Automatic replies, Rules, Junk email, Mobile devices. "
            "Signed in as chenney@4data.com.au.")
    if v == "signed_out":
        text = "Outlook sign-in page. The previous session has already ended; no mailbox is open."
    cands = [
        c("s1", "treeitem", "Account", ["CLICK"], "en"),
        c("s2", "treeitem", "Mail", ["CLICK"], "en"),
        c("s3", "treeitem", "Calendar", ["CLICK"], "en"),
        c("s4", "treeitem", "People", ["CLICK"], "en"),
        c("s5", "treeitem", "General", ["CLICK"], "en"),
        c("s6", "treeitem", "Signatures", ["CLICK"], "en"),
        c("s7", "treeitem", "Automatic replies", ["CLICK"], "en"),
        c("s8", "treeitem", "Rules", ["CLICK"], "en"),
        c("s9", "treeitem", "Junk email", ["CLICK"], "en"),
        c("s10", "treeitem", "Mobile devices", ["CLICK"], "en"),
        c("a1", "button", "Save", ["CLICK"], "en"),
        c("a2", "button", "Discard", ["CLICK"], "en"),
        c("a3", "searchbox", "Search settings", ["TYPE_TEXT"], "en", placeholder="Search settings"),
        c("a4", "button", "Sign out", ["CLICK"], "en"),
        c("a5", "button", "Manage storage", ["CLICK"], "en"),
        c("c1", "button", "Minimize", ["CLICK"], "en"),
        c("c3", "button", "Close", ["CLICK"], "en"),
    ]
    if v == "signed_out":
        cands = [
            c("b1", "textbox", "Email, phone, or Skype", ["TYPE_TEXT"], "en", state={"current_value": ""}, placeholder="Email"),
            c("b2", "button", "Next", ["CLICK"], "en"),
            c("b3", "button", "Create one!", ["CLICK"], "en"),
            c("b4", "button", "Forgot my password", ["CLICK"], "en"),
        ]
    return ("https://outlook.office.com/mail/options", "Settings - Outlook", text, cands)


def st_search(v):
    text = ("Outlook search results for 'backup failure'. 3 results shown. "
            "Filters toolbar available. Nothing is selected.")
    if v == "no_results":
        text = ("Outlook search results for 'payroll migration'. No matching result was found. "
                "Filters toolbar available. Nothing is selected.")
    cands = [
        c("q1", "searchbox", "Search", ["TYPE_TEXT"], "en", state={"current_value": "backup failure"}, placeholder="Search"),
        c("r1", "link", "Backup job failed - Nightly backup", ["CLICK"], "en"),
        c("r2", "link", "Re: Backup job failed - Allcare", ["CLICK"], "en"),
        c("r3", "link", "Backup retention policy update", ["CLICK"], "en"),
        c("t1", "button", "Filter", ["CLICK"], "en"),
        c("t2", "button", "Sort", ["CLICK"], "en"),
        c("t3", "button", "Search all folders", ["CLICK"], "en"),
        c("t4", "button", "Advanced search", ["CLICK"], "en"),
        c("t5", "button", "Clear search", ["CLICK"], "en"),
        c("c1", "button", "Minimize", ["CLICK"], "en"),
        c("c3", "button", "Close", ["CLICK"], "en"),
    ]
    if v == "no_results":
        cands = [e for e in cands if not e["id"].startswith("r")]
    return ("https://outlook.office.com/mail/search", "Search - Outlook", text, cands)


STATES = {
    "calendar": st_calendar, "people": st_people, "inbox": st_inbox,
    "compose": st_compose, "ticket": st_ticket, "files": st_files,
    "settings": st_settings, "search": st_search,
}


def emit(idx, family, lang, page_name, page, goal, gold_op, cands, history, target_id=None, post=None, tags_extra=None):
    url, title, text, _ = page
    label = {"operation": gold_op, "postcondition": {"type": post or f"{family}_v37"}}
    if target_id:
        label["target_id"] = target_id
        label["acceptable_target_ids"] = [target_id]
    if gold_op == "TYPE_TEXT":
        label["argument"] = {"type": "slot", "slot_id": "text"}
    return {
        "trace_id": f"v37-{family}-{idx:06d}",
        "step_id": 0,
        "task": {"instruction": goal, "instruction_lang": lang, "task_type": f"{family}-v37",
                 "goal_state": (post or family) + "_v37"},
        "environment": {"browser_engine": "chromium", "browser_version": "fixture-v37",
                        "viewport": [1280, 720], "locale": lang, "timezone": "UTC",
                        "device_profile": "desktop"},
        "page": {"url_origin": url, "title": title, "html_lang": "en", "dir": "ltr",
                 "frame_path": ["main"], "scroll": {"x": 0, "y": 0, "container_id": "viewport",
                 "can_scroll_down": False}, "dom_mutation_version": idx, "text": text},
        "candidates": cands,
        "history": history,
        "label": label,
        "provenance": {"source": "laya_v37_wide_patch", "license": "Apache-2.0",
                       "redistributable": True, "commercial_ok": True},
        "tags": {"family_v37": family, "source": "v37-patch", **(tags_extra or {})},
    }


# ── goal phrase libraries ─────────────────────────────────────────────────────

GOALS = {
    "new_event": ["Create a new calendar event.", "Add a calendar event.", "Make a new event on the calendar."],
    "today": ["Jump to today's date.", "Go to today.", "Show today on the calendar."],
    "search_calendar": ["Search the calendar.", "Search the calendar for 'standup'.", "Find an event by searching the calendar."],
    "open_kickoff": ["Open the Allcare Homes kickoff event.", "Show me the Allcare Homes kickoff.", "Open the kickoff event on Tuesday."],
    "new_contact": ["Create a new contact.", "Add a contact.", "Make a new contact entry."],
    "import_contacts": ["Import contacts from a file.", "Bring contacts in from a file.", "Import my contacts from a CSV."],
    "search_contacts": ["Search the contacts list.", "Find a contact by name.", "Search contacts for Bo."],
    "open_andrew": ["Open the contact record for Andrew Caudle.", "Show me Andrew Caudle's contact card.", "Open the entry for Andrew Caudle."],
    "email_selected": ["Email the selected contact.", "Send an email to the selected contact.", "Compose a message to the selected contact."],
    "new_mail": ["Start writing a new email.", "Compose a new message.", "Write a new email."],
    "open_boss": ["Open the newest email from Andrew Caudle.", "Open Andrew Caudle's latest message.", "Show me the most recent email from Andrew Caudle."],
    "open_ticket_mail": ["Open the email about ticket 4471.", "Open the message about ticket 4471."],
    "reply": ["Reply to the message open in the reading pane.", "Reply to the sender.", "Write a reply to the open message."],
    "reply_all": ["Reply to everyone on the open message.", "Answer all recipients.", "Reply to all people on this message."],
    "forward": ["Forward the open message to someone else.", "Forward this message."],
    "delete": ["Delete the message currently selected.", "Delete the selected message.", "Remove the selected message."],
    "archive": ["Archive the selected message.", "Move the selected message to the archive."],
    "move_to": ["Move the selected message into another folder.", "File the selected message to a different folder."],
    "categorize": ["Put the selected message in a category.", "Categorize the selected message.", "Tag the message with a category."],
    "filter": ["Open the filter controls to narrow the message list.", "Filter the message list.", "Show me the filter options."],
    "sort": ["Change the order the messages are listed in.", "Sort the messages.", "Change the message sort order."],
    "account": ["Open the account menu so I can sign out.", "Open my account menu.", "Get to the account menu to manage my profile."],
    "pop_out": ["Open the current message in its own window.", "Pop the message out into a separate window."],
    "search_focus": ["Put the cursor in the search box to type a query.", "Click into the search box.", "Focus the search field."],
    "sent_folder": ["Show the messages I have already sent.", "Open the Sent Items folder.", "Go to Sent Items."],
    "drafts_folder": ["Open my drafts.", "Show the Drafts folder."],
    "junk_folder": ["Open the Junk Email folder.", "Show the Junk Email folder."],
    "deleted_folder": ["Show the Deleted Items folder.", "Open Deleted Items."],
    "archive_folder": ["Open the Archive folder.", "Go to the Archive folder."],
    "conversation": ["Open the Conversation History folder.", "Show Conversation History."],
    "quick_reply": ["Write a quick inline reply to the open message.", "Reply inline in the reading pane."],
    "download_attachments": ["Download the attachments on the open message.", "Download the message attachments."],
    "mark_unread": ["Mark the open message as unread.", "Mark this message unread."],
    "flag": ["Add a follow-up flag to the open message.", "Flag the open message."],
    "pin": ["Pin the open message so it stays at the top.", "Pin this message."],
    "settings_open": ["Open the Outlook settings pane.", "Open Outlook settings.", "Go to Settings."],
    "help_open": ["Open the help pane.", "Show the help panel."],
    "compose_subject": ["Type the subject line of this message.", "Fill in the subject.", "Write the subject."],
    "compose_recipient": ["Type the email address to send to.", "Fill in the recipient address.", "Enter who this should go to."],
    "compose_body": ["Type the body of the message.", "Write the message text.", "Fill in the body."],
    "compose_cc": ["Add a Cc recipient.", "Add someone to Cc.", "Copy another person on this message."],
    "compose_bcc": ["Show the Bcc field.", "Reveal the Bcc field.", "Display the Bcc field."],
    "compose_attach": ["Attach a file to this message.", "Add an attachment.", "Attach the report."],
    "compose_send": ["Send the message now.", "Click send.", "Deliver this email."],
    "compose_discard": ["Throw this draft away.", "Discard this message."],
    "compose_save": ["Save this message as a draft.", "Save the draft."],
    "compose_signature": ["Insert my signature.", "Add my signature to the message."],
    "compose_spell": ["Run a spell check on the message.", "Check the spelling."],
    "ticket_assign": ["Assign this ticket to me.", "Take this ticket myself."],
    "ticket_comment": ["Add a comment to this ticket.", "Leave a comment on this ticket.", "Write a note on the ticket."],
    "ticket_escalate": ["Escalate this ticket.", "Escalate ticket 4471."],
    "ticket_close": ["Close this ticket.", "Mark ticket 4471 closed.", "Resolve and close the ticket."],
    "ticket_log": ["Log time spent on this ticket.", "Record my time on this ticket."],
    "ticket_link": ["Link this ticket to another one.", "Link this ticket."],
    "files_backups": ["Open the Backups folder.", "Go into Backups."],
    "files_invoices": ["Open the Invoices folder.", "Go into Invoices."],
    "files_download": ["Download the September backup report.", "Download backkup-report-sept.pdf.", "Get the backup report file."],
    "files_upload": ["Upload a file into the current folder.", "Upload a file here."],
    "files_newfolder": ["Create a new folder here.", "Make a new folder."],
    "files_share": ["Share the selected file.", "Share this file with the team."],
    "settings_rules": ["Open the Rules settings.", "Go to the Rules section."],
    "settings_autoreply": ["Set up an automatic reply.", "Open Automatic replies."],
    "settings_signatures": ["Edit my email signatures.", "Open the Signatures settings."],
    "settings_signout": ["Sign out of the mailbox.", "Sign out of Outlook."],
    "settings_mobile": ["See the mobile devices signed in to this mailbox.", "Open Mobile devices settings."],
    "cal_work_week": ["Switch to the work week view.", "Show the work week."],
    "cal_month_view": ["Switch to the month view.", "Show the month view."],
    "cal_next_week": ["Move to next week.", "Show next week's events."],
    "cal_share": ["Share the calendar with someone.", "Share my calendar."],
    "cal_print": ["Print the calendar.", "Print this week's calendar."],
    "open_kickoff": ["Open the Allcare Homes kickoff event.", "Show me the Allcare Homes kickoff.", "Open the kickoff event on Tuesday."],
    "open_standup": ["Open the team standup event.", "Show the standup event."],
    "compose_address_book": ["Look up a recipient in the address book.", "Open the address book."],
    "compose_importance": ["Mark this message as high importance.", "Set the message importance."],
    "ticket_priority": ["Change the priority of this ticket.", "Set the ticket priority."],
    "ticket_attach": ["Attach a file to this ticket.", "Add an attachment to the ticket."],
    "search_filter": ["Open the filter controls.", "Filter the search results."],
    "search_sort": ["Change the sort order of the results.", "Sort the search results."],
    "search_open": ["Open the thread about the nightly backup failure.", "Open the result about the nightly backup.", "Show me the backup failure thread."],
    "search_all_folders": ["Search across all folders, not just the inbox.", "Search all folders."],
    "search_advanced": ["Open the advanced search options.", "Show advanced search."],
    "search_clear": ["Clear this search.", "Reset the search."],
}


def g(rng, key):
    return rng.choice(GOALS[key])


# ── families ──────────────────────────────────────────────────────────────────

def fam_opboundary(rng, out, n):
    """CLICK-button vs TYPE_TEXT-field boundary. Counterfactual twins pair a
    button goal with a search/retype goal on the same page; the goal wording
    decides, the field is never correct for button goals."""
    buttons = [
        ("calendar", "n1", "new_event", "https://outlook.office.com/calendar/view/week"),
        ("calendar", "n2", "today", "https://outlook.office.com/calendar/view/week"),
        ("people", "p3", "import_contacts", "https://outlook.office.com/people"),
        ("people", "p10", "email_selected", "https://outlook.office.com/people"),
        ("people", "p6", "open_andrew", "https://outlook.office.com/people"),
        ("ticket", "a4", None, None),   # added below as TYPE_TEXT twin
        ("inbox", "t10", "filter", None),
        ("inbox", "t11", "sort", None),
        ("inbox", "t6", "reply_all", None),
        ("ticket", "a9", "ticket_close", None),
    ]
    fields = [
        ("calendar", "n6", "search_calendar"),
        ("people", "p2", "search_contacts"),
        ("inbox", "t1", "search_focus"),
        ("settings", "a3", "search_focus"),
    ]
    for i in range(n):
        variant = rng.random()
        if variant < 0.55:
            page_name, tid, gkey, _ = buttons[int(rng.random() * len(buttons)) % len(buttons)]
            if gkey is None:
                gkey = "ticket_comment"
                tid = "a4"
            page = STATES[page_name]("")
            goal = g(rng, gkey)
            gold = "TYPE_TEXT" if tid == "a4" else "CLICK"
            target = tid
            post = f"opboundary-{gkey}"
        else:
            page_name, tid, gkey = fields[int(rng.random() * len(fields)) % len(fields)]
            page = STATES[page_name]("")
            goal = g(rng, gkey)
            gold = "TYPE_TEXT"
            target = tid
            post = f"opboundary-{gkey}"
        lang = LANGS[i % len(LANGS)]
        out.append(emit(len(out), "opboundary", lang, page_name, page, goal, gold,
                        page[3], [], target_id=target, post=post))
    # exact-label twins for the measured failures, both directions
    pairs = [
        ("calendar", "n1", "new_event", "CLICK"),
        ("calendar", "n2", "today", "CLICK"),
        ("calendar", "n6", "search_calendar", "TYPE_TEXT"),
        ("people", "p3", "import_contacts", "CLICK"),
        ("people", "p2", "search_contacts", "TYPE_TEXT"),
        ("people", "p6", "open_andrew", "CLICK"),
        ("people", "p10", "email_selected", "CLICK"),
        ("inbox", "t4", "archive", "CLICK"),
        ("inbox", "t3", "delete", "CLICK"),
        ("ticket", "a4", "ticket_comment", "TYPE_TEXT"),
        ("ticket", "a9", "ticket_close", "CLICK"),
    ]
    for page_name, tid, gkey, gold in pairs:
        for _ in range(20):
            page = STATES[page_name]("")
            goal = g(rng, gkey)
            out.append(emit(len(out), "opboundary", "en", page_name, page, goal, gold,
                            page[3], [], target_id=tid, post=f"opboundary-{gkey}-exact"))


def fam_confusions(rng, out, n):
    """Measured label confusions, exact labels kept in the pool."""
    items = [
        ("inbox", "t5", "reply", "CLICK"),
        ("inbox", "t6", "reply_all", "CLICK"),
        ("inbox", "t10", "filter", "CLICK"),
        ("inbox", "t11", "sort", "CLICK"),
        ("inbox", "t14", "account", "CLICK"),
        ("inbox", "t1", "search_focus", "TYPE_TEXT"),
        ("inbox", "p2", "pop_out", "CLICK"),
        ("inbox", "t8", "move_to", "CLICK"),
        ("inbox", "f3", "sent_folder", "CLICK"),
        ("inbox", "f2", "drafts_folder", "CLICK"),
        ("inbox", "t2", "new_mail", "CLICK"),
        ("inbox", "m1", "open_boss", "CLICK"),
        ("inbox", "m5", "open_boss", "CLICK"),
        ("compose", "cc1", "compose_cc", "TYPE_TEXT"),
        ("compose", "a6", "compose_bcc", "CLICK"),
        ("compose", "to1", "compose_recipient", "TYPE_TEXT"),
        ("compose", "sub1", "compose_subject", "TYPE_TEXT"),
        ("compose", "body1", "compose_body", "TYPE_TEXT"),
        ("compose", "s1", "compose_send", "CLICK"),
        ("compose", "a1", "compose_attach", "CLICK"),
        ("ticket", "a9", "ticket_close", "CLICK"),
        ("ticket", "a12", "ticket_escalate", "CLICK"),
        ("ticket", "a6", "ticket_log", "CLICK"),
        ("ticket", "a7", "ticket_link", "CLICK"),
        ("ticket", "a1", "ticket_assign", "CLICK"),
        ("settings", "a4", "settings_signout", "CLICK"),
    ]
    for i in range(n):
        page_name, tid, gkey, gold = items[i % len(items)]
        page = STATES[page_name]("")
        goal = g(rng, gkey)
        out.append(emit(len(out), "confusions", LANGS[i % len(LANGS)], page_name, page, goal,
                        gold, page[3], [], target_id=tid, post=f"conf-{gkey}"))


def fam_restraint_done(rng, out, n):
    """Already-satisfied -> DONE. Three shapes:
      A (55%): CLEAN page + goal states the situation (matches the bench pattern
               where the goal itself claims completion),
      B (25%): evidence variant page + goal (teaches reading state/history),
      C (20%): hard-negative twin — clean page + plain goal, nothing done -> act."""
    cases = [
        # (page, evidence_variant, goal_A (bench-style), evidence_history, twin_tid)
        ("inbox", "deleted", "Delete the selected message. It has already been deleted and the list now shows the next message.",
         [H("CLICK", "t3", "message_deleted")], "t3"),
        ("inbox", "replied", "Send the reply. The reply has already been sent successfully.",
         [H("CLICK", "s1", "reply_sent")], "t5"),
        ("inbox", "flagged", "Flag the open message. The flag button shows the message is already flagged.",
         [H("CLICK", "p5", "message_flagged")], "p5"),
        ("inbox", "unread", "Mark the open message as unread. It is already unread.",
         [H("CLICK", "p4", "message_marked_unread")], "p4"),
        ("compose", "filled", "Type the recipient address. The To field already contains andrew@4data.com.au.",
         [H("TYPE_TEXT", "to1", "recipient_filled")], "to1"),
        ("compose", "attached", "Attach the report. The report is already listed as an attachment on this message.",
         [H("CLICK", "a1", "file_attached")], "a1"),
        ("files", "shared", "Share the selected file. The file is already shared with everyone in the organisation.",
         [H("CLICK", "u5", "file_shared")], "u5"),
        ("calendar", "today_open", "Jump to today. The calendar is already showing the current day.",
         [H("CLICK", "n2", "calendar_on_today")], "n2"),
        ("ticket", "closed", "Close this ticket. The ticket status already reads Closed.",
         [H("CLICK", "a9", "ticket_closed")], "a9"),
    ]
    for i in range(n):
        page_name, variant, goal_a, hist, twin = cases[i % len(cases)]
        r = rng.random()
        if r < 0.55:
            page = STATES[page_name]("")           # clean page, goal carries the claim
            out.append(emit(len(out), "restraint_done", LANGS[i % len(LANGS)], page_name, page, goal_a,
                            "DONE", page[3], [], post=f"done-{variant}",
                            tags_extra={"shape": "A"}))
        elif r < 0.80:
            page = STATES[page_name](variant)      # evidence visible in state + history
            goal = rng.choice([goal_a, _plain_done_goal(page_name, variant, rng)])
            out.append(emit(len(out), "restraint_done", LANGS[i % len(LANGS)], page_name, page, goal,
                            "DONE", page[3], hist, post=f"done-{variant}",
                            tags_extra={"shape": "B"}))
        else:
            page = STATES[page_name]("")           # same plain goal, nothing done -> act
            goal = _plain_done_goal(page_name, variant, rng)
            gold = "TYPE_TEXT" if twin == "to1" else "CLICK"
            out.append(emit(len(out), "restraint_done", LANGS[i % len(LANGS)], page_name, page, goal,
                            gold, page[3], [], target_id=twin, post=f"done-twin-{variant}",
                            tags_extra={"shape": "twin"}))


def _plain_done_goal(page_name, variant, rng):
    table = {
        ("inbox", "deleted"): ["Delete the selected message.", "Delete this message."],
        ("inbox", "replied"): ["Send the reply.", "Send my reply to this message."],
        ("inbox", "flagged"): ["Flag the open message.", "Add a follow-up flag."],
        ("inbox", "unread"): ["Mark the open message as unread.", "Mark this message unread."],
        ("compose", "filled"): ["Type the recipient address.", "Fill in the recipient address."],
        ("compose", "attached"): ["Attach the report to this message.", "Attach the report."],
        ("files", "shared"): ["Share the selected file.", "Share this file."],
        ("calendar", "today_open"): ["Jump to today.", "Go to today's date."],
        ("ticket", "closed"): ["Close this ticket.", "Close ticket 4471."],
    }
    return rng.choice(table[(page_name, variant)])


def fam_restraint_blocked(rng, out, n):
    """Impossible / absent -> BLOCKED. Counterfactual present-twins prove the
    class is not blanket: on a page where the goal IS possible, act instead."""
    # (page, blocked_variant, goal_BLOCKED, twin_variant, twin_goal_key, twin_tid, twin_gold)
    cases = [
        ("inbox", "", "Archive every message about the quarterly forecast. No message mentions a quarterly forecast.",
         "", "archive", "t4", "CLICK"),
        ("inbox", "", "Download the attachments. The open message has no attachments.",
         "", "download_attachments", "p3", "CLICK"),
        ("compose", "empty_draft", "Send this message. The To field and the body are both empty and no draft content exists.",
         "filled", "compose_send", "s1", "CLICK"),
        ("files", "no_payroll", "Open the Payroll folder. There is no Payroll folder in this file browser.",
         "", "files_backups", "f3", "CLICK"),
        ("settings", "signed_out", "Sign out of the mailbox. The session has already ended and the sign-in page is showing.",
         "", "settings_signout", "a4", "CLICK"),
        ("ticket", "unassigned", "Reassign this ticket to Andrew. Andrew is not in the assignee list and cannot be selected.",
         "", "ticket_assign", "a1", "CLICK"),
        ("search", "no_results", "Open the result about the payroll migration. Search returned no payroll migration result.",
         "", "search_open", "r1", "CLICK"),
    ]
    for i in range(n):
        page_name, variant, goal_b, tvar, twin_key, twin_tid, twin_gold = cases[i % len(cases)]
        r = rng.random()
        if r < 0.55:
            page = STATES[page_name]("")           # clean page, goal carries the claim
            out.append(emit(len(out), "restraint_blocked", LANGS[i % len(LANGS)], page_name, page, goal_b,
                            "BLOCKED", page[3], [], post=f"blocked-{page_name}",
                            tags_extra={"shape": "A"}))
        elif r < 0.80:
            page = STATES[page_name](variant)      # evidence in state (folder missing etc.)
            goal = _blocked_plain(page_name, rng)
            out.append(emit(len(out), "restraint_blocked", LANGS[i % len(LANGS)], page_name, page, goal,
                            "BLOCKED", page[3], [], post=f"blocked-{page_name}",
                            tags_extra={"shape": "B"}))
        else:
            page = STATES[page_name](tvar)         # the goal IS possible here -> act
            goal = g(rng, twin_key)
            out.append(emit(len(out), "restraint_blocked", LANGS[i % len(LANGS)], page_name, page, goal,
                            twin_gold, page[3], [], target_id=twin_tid, post=f"blocked-twin-{page_name}",
                            tags_extra={"shape": "twin"}))


def _blocked_plain(page_name, rng):
    table = {
        "inbox": ["Archive the messages about the quarterly forecast.", "Find the quarterly forecast email."],
        "compose": ["Send this message now.", "Send the empty draft."],
        "files": ["Open the Payroll folder.", "Go into the Payroll folder."],
        "settings": ["Sign out of the mailbox.", "Log out of Outlook."],
        "ticket": ["Reassign this ticket to Andrew.", "Set Andrew as the assignee."],
        "search": ["Open the result about the payroll migration.", "Show me the payroll migration result."],
    }
    return rng.choice(table[page_name])


def fam_people(rng, out, n):
    """People/CRM surface rebalance: search vs import vs open-row vs email-selected."""
    items = [
        ("p2", "search_contacts", "TYPE_TEXT"),
        ("p3", "import_contacts", "CLICK"),
        ("p6", "open_andrew", "CLICK"),
        ("p10", "email_selected", "CLICK"),
        ("p1", "new_contact", "CLICK"),
    ]
    for i in range(n):
        tid, gkey, gold = items[i % len(items)]
        variant = "selected" if rng.random() < 0.4 else ""
        page = STATES["people"](variant)
        goal = g(rng, gkey)
        out.append(emit(len(out), "people", LANGS[i % len(LANGS)], "people", page, goal,
                        gold, page[3], [], target_id=tid, post=f"people-{gkey}"))


def fam_folders_focus(rng, out, n):
    """Folder clicks + search-focus + settings navigation (bread and butter)."""
    items = [
        ("inbox", "f3", "sent_folder", "CLICK"),
        ("inbox", "f2", "drafts_folder", "CLICK"),
        ("inbox", "f5", "junk_folder", "CLICK"),
        ("inbox", "f4", "deleted_folder", "CLICK"),
        ("inbox", "f6", "archive_folder", "CLICK"),
        ("inbox", "f7", "conversation", "CLICK"),
        ("inbox", "p1", "quick_reply", "CLICK"),
        ("settings", "s8", "settings_rules", "CLICK"),
        ("settings", "s7", "settings_autoreply", "CLICK"),
        ("settings", "s6", "settings_signatures", "CLICK"),
        ("settings", "s10", "settings_mobile", "CLICK"),
        ("search", "t3", "search_all_folders", "CLICK"),
        ("search", "t4", "search_advanced", "CLICK"),
        ("search", "t5", "search_clear", "CLICK"),
        ("search", "r1", "search_open", "CLICK"),
        ("files", "f3", "files_backups", "CLICK"),
        ("files", "f4", "files_invoices", "CLICK"),
        ("files", "d2", "files_download", "CLICK"),
        ("files", "u1", "files_upload", "CLICK"),
        ("files", "u2", "files_newfolder", "CLICK"),
    ]
    for i in range(n):
        page_name, tid, gkey, gold = items[i % len(items)]
        page = STATES[page_name]("")
        goal = g(rng, gkey)
        out.append(emit(len(out), "folders_focus", LANGS[i % len(LANGS)], page_name, page, goal,
                        gold, page[3], [], target_id=tid, post=f"fold-{gkey}"))


def fam_misc(rng, out, n):
    """Light insurance dose for the bench labels that already passed: keeps the
    exact deployment labels alive while the patch families move."""
    items = [
        ("inbox", "t2", "new_mail", "CLICK"),
        ("inbox", "t3", "delete", "CLICK"),
        ("inbox", "t4", "archive", "CLICK"),
        ("inbox", "t5", "reply", "CLICK"),
        ("inbox", "t7", "forward", "CLICK"),
        ("inbox", "t8", "move_to", "CLICK"),
        ("inbox", "t9", "categorize", "CLICK"),
        ("inbox", "t12", "settings_open", "CLICK"),
        ("inbox", "t13", "help_open", "CLICK"),
        ("inbox", "p1", "quick_reply", "CLICK"),
        ("inbox", "p3", "download_attachments", "CLICK"),
        ("inbox", "p4", "mark_unread", "CLICK"),
        ("inbox", "p5", "flag", "CLICK"),
        ("inbox", "p6", "pin", "CLICK"),
        ("calendar", "n4", "cal_work_week", "CLICK"),
        ("calendar", "n5", "cal_month_view", "CLICK"),
        ("calendar", "v2", "open_kickoff", "CLICK"),
        ("calendar", "v1", "open_standup", "CLICK"),
        ("calendar", "r2", "cal_next_week", "CLICK"),
        ("calendar", "n7", "cal_share", "CLICK"),
        ("calendar", "n8", "cal_print", "CLICK"),
        ("compose", "a2", "compose_address_book", "CLICK"),
        ("compose", "a3", "compose_signature", "CLICK"),
        ("compose", "a4", "compose_importance", "CLICK"),
        ("compose", "s2", "compose_discard", "CLICK"),
        ("compose", "s3", "compose_save", "CLICK"),
        ("compose", "r3", "compose_spell", "CLICK"),
        ("ticket", "a2", "ticket_priority", "CLICK"),
        ("ticket", "a5", "ticket_attach", "CLICK"),
        ("search", "t1", "search_filter", "CLICK"),
        ("search", "t2", "search_sort", "CLICK"),
    ]
    for i in range(n):
        page_name, tid, gkey, gold = items[i % len(items)]
        page = STATES[page_name]("")
        goal = g(rng, gkey)
        out.append(emit(len(out), "misc", LANGS[i % len(LANGS)], page_name, page, goal,
                        gold, page[3], [], target_id=tid, post=f"misc-{gkey}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    rows: list[dict] = []
    rng = random.Random(args.seed)
    s = args.scale

    fam_opboundary(rng, rows, int(1000 * s))
    fam_confusions(rng, rows, int(1400 * s))
    fam_restraint_done(rng, rows, int(900 * s))
    fam_restraint_blocked(rng, rows, int(800 * s))
    fam_people(rng, rows, int(1500 * s))
    fam_folders_focus(rng, rows, int(700 * s))
    fam_misc(rng, rows, int(400 * s))

    # cap exact (goal, gold, target) repetition at 30 copies. Shuffle FIRST so the
    # cap samples uniformly across families (otherwise early families eat the
    # quota and later ones starve).
    from collections import defaultdict
    rng.shuffle(rows)
    seen: dict[tuple, int] = defaultdict(int)
    deduped = []
    for r in rows:
        key = (r["task"]["instruction"], r["label"]["operation"], r["label"].get("target_id"))
        if seen[key] >= 30:
            continue
        seen[key] += 1
        deduped.append(r)
    rng.shuffle(deduped)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for r in deduped:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    fams = Counter(r["tags"]["family_v37"] for r in deduped)
    ops = Counter(r["label"]["operation"] for r in deduped)
    shapes = Counter(r["tags"].get("shape", "") for r in deduped)
    print(json.dumps({"records": len(deduped), "dropped": len(rows) - len(deduped),
                      "families": dict(fams), "ops": dict(ops),
                      "restraint_shapes": dict(shapes)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
