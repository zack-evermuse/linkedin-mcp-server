"""Invitation actions taken on a loaded person page.

``connection.py`` answers what a profile's action area *says* about the
relationship and stays browser-free; this module is everything that reads
or touches that area: the structural signal probe, the More menu, the
incoming-request Accept click, the invite dialog, the non-submitting
note-quota probe and the verification re-read after a write.

Per the AGENTS.md LinkedIn Page Rules every decision here rests on a URL pattern
(``/preload/custom-invite/?vanityName=USER``, ``/in/USER/edit/intro/``,
``/messaging/compose/``), on the *presence* of an ARIA attribute
(``aria-label`` on a button versus an anchor, ``aria-expanded`` on the menu
opener) or on a structural count. No label value is read anywhere, so a
German or an opaquely labelled page classifies exactly as an English one;
``tests/test_action_signals_dom.py`` holds that line against a real DOM in
all four label sets.

The write gate is the reason the order of the checks below matters: the
invite deeplink fires only after ``has_invite_anchor`` is true, and the only
other thing that may open it is the note-quota probe, which never clicks a
primary button.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote_plus

import asyncio
import logging

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.destination import (
    linkedin_element,
    linkedin_handle,
    raise_if_off_linkedin,
)
from linkedin_mcp_server.core.exceptions import OffLinkedInLandingError
import linkedin_mcp_server.linkedin.connection as connection
from linkedin_mcp_server.linkedin.connection import ActionSignals
from linkedin_mcp_server.linkedin.identifiers import (
    normalize_person_identifier,
    person_profile_url,
)
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.session import PageSession

logger = logging.getLogger(__name__)

# Every dialog selector below is scoped to LinkedIn's modal outlet, and that
# scoping is load-bearing -- not tidiness.
#
# `[role="dialog"]` on its own is NOT unique to modals. LinkedIn's messaging
# overlay renders each open chat bubble as a `[role="dialog"]` inside
# `ASIDE#msg-overlay`, and those bubbles hydrate about a second after
# `domcontentloaded`. So on any account with open message threads, an
# unscoped `dialog[open], [role="dialog"]` matched the invite modal *plus*
# every chat bubble, and the positional button indexing this module relies on
# (`nth(count - 1)` for the primary action, `nth(count - 2)` for the
# secondary) then indexed into a list spanning all of them.
#
# Measured live 2026-07-29 on an account with two chat bubbles open:
#   unscoped:  3 dialogs, 51 buttons -> nth(50) = "Open send options",
#                                       nth(49) = "Send"  (both belong to a
#                                       chat bubble, not the invitation)
#   outlet:    1 dialog,   3 buttons -> nth(2)  = "Send without a note",
#                                       nth(1)  = "Add a note"  (correct)
# That is the whole of the "deeplink opens no dialog" defect: the deeplink
# opens the dialog fine, but Send was clicked in a chat window, the invite
# modal never closed, and the caller reported connect_unavailable. It looked
# profile-dependent only because it is really a race with overlay hydration.
#
# `aria-modal` is not usable as the discriminator -- LinkedIn does not set it
# on the invite dialog (verified live). The outlet id is the available
# structural signal, and it is locale-independent per the AGENTS.md Scraping
# Rules: an element id, not layout classes and not UI copy. If LinkedIn ever
# renames it, every helper here reports "no dialog" and callers fail closed
# with connect_unavailable rather than clicking something unintended --
# `_dialog_is_open` logs explicitly when that happens.
_MODAL_OUTLET_SELECTOR = "#artdeco-modal-outlet"
_DIALOG_SELECTOR = (
    f'{_MODAL_OUTLET_SELECTOR} dialog[open], {_MODAL_OUTLET_SELECTOR} [role="dialog"]'
)
_DIALOG_PREMIUM_LINK_SELECTOR = (
    f'{_MODAL_OUTLET_SELECTOR} dialog[open] a[href*="/premium/"], '
    f'{_MODAL_OUTLET_SELECTOR} [role="dialog"] a[href*="/premium/"]'
)
_DIALOG_TEXTAREA_SELECTOR = (
    f'{_MODAL_OUTLET_SELECTOR} [role="dialog"] textarea, '
    f"{_MODAL_OUTLET_SELECTOR} dialog textarea"
)
# LinkedIn gates some invitations behind the recipient's email address
# ("we need to verify you know this person"). Detected by input *type*,
# which is an HTML attribute value rather than UI copy, so it holds across
# locales. Only the account owner can answer that prompt, so the tool
# reports it instead of attempting to satisfy it.
_DIALOG_EMAIL_INPUT_SELECTOR = (
    f'{_MODAL_OUTLET_SELECTOR} [role="dialog"] input[type="email"], '
    f'{_MODAL_OUTLET_SELECTOR} dialog input[type="email"]'
)
# Any dialog anywhere, used only to tell "LinkedIn rendered nothing" apart
# from "LinkedIn rendered a dialog somewhere we no longer recognise". Never
# use this to click: that is exactly the bug described above.
_ANY_DIALOG_SELECTOR = 'dialog[open], [role="dialog"]'

# Shared JS function that walks up from any /messaging/compose/ anchor
# inside <main> to find the smallest ancestor that satisfies the
# action-root predicate (>=2 interactive children, >=1 button). This is
# the top-card action row regardless of LinkedIn's class names.
#
# Inlined into both ACTION_SIGNALS_JS and OPEN_MORE_BUTTON_JS so a
# single change to the heuristic propagates to both call sites.
_FIND_ACTION_ROOT_FN_JS = r"""
function findActionRoot(main) {
  const composeAnchors = main.querySelectorAll('a[href*="/messaging/compose/"]');
  for (const a of composeAnchors) {
    let el = a.parentElement;
    while (el && el !== main) {
      const interactive = el.querySelectorAll('button, a').length;
      const buttons = el.querySelectorAll('button').length;
      if (interactive >= 2 && buttons >= 1) {
        return el;
      }
      el = el.parentElement;
    }
  }
  return null;
}
"""

# Shared JS function that fingerprints the incoming-request action row.
# Incoming-request profiles render no Message button in the top card, so
# findActionRoot (compose-anchor walk) cannot locate their action row and
# would mis-anchor on sidebar mutual-connection cards instead. This walk
# anchors on button[aria-expanded] (the More button) and validates the
# smallest multi-button ancestor against the fingerprint verified live
# 2026-06-11 on two German-locale incoming-request profiles:
#
#   [button aria-label (Accept)] [button aria-label (Ignore)]
#   [button aria-expanded, no aria-label (More)]
#
# All checks are attribute presence and structural counts per the
# AGENTS.md LinkedIn Page Rules — no label values are read. Every guard kills
# a known false positive: total-button-count === 3 and labeled === 2
# exclude video-player control bars (play/mute/captions all carry
# aria-label); the unlabeled-expander check excludes player settings
# expanders (the profile More button never carries aria-label); the
# DOM-order guard excludes bars with trailing labeled buttons; the
# compose/invite/labeled-anchor exclusions kill pending, connected top
# cards and sidebar cards. The scan continues over ALL expander candidates
# because cover-video profiles render the player's expander before the
# top-card row in DOM order.
#
# NOT excluded: creator-mode / high-follower cards that render
# [Follow][Save in Sales Navigator][More] with no Message button and
# Connect demoted into the More menu. They satisfy every guard above, so
# this fingerprint alone is NOT sufficient evidence of an incoming
# request — connect_with_person must disprove it via the More-menu invite
# probe before clicking Accept. Do not delete that probe on the grounds
# that the exclusions here look exhaustive; they are not (marc-banoub,
# 2026-07-29, where the first labeled button was Follow).
#
# The search is scoped to the top card — the first <section> of <main>
# (falling back to main's first child, then main). Profile pages render
# the action row in the top card; feed, "people also viewed", and other
# widgets live in later sections. Without the scope an unrelated widget
# elsewhere in main with the same button shape could be misclassified and
# its first labeled button clicked.
#
# Inlined into ACTION_SIGNALS_JS and CLICK_INCOMING_ACCEPT_JS so a
# single change to the fingerprint propagates to both call sites.
_FIND_INCOMING_ACTION_ROW_FN_JS = r"""
function findIncomingActionRow(main) {
  const scope = main.querySelector('section') || main.firstElementChild || main;
  const matches = [];
  for (const expander of scope.querySelectorAll('button[aria-expanded]')) {
    let el = expander.parentElement;
    while (el && el !== scope && el !== main) {
      if (el.querySelectorAll('button').length >= 2) {
        const buttons = el.querySelectorAll('button');
        const labeled = el.querySelectorAll('button[aria-label]');
        const expanders = el.querySelectorAll('button[aria-expanded]');
        if (
          buttons.length === 3 &&
          labeled.length === 2 &&
          expanders.length === 1 &&
          !expanders[0].hasAttribute('aria-label') &&
          expanders[0].compareDocumentPosition(labeled[1]) &
            Node.DOCUMENT_POSITION_PRECEDING &&
          !el.querySelector('a[href*="/messaging/compose/"]') &&
          !el.querySelector('a[href*="/preload/custom-invite/"]') &&
          !el.querySelector('a[aria-label]')
        ) {
          matches.push(el);
        }
        break;
      }
      el = el.parentElement;
    }
  }
  // Require a unique match: a profile's top card has exactly one action
  // row. Ambiguity (two rows matching the shape) is treated as no match so
  // the irreversible Accept click never fires on a guessed control.
  return matches.length === 1 ? matches[0] : null;
}
"""

# Locale-independent connection-state probe. Returns four booleans;
# per AGENTS.md LinkedIn Page Rules, every signal is based on URL patterns
# or ARIA-attribute *presence* — never on label text values.
#
# - hasInvite: vanityName-scoped invite anchor anywhere in document.
#   Searches document (not main) so a post-More-menu reread sees
#   portal-rendered menu items. The vanityName parameter is unique to
#   the target user, so document-wide search has no false-positive risk.
# - hasComposeInActionRoot: any /messaging/compose/ anchor exists inside
#   the action root. Scoped to main (not document) to avoid the More
#   menu's "Send profile in a message" anchor, which is a compose URL
#   but lives outside the action area.
# - hasEditIntro: edit-intro URL exists, only rendered on own profile.
# - hasLabeledActionButton: at least one <button[aria-label]> inside the
#   action root. Primary action buttons (Follow / Connect /
#   Save in Sales Navigator) carry aria-label for screen readers; the
#   profile More button uses aria-expanded instead and is not counted.
# - hasLabeledActionAnchor: at least one <a[aria-label]> inside the
#   action root. LinkedIn renders the Pending state as an anchor (linking
#   back to the profile URL) carrying aria-label like "Pending, click to
#   withdraw…". The Message anchor has only aria-disabled, so a labeled
#   anchor is the locale-independent Pending signal.
# - hasIncomingActionRow: the incoming-request fingerprint matched (see
#   _FIND_INCOMING_ACTION_ROW_FN_JS). Computed independently of
#   findActionRoot, which cannot locate the top-card row on incoming
#   profiles (no compose anchor there) and would mis-anchor on sidebar
#   cards.
#
# The username is CSS-escaped before interpolation into attribute
# selectors to defend against malformed inputs containing characters
# that would otherwise break the selector syntax (quotes, brackets).
ACTION_SIGNALS_JS = (
    r"""
((username) => {
"""
    + _FIND_ACTION_ROOT_FN_JS
    + _FIND_INCOMING_ACTION_ROW_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return null;

  const safe = CSS.escape(username);
  const inviteSel = `a[href*="/preload/custom-invite/?vanityName=${safe}"]`;
  const editSel = `a[href*="/in/${safe}/edit/intro/"]`;

  const hasInvite = !!document.querySelector(inviteSel);
  const hasEditIntro = !!main.querySelector(editSel);

  const actionRoot = findActionRoot(main);

  let hasComposeInActionRoot = false;
  let hasLabeledActionButton = false;
  let hasLabeledActionAnchor = false;
  if (actionRoot) {
    hasComposeInActionRoot =
      !!actionRoot.querySelector('a[href*="/messaging/compose/"]');
    for (const b of actionRoot.querySelectorAll('button')) {
      if (b.hasAttribute('aria-label')) {
        hasLabeledActionButton = true;
        break;
      }
    }
    for (const a of actionRoot.querySelectorAll('a')) {
      if (a.hasAttribute('aria-label')) {
        hasLabeledActionAnchor = true;
        break;
      }
    }
  }

  return {
    hasInvite,
    hasComposeInActionRoot,
    hasEditIntro,
    hasLabeledActionButton,
    hasLabeledActionAnchor,
    hasIncomingActionRow: !!findIncomingActionRow(main),
  };
})
"""
)

# Open the profile's More button, located inside the action root via the
# aria-expanded attribute. The aria-expanded attribute uniquely identifies
# the menu opener without text labels (the More button has no aria-label,
# while Follow/Connect/Pending buttons do — the inverse pattern). Returns
# true iff the click landed; the caller waits for [role='menu'] visibility
# before re-scanning signals.
OPEN_MORE_BUTTON_JS = (
    r"""
(() => {
"""
    + _FIND_ACTION_ROOT_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return false;
  const actionRoot = findActionRoot(main);
  if (!actionRoot) return false;
  const moreBtn = actionRoot.querySelector('button[aria-expanded]');
  if (!moreBtn) return false;
  moreBtn.click();
  return true;
})
"""
)

# Click Accept on an incoming-request profile. Accept is the FIRST labeled
# button in the fingerprinted row — primary actions render first in
# top-card action rows (Connect/Message lead on other profile states; the
# inverse of dialogs, where the primary button renders last). Clicking the
# second button would silently and irreversibly Ignore the request, so the
# click only fires when the full fingerprint matched.
CLICK_INCOMING_ACCEPT_JS = (
    r"""
(() => {
"""
    + _FIND_INCOMING_ACTION_ROW_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return false;
  const row = findIncomingActionRow(main);
  if (!row) return false;
  row.querySelectorAll('button[aria-label]')[0].click();
  return true;
})
"""
)

# Open the More menu of the *fingerprinted incoming-request row*, rather
# than of the compose-anchor action root.
#
# This deliberately does NOT reuse OPEN_MORE_BUTTON_JS. That helper walks
# up from a /messaging/compose/ anchor (findActionRoot), and the profiles
# that produce the incoming-request false positive are exactly the ones
# with no Message button in the top card — so findActionRoot returns null
# there and the menu would never open. The fingerprint already guarantees
# the row contains exactly one button[aria-expanded]; clicking that is the
# only reliable way to reach the menu on these cards.
OPEN_INCOMING_ROW_MORE_BUTTON_JS = (
    r"""
(() => {
"""
    + _FIND_INCOMING_ACTION_ROW_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return false;
  const row = findIncomingActionRow(main);
  if (!row) return false;
  const moreBtn = row.querySelector('button[aria-expanded]');
  if (!moreBtn) return false;
  moreBtn.click();
  return true;
})
"""
)


def _connection_result(
    url: str,
    status: str,
    message: str,
    *,
    note_sent: bool = False,
    profile: str = "",
) -> dict[str, Any]:
    """Build a structured response for a profile connection attempt."""
    result: dict[str, Any] = {
        "url": url,
        "status": status,
        "message": message,
        "note_sent": note_sent,
    }
    if profile:
        result["profile"] = profile
    return result


# One main-profile read of one member, by username. The person workflow owns
# that read and this one only ever needs its ``main_profile`` text, so the
# borrow is a callable rather than the reader: this module never learns what
# the facade is, and the day the read moves again only the wiring does.
ReadMainProfile = Callable[[str], Awaitable[dict[str, Any]]]


class ConnectionActions:
    """Send, accept and probe invitations for one LinkedIn member."""

    def __init__(
        self,
        session: PageSession,
        navigator: PageNavigator,
        read_main_profile: ReadMainProfile,
    ):
        self._session = session
        self._navigator = navigator
        self._read_main_profile = read_main_profile

    async def _dialog_is_open(self, *, timeout: int = 1000) -> bool:
        """Return whether a modal dialog is currently open (structural check).

        Waits for a *visible* modal to appear rather than sampling the DOM
        once: the previous implementation returned False immediately when
        ``count() == 0``, which made the ``timeout`` argument dead for the
        case it exists to cover -- a dialog that has not rendered yet. The
        page is navigated with ``wait_until="domcontentloaded"``, so a
        dialog mounted during hydration could be missed entirely.
        """
        try:
            await self._session.page.wait_for_selector(
                _DIALOG_SELECTOR, state="visible", timeout=timeout
            )
            return True
        except Exception:
            pass
        # No modal in the outlet. Distinguish "LinkedIn showed nothing" from
        # "LinkedIn showed a dialog somewhere _MODAL_OUTLET_SELECTOR no
        # longer covers" -- the second means this module's scoping has gone
        # stale and every invite will fail closed until it is updated. That
        # is worth a log line rather than a silent connect_unavailable.
        try:
            stray = await self._session.page.locator(_ANY_DIALOG_SELECTOR).count()
            if stray:
                logger.warning(
                    "No dialog inside %s, but %d dialog(s) exist elsewhere on "
                    "the page. If invites are failing, LinkedIn may have moved "
                    "the modal outlet and the selector needs updating.",
                    _MODAL_OUTLET_SELECTOR,
                    stray,
                )
        except Exception:
            logger.debug("Stray-dialog diagnostic failed", exc_info=True)
        return False

    async def _invite_dialog_requires_email(self) -> bool:
        """Return whether an open invite dialog is gated on a recipient email.

        Structural check on ``input[type="email"]`` inside the dialog, per
        the AGENTS.md Scraping Rules — an attribute value fixed by HTML,
        not localized UI copy. Verified live 2026-07-29 against
        angelosangelou, whose invite dialog opens normally but cannot be
        submitted without the recipient's address.
        """
        if not await self._dialog_is_open(timeout=1000):
            return False
        try:
            return (
                await self._session.page.locator(_DIALOG_EMAIL_INPUT_SELECTOR).count()
                > 0
            )
        except Exception:
            logger.debug("Email-gate probe failed", exc_info=True)
            return False

    async def _click_dialog_primary_button(self, *, timeout: int = 5000) -> bool:
        """Click the last (primary/Send) button in the open dialog.

        LinkedIn consistently places the primary action as the last button.
        Returns False (rather than raising) when the click is intercepted or
        times out, so callers can fall back to a keyboard submit.
        """
        buttons = self._session.page.locator(
            f"{_DIALOG_SELECTOR} button, {_DIALOG_SELECTOR} [role='button']"
        )
        count = await buttons.count()
        if count == 0:
            return False
        try:
            async with linkedin_element(
                buttons.nth(count - 1), timeout=timeout
            ) as button:
                await button.click(timeout=timeout)
            return True
        except OffLinkedInLandingError:
            raise
        except Exception:
            logger.debug("Primary dialog button click failed", exc_info=True)
            return False

    async def _fill_dialog_textarea(self, value: str, *, timeout: int = 5000) -> bool:
        """Fill the first textarea inside the open dialog (structural)."""
        locator = self._session.page.locator(_DIALOG_TEXTAREA_SELECTOR).first
        try:
            if await self._session.page.locator(_DIALOG_TEXTAREA_SELECTOR).count() == 0:
                return False
            async with linkedin_element(locator, timeout=timeout) as textarea:
                await textarea.fill(value, timeout=timeout)
            return True
        except OffLinkedInLandingError:
            raise
        except Exception:
            logger.debug("Invite note fill failed", exc_info=True)
            return False

    async def _press_escape(self) -> None:
        """Press Escape in the current document, and only if it is LinkedIn's.

        Never through `page.keyboard`, which delivers to whatever document is
        current: a portal that replaced the page runs its own key handlers on
        it. Pressed on the handle of the element that has focus, or the body
        when nothing does: a handle press focuses its element first, so
        pressing on the body would take focus from a dialog whose own handler
        listens for Escape whenever the body can be focused. A document
        replaced after the check detaches the handle and fails the press
        instead of receiving it.

        Raises:
            OffLinkedInLandingError: When the current document is not LinkedIn's.
        """
        focused = await self._session.page.evaluate_handle(
            "() => document.activeElement || document.body"
        )
        element = focused.as_element()
        if element is None:
            await focused.dispose()
            async with linkedin_element(self._session.page.locator("body")) as body:
                await body.press("Escape")
            return
        async with linkedin_handle(element) as target:
            await target.press("Escape")

    async def _dismiss_dialog(self) -> None:
        """Dismiss any open dialog via Escape key (structural)."""
        await self._press_escape()
        try:
            await self._session.page.wait_for_selector(
                _DIALOG_SELECTOR, state="hidden", timeout=3000
            )
        except PlaywrightTimeoutError:
            pass

    async def _get_premium_upsell_message(self, *, timeout: int = 2500) -> str | None:
        """Return the raw LinkedIn Premium upsell dialog text when visible.

        LinkedIn intercepts invite-with-note flows with an upsell modal when
        the free personalized-note quota is exhausted. The detector itself is
        locale-independent: the modal links to ``/premium/...``. The returned
        message is the dialog text as rendered by LinkedIn, not a synthesized
        explanation.
        """
        locator = self._session.page.locator(_DIALOG_PREMIUM_LINK_SELECTOR).first
        try:
            await locator.wait_for(state="visible", timeout=timeout)
        except PlaywrightTimeoutError:
            return None
        except Exception:
            try:
                if not await locator.is_visible():
                    return None
            except Exception:
                return None

        try:
            message = await self._session.run_on_linkedin(
                """(selector) => {
                    const link = document.querySelector(selector);
                    const dialog = link?.closest('dialog,[role="dialog"]');
                    return dialog?.innerText || dialog?.textContent || link?.innerText || '';
                }""",
                # Same outlet-scoped selector the locator above used. Kept as
                # an argument rather than inlined so the two can never drift.
                _DIALOG_PREMIUM_LINK_SELECTOR,
            )
            if isinstance(message, str) and message.strip():
                return message.strip()
        except OffLinkedInLandingError:
            raise
        except Exception:
            logger.debug("Could not read Premium upsell dialog text", exc_info=True)

        # The snapshot above can fail because the page left mid-read, and then
        # neither the link nor the bare fact of a dialog is LinkedIn's. Both
        # are taken only from a page that still is.
        try:
            async with linkedin_element(locator, timeout=timeout) as link:
                link_text = await link.inner_text()
            if link_text.strip():
                return link_text.strip()
        except OffLinkedInLandingError:
            raise
        except Exception:
            pass
        raise_if_off_linkedin(self._session.page.url)
        return "LinkedIn Premium upsell modal detected."

    async def _open_more_menu(self) -> bool:
        """Open the profile's More (three-dot) menu in a locale-independent way.

        Locates the More button structurally as ``actionRoot
        button[aria-expanded]`` — the action-root walk discriminates the
        profile More button from any other More-labelled buttons elsewhere
        on the page (notably the video-player More on profiles with
        background videos), and ``aria-expanded`` distinguishes the menu
        opener from primary action buttons (which carry ``aria-label``
        instead). Returns True iff the click landed and a ``[role='menu']``
        became visible. The caller is expected to follow up with
        ``_read_action_signals`` to scan the now-rendered menu items for
        the vanityName invite anchor; this helper does not classify menu
        contents itself.
        """
        try:
            clicked = await self._session.run_on_linkedin(OPEN_MORE_BUTTON_JS)
        except OffLinkedInLandingError:
            raise
        except Exception:
            logger.debug("More button click via JS failed", exc_info=True)
            return False
        if not clicked:
            return False
        try:
            await self._session.page.wait_for_selector("[role='menu']", timeout=3000)
            return True
        except PlaywrightTimeoutError:
            logger.debug("More menu did not appear after click")
            return False

    async def _click_incoming_accept(self) -> bool:
        """Click Accept on an incoming-request profile, locale-independently.

        Delegates to ``CLICK_INCOMING_ACCEPT_JS``: the click fires only
        when the full incoming-row fingerprint matches, and it targets the
        FIRST labeled button (Accept renders before Ignore — primary
        actions lead in top-card rows). Clicking the second button would
        silently and irreversibly Ignore the request; the strict
        fingerprint plus the caller's verify-after-click are the
        mitigations. Returns True iff the click landed.
        """
        try:
            return bool(await self._session.run_on_linkedin(CLICK_INCOMING_ACCEPT_JS))
        except OffLinkedInLandingError:
            raise
        except Exception:
            logger.debug("Incoming accept click via JS failed", exc_info=True)
            return False

    async def _open_incoming_row_more_menu(self) -> bool:
        """Open the More menu of the fingerprinted incoming-request row.

        Same contract as ``_open_more_menu`` (True iff the click landed and
        a ``[role='menu']`` became visible), but anchored on the incoming
        row's own ``button[aria-expanded]`` instead of the compose-anchor
        action root. ``_open_more_menu`` cannot be used here: it locates the
        More button via ``findActionRoot``, which walks up from a
        ``/messaging/compose/`` anchor, and the creator-mode cards that
        trip the incoming-request fingerprint render no Message button at
        all — so it would return False on exactly the profiles this probe
        exists to disambiguate.
        """
        try:
            clicked = await self._session.page.evaluate(
                OPEN_INCOMING_ROW_MORE_BUTTON_JS
            )
        except Exception:
            logger.debug("Incoming-row More click via JS failed", exc_info=True)
            return False
        if not clicked:
            return False
        try:
            await self._session.page.wait_for_selector("[role='menu']", timeout=3000)
            return True
        except PlaywrightTimeoutError:
            logger.debug("Incoming-row More menu did not appear after click")
            return False

    async def _read_action_signals(self, username: str) -> ActionSignals:
        """Read locale-independent structural signals for a profile's
        relationship state.

        Detection uses URL patterns and ARIA attribute presence only — never
        text values — per the AGENTS.md LinkedIn Page Rules. The vanityName invite
        anchor is searched document-wide because LinkedIn renders the More
        menu's contents in a portal-mounted ``[role='menu']`` outside ``<main>``;
        the URL is uniquely scoped to the target user, so document-wide
        search introduces no false positives. The compose anchor used for
        action-root discovery is scoped to ``<main>`` to avoid the
        portal-rendered "Send profile in a message" anchor that appears
        inside the More menu after click.
        """
        data = await self._session.run_on_linkedin(ACTION_SIGNALS_JS, username)
        if not isinstance(data, dict):
            return ActionSignals(
                has_invite_anchor=False,
                has_compose_anchor_in_action_root=False,
                has_edit_intro_anchor=False,
                has_labeled_action_button=False,
                has_labeled_action_anchor=False,
                has_incoming_action_row=False,
            )
        return ActionSignals(
            has_invite_anchor=bool(data.get("hasInvite")),
            has_compose_anchor_in_action_root=bool(data.get("hasComposeInActionRoot")),
            has_edit_intro_anchor=bool(data.get("hasEditIntro")),
            has_labeled_action_button=bool(data.get("hasLabeledActionButton")),
            has_labeled_action_anchor=bool(data.get("hasLabeledActionAnchor")),
            has_incoming_action_row=bool(data.get("hasIncomingActionRow")),
        )

    async def _submit_invite_dialog(
        self, note: str | None
    ) -> tuple[bool, bool, str | None]:
        """Submit the invite dialog opened by the custom-invite deeplink.

        Returns ``(submitted, note_sent, note_limit_message)``.

        ``note_sent`` reports *delivery*, not textarea fill — it stays
        False on any failure path, including the Premium upsell that
        LinkedIn shows when the free personalized-note quota is exhausted.
        ``note_limit_message`` is the raw LinkedIn Premium dialog text when
        the upsell was detected; in that case ``submitted`` is False, the
        dialog is dismissed, and callers should surface that text directly.

        All interaction uses structural selectors and positional indexing
        — no localized text matching. Owns dialog cleanup: the dialog is
        dismissed on every failure path, callers must not dismiss again.
        """
        if not await self._dialog_is_open(timeout=5000):
            return False, False, None

        note_filled = False
        if note:
            textarea_count = await self._session.page.locator(
                _DIALOG_TEXTAREA_SELECTOR
            ).count()
            if textarea_count == 0:
                # Reveal the note textarea via the secondary action.
                # Two layouts are now in the wild and both place "Add a
                # note" at index ``btn_count - 2``:
                #   * Legacy invite dialog (3 buttons): dismiss, secondary
                #     "Add a note", primary "Send" -> nth(1) is secondary.
                #   * "Add a note to your invitation?" gating dialog (2
                #     buttons, rolled out 2026-05): "Add a note",
                #     "Send without a note" -> nth(0) is the only path
                #     that mounts the textarea. See issue #455.
                # If LinkedIn ever serves a 2-button dismiss/primary
                # no-note layout, the click below misroutes to dismiss;
                # the textarea-presence recheck via _fill_dialog_textarea
                # then fails and the caller returns connect_unavailable
                # without sending — the same outcome as today.
                buttons = self._session.page.locator(
                    f"{_DIALOG_SELECTOR} button, {_DIALOG_SELECTOR} [role='button']"
                )
                btn_count = await buttons.count()
                if btn_count >= 2:
                    async with linkedin_element(buttons.nth(btn_count - 2)) as add_note:
                        await add_note.click()
                    textarea_appeared = True
                    try:
                        await self._session.page.wait_for_selector(
                            _DIALOG_TEXTAREA_SELECTOR,
                            state="visible",
                            timeout=3000,
                        )
                    except PlaywrightTimeoutError:
                        logger.debug("Note textarea did not appear")
                        textarea_appeared = False
                    # ponytail: LinkedIn now renders a persistent Premium
                    # nudge banner on this step even when quota is NOT
                    # exhausted (observed: "3 personalized invitations
                    # remaining this month" alongside a live, fillable
                    # textarea). Bailing on banner presence alone false-
                    # positives on every note send. Only treat it as a
                    # real block when the textarea never mounted at all —
                    # the one case where LinkedIn actually replaces the
                    # note UI with the upsell instead of showing both.
                    if not textarea_appeared:
                        note_limit_message = await self._get_premium_upsell_message()
                        if note_limit_message is not None:
                            logger.info(
                                "Premium upsell blocked opening invite note editor"
                            )
                            await self._dismiss_dialog()
                            return False, False, note_limit_message

            note_filled = await self._fill_dialog_textarea(note)
            if not note_filled:
                # Same gate as the reveal step: the Premium nudge banner sits
                # beside a live textarea, so a failed fill is a quota block
                # only once no visible textarea is left. A count that fails
                # proves no absence, so it claims no block either: a false
                # block invites the caller to resend without the note.
                try:
                    textarea_visible = (
                        await self._session.page.locator(
                            f"{_DIALOG_TEXTAREA_SELECTOR} >> visible=true"
                        ).count()
                        > 0
                    )
                except Exception:
                    textarea_visible = True
                if textarea_visible:
                    logger.info(
                        "Invite note fill failed without evidence of a quota block"
                    )
                    await self._dismiss_dialog()
                    return False, False, None
                note_limit_message = await self._get_premium_upsell_message()
                if note_limit_message is not None:
                    logger.info("Premium upsell blocked filling invite note")
                    await self._dismiss_dialog()
                    return False, False, note_limit_message
                await self._dismiss_dialog()
                return False, False, None

        sent = await self._click_dialog_primary_button()
        if not sent:
            # Fallback: focus the primary button positionally so a subsequent
            # Enter targets it instead of a focused textarea (where Enter
            # would just insert a newline).
            buttons = self._session.page.locator(
                f"{_DIALOG_SELECTOR} button, {_DIALOG_SELECTOR} [role='button']"
            )
            btn_count = await buttons.count()
            if btn_count > 0:
                try:
                    # Through the handle, so Enter lands on this button in
                    # this document rather than on whatever holds focus.
                    async with linkedin_element(buttons.nth(btn_count - 1)) as button:
                        await button.press("Enter")
                    sent = not await self._dialog_is_open(timeout=2000)
                except OffLinkedInLandingError:
                    raise
                except Exception:
                    logger.debug("Keyboard submit fallback failed", exc_info=True)
            if not sent:
                # The Send click can also fail because LinkedIn swapped the
                # invite dialog for the Premium upsell at submit time — the
                # original primary button is then detached or pointer-event
                # covered, so the click raises or times out. Check for the
                # upsell here so we surface the raw note-limit message
                # instead of dismissing silently and returning
                # connect_unavailable.
                if note:
                    note_limit_message = await self._get_premium_upsell_message()
                    if note_limit_message is not None:
                        logger.info(
                            "Premium upsell modal intercepted invite submit click"
                        )
                        await self._dismiss_dialog()
                        return False, False, note_limit_message
                await self._dismiss_dialog()
                return False, False, None

        dialog_closed = True
        try:
            await self._session.page.wait_for_selector(
                _DIALOG_SELECTOR, state="hidden", timeout=5000
            )
        except PlaywrightTimeoutError:
            logger.debug("Invite dialog did not close after submit")
            dialog_closed = False

        # LinkedIn may swap the invite dialog for a Premium upsell when the
        # free note quota is exhausted, instead of closing it after Send —
        # the textarea was filled but the invite was not delivered, so
        # surface LinkedIn's raw dialog text. Gated on the dialog still
        # being open: the same benign nudge banner that can sit alongside a
        # live, fillable textarea (see the reveal-step fix above) can also
        # still be in the DOM for a moment right after a successful Send,
        # before it unmounts with the closing dialog. Checking unconditionally
        # here would report a genuinely delivered invite as blocked. A dialog
        # that failed to close is real evidence something went wrong; one
        # that closed on schedule is not, banner or no banner.
        if note and not dialog_closed:
            note_limit_message = await self._get_premium_upsell_message()
            if note_limit_message is not None:
                logger.info("Premium upsell modal intercepted invite submit")
                await self._dismiss_dialog()
                return False, False, note_limit_message

        return True, note_filled, None

    async def _probe_invite_note_limit(self) -> str | None:
        """Open the note editor only to read a Premium note-quota message.

        This is used when the profile did not expose the normal invite anchor.
        Navigating to the custom-invite deeplink and opening the note editor is
        non-destructive, but submitting would weaken the write gate for
        follow-only/unavailable profiles. Therefore this helper never clicks
        the primary Send button: it returns the raw LinkedIn Premium dialog
        text if LinkedIn shows it while opening the note editor, then
        dismisses the dialog.
        """
        if not await self._dialog_is_open(timeout=5000):
            return None
        note_limit_message = await self._get_premium_upsell_message(timeout=500)
        if note_limit_message is not None:
            await self._dismiss_dialog()
            return note_limit_message

        try:
            textarea_count = await self._session.page.locator(
                _DIALOG_TEXTAREA_SELECTOR
            ).count()
        except Exception:
            textarea_count = 0
        if textarea_count > 0:
            await self._dismiss_dialog()
            return None

        buttons = self._session.page.locator(
            f"{_DIALOG_SELECTOR} button, {_DIALOG_SELECTOR} [role='button']"
        )
        try:
            btn_count = await buttons.count()
        except Exception:
            btn_count = 0
        if btn_count >= 3:
            try:
                async with linkedin_element(buttons.nth(btn_count - 2)) as add_note:
                    await add_note.click()
            except OffLinkedInLandingError:
                raise
            except Exception:
                logger.debug("Could not open invite note editor", exc_info=True)
            try:
                await self._session.page.wait_for_selector(
                    _DIALOG_TEXTAREA_SELECTOR,
                    state="visible",
                    timeout=3000,
                )
            except PlaywrightTimeoutError:
                logger.debug("Note textarea did not appear during quota probe")

        note_limit_message = await self._get_premium_upsell_message()
        await self._dismiss_dialog()
        return note_limit_message

    async def _send_via_custom_invite_deeplink(
        self,
        url: str,
        username: str,
        note: str | None,
        page_text: str,
    ) -> dict[str, Any] | None:
        """Send an invitation through the custom-invite deeplink.

        Returns a connection result, or ``None`` when LinkedIn did not open an
        invite dialog for this vanityName — the caller decides how to report
        that, because "no dialog" means different things on the write-gate
        path (not connectable) and the incoming-request path (disproof).

        WHY this is the write path now (2026-08-26): LinkedIn removed the
        ``a[href*="/preload/custom-invite/?vanityName="]`` anchor from the
        profile DOM. Measured zero such anchors on every profile tested,
        including 2nd-degree cards showing a plainly visible, clickable
        Connect button. Every gate keyed on that anchor therefore rejected
        100% of sends, and ``connect_with_person`` could not send at all --
        it returned ``connect_unavailable`` before ever opening the deeplink.
        The deeplink itself still works and still opens the invite flow.

        Identity safety is preserved WITHOUT the anchor: the deeplink URL
        carries ``?vanityName=<username>``, so LinkedIn resolves it to exactly
        the requested person. That is the same guarantee the anchor gave.

        Do NOT "fix" this by relaxing the DOM selector to "any Connect
        control". The sidebar ("Explore Premium profiles") renders Connect
        anchors for OTHER people whose href points back at the current page,
        so a loosened selector invites the wrong person. The deeplink URL is
        the only identity-bearing signal left.
        """
        invite_url = (
            "https://www.linkedin.com/preload/custom-invite/"
            f"?vanityName={quote_plus(username)}"
        )
        await self._navigator._navigate_to_page(invite_url)

        # Sendability gate: LinkedIn actually opening an invite dialog for
        # this vanityName replaces the deleted anchor as the signal. Callers
        # reach here only after already_connected and pending have returned,
        # so an open dialog is not an existing relationship.
        if not await self._dialog_is_open(timeout=5000):
            return None

        # Explicit pre-submit guardrail. A dialog opening is NOT by itself
        # permission to send.
        #
        # For genuinely restricted / follow-only profiles LinkedIn still opens
        # the invite dialog, but gates it on the recipient's email address and
        # disables the send control. Verified live 2026-08-26 on
        # williamhgates: the dialog reads "To verify this member knows you,
        # please enter their email to connect", renders an email input, and
        # "Send without a note" is disabled.
        #
        # Check that here rather than letting _submit_invite_dialog attempt a
        # send and fail, so the write path is never entered for a profile
        # LinkedIn will not let us invite. This is what preserves the
        # follow-only guardrail now that the vanityName anchor -- the signal
        # that guardrail used to rest on -- no longer exists.
        if await self._invite_dialog_requires_email():
            await self._dismiss_dialog()
            return _connection_result(
                url,
                "manual_send_required",
                "LinkedIn requires this person's email address before the "
                "invitation can be sent. Send it manually.",
                note_sent=False,
                profile=page_text,
            )

        submitted, note_sent, note_limit_message = await self._submit_invite_dialog(
            note
        )
        if note_limit_message is not None:
            return _connection_result(
                url,
                "custom_note_limit_reached",
                note_limit_message,
                note_sent=False,
                profile=page_text,
            )
        if not submitted:
            if await self._invite_dialog_requires_email():
                await self._dismiss_dialog()
                return _connection_result(
                    url,
                    "manual_send_required",
                    "LinkedIn requires this person's email address before the "
                    "invitation can be sent. Send it manually.",
                    note_sent=False,
                    profile=page_text,
                )
            return _connection_result(
                url,
                "connect_unavailable",
                "LinkedIn did not open a usable invite dialog for this profile.",
                profile=page_text,
            )

        verified = await self._read_main_profile(username)
        verified_signals = await self._read_action_signals(username)
        if verified_signals.has_invite_anchor:
            # The same settle retry as the accept path: an immediate re-read
            # can still render Connect for an invitation LinkedIn already
            # recorded (observed live 2026-09-26: send_failed, then Pending).
            # Only a pending or already accepted invitation is evidence it
            # landed.
            await asyncio.sleep(3.0)
            retry = await self._read_main_profile(username)
            retry_signals = await self._read_action_signals(username)
            if connection.detect_connection_state(retry_signals) in (
                "pending",
                "already_connected",
            ):
                verified, verified_signals = retry, retry_signals
        verified_text = verified.get("sections", {}).get("main_profile", "")
        verified_state = connection.detect_connection_state(verified_signals)

        if verified_signals.has_invite_anchor:
            return _connection_result(
                url,
                "send_failed",
                "Submitted the invite dialog but the profile still exposes Connect.",
                note_sent=note_sent,
                profile=verified_text or page_text,
            )

        # Post-send verification, anchor-free.
        #
        # The historical check above ("still exposes Connect") is now vacuous:
        # with the anchor deleted, has_invite_anchor is False for everyone, so
        # it can never fire and would report success unconditionally. That is
        # the dangerous direction -- a silent submit failure reported as
        # `connected` writes a false dedup record and the person is never
        # contacted again.
        #
        # `pending` is the surviving positive signal: LinkedIn renders it as an
        # anchor carrying an aria-label, which detect_connection_state already
        # reads locale-independently. It is not universal -- creator-mode /
        # Follow-primary cards do NOT flip to Pending after a successful send
        # (verified by hand 2026-08-26, mike-koh) -- so absence is not proof of
        # failure. Report the distinction honestly instead of flattening it.
        if verified_state == "pending":
            message = "Connection request sent. Confirmed: profile now shows Pending."
        elif verified_state == "already_connected":
            # Reachable only via the settle retry above: the recipient
            # accepted within the retry window. Stronger evidence than
            # Pending, not a failure to confirm -- report it as such rather
            # than falling into the "could not be positively confirmed" case
            # below.
            message = "Connection request sent. Confirmed: already connected."
        else:
            message = (
                "Connection request sent, but the post-send state could not be "
                "positively confirmed"
                + (f" (state: {verified_state})" if verified_state else "")
                + ". LinkedIn removed the invite anchor this check historically "
                "relied on, and creator-mode cards do not flip to Pending. "
                "Reconcile against the Sent invitation manager if certainty "
                "matters."
            )
        return _connection_result(
            url,
            "connected",
            message,
            note_sent=note_sent,
            profile=verified_text or page_text,
        )

    async def connect_with_person(
        self,
        username: str,
        *,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Send a LinkedIn connection request or accept an incoming one.

        Detection is locale-independent: classification uses URL patterns
        (vanityName invite anchor, edit-intro anchor) and ARIA-attribute
        presence on top-card buttons (`aria-label` for primary actions,
        `aria-expanded` for the More-menu opener). The deeplink-submit
        path is gated strictly on `has_invite_anchor=True` *after* the
        optional More-menu retry, so Pending and follow-only profiles
        cannot trigger a write. If a note was requested but no invite
        anchor is visible, the custom-invite deeplink may still be opened
        only as a non-submitting note-quota probe. Sending itself uses the
        ``/preload/custom-invite/?vanityName=`` deeplink, which works
        whether the user-visible Connect button is in the action bar
        or buried under the More menu.
        """
        username = normalize_person_identifier(username)
        url = person_profile_url(username, "/")

        profile = await self._read_main_profile(username)
        page_text = profile.get("sections", {}).get("main_profile", "")
        if not page_text:
            return _connection_result(
                url, "unavailable", "Could not read profile page."
            )

        signals = await self._read_action_signals(username)
        state = connection.detect_connection_state(signals)
        logger.info(
            "Connection signals for %s: state=%s signals=%s", username, state, signals
        )

        if state == "self_profile":
            return _connection_result(
                url,
                "connect_unavailable",
                "Cannot send a connection request to your own profile.",
                profile=page_text,
            )
        if state == "already_connected":
            return _connection_result(
                url,
                "already_connected",
                "You are already connected with this profile.",
                profile=page_text,
            )
        if state == "pending":
            return _connection_result(
                url,
                "pending",
                "A connection request is already pending for this profile.",
                profile=page_text,
            )

        if state == "incoming_request":
            # Disprove the classification before acting on it.
            #
            # The fingerprint is not unique to incoming requests. A
            # creator-mode / high-follower card renders exactly
            #   [button aria-label -> Follow]
            #   [button aria-label -> Save in Sales Navigator]
            #   [button aria-expanded, unlabeled -> More]
            # with no Message button (so no compose anchor to exclude it)
            # and Connect demoted into the More menu (so no invite anchor
            # until that menu is opened). Every exclusion in
            # _FIND_INCOMING_ACTION_ROW_FN_JS passes and the shape matches,
            # so _click_incoming_accept clicks the first labeled button —
            # Follow. That is an unintended, user-visible write on a
            # profile the user only meant to invite, and it is reported as
            # send_failed because the card never reaches 1st-degree.
            # Observed live 2026-07-29 (marc-banoub): the run followed him
            # and created no invitation.
            #
            # A Connect action and an invitation pending *from* that person
            # are mutually exclusive, so a vanityName invite anchor
            # surfacing under More is a decisive, locale-independent
            # disproof of incoming_request. Note these cards can never be
            # rescued by the follow_only More retry below: follow_only
            # requires a compose anchor in the action root, which they do
            # not have.
            probe_opened = await self._open_incoming_row_more_menu()
            if probe_opened:
                probed = await self._read_action_signals(username)
                # Close the menu before clicking Accept or navigating, so
                # the overlay cannot intercept either.
                try:
                    await self._session.page.keyboard.press("Escape")
                except Exception:
                    logger.debug(
                        "Escape after incoming More-menu probe failed", exc_info=True
                    )
                logger.info(
                    "Post-More incoming probe for %s: signals=%s", username, probed
                )
                if probed.has_invite_anchor:
                    logger.info(
                        "Invite anchor found under More for %s; reclassifying "
                        "incoming_request -> connectable",
                        username,
                    )
                    signals = probed
                    state = "connectable"

        if state == "incoming_request":
            # Fail closed when the disproof could not run. The fingerprint
            # guarantees the row holds exactly one button[aria-expanded],
            # so a menu that refuses to open is anomalous — and Accept is
            # irreversible while a missed accept is not. Report send_failed
            # and let the user accept manually rather than risk clicking
            # Follow on a misclassified card.
            if not probe_opened:
                return _connection_result(
                    url,
                    "send_failed",
                    "Could not open the More menu to confirm this is an incoming "
                    "request; refusing to click Accept.",
                    profile=page_text,
                )
            # Second fail-closed gate, on caller intent rather than DOM shape.
            #
            # The More-menu disproof above only clears cards that expose an
            # invite anchor. A creator-mode card that matches the fingerprint
            # AND has no anchor under More reaches here having disproved
            # nothing, and the Accept click below lands on its first labeled
            # button — Follow. Observed live 2026-07-29 (gal-melamed-2286221):
            # the caller asked to send an invitation with a note and instead
            # followed him, reported as send_failed.
            #
            # A caller that supplies a note has stated its intent: send an
            # invitation. Accepting an incoming request is a different action
            # with a different outcome, and it is never what a note-bearing
            # call wanted. So when the disproof came back empty and a note is
            # present, refuse rather than guess. This costs a genuine incoming
            # request nothing but a manual accept; guessing wrong costs an
            # unintended, user-visible write on someone else's profile.
            if note:
                logger.info(
                    "Disproof found no invite anchor for %s and a note was "
                    "requested; trying the custom-invite deeplink instead of "
                    "clicking Accept on a possible misclassification",
                    username,
                )
                # The disproof can no longer run on the anchor (deleted by
                # LinkedIn 2026-08-26), so it comes back empty for EVERY
                # creator-mode card and this branch used to refuse
                # unconditionally -- which is why Follow-primary profiles
                # returned "Could not confirm this is an incoming request"
                # 100% of the time (verified live on ertugrul-sahin).
                #
                # The deeplink is a strictly safer disproof than refusing:
                # if LinkedIn offers to SEND an invitation to this person,
                # the card was not an incoming request, and sending is
                # exactly what a note-bearing call asked for. Accept is
                # irreversible and lands on the first labeled button
                # (Follow) when misclassified; the deeplink cannot do that.
                deeplink_result = await self._send_via_custom_invite_deeplink(
                    url, username, note, page_text
                )
                if deeplink_result is not None:
                    return deeplink_result
                # No invite dialog either -- fall back to the original
                # refusal rather than guessing at Accept.
                return _connection_result(
                    url,
                    "send_failed",
                    "Could not confirm this is an incoming request and a note "
                    "was requested; refusing to click Accept. If this profile "
                    "really has a pending invitation from them, accept it "
                    "manually.",
                    note_sent=False,
                    profile=page_text,
                )
            # Accept clicks the first labeled button in the fingerprinted
            # row. There is deliberately no locale-text fallback: clicking
            # a button matched by exact text anywhere in the page risks
            # hitting the wrong control (or the Ignore button in another
            # locale), and accepting/ignoring is irreversible. When the
            # fingerprint does not match we report send_failed rather than
            # guess.
            clicked = await self._click_incoming_accept()
            if not clicked:
                return _connection_result(
                    url,
                    "send_failed",
                    "Could not find or click the Accept button.",
                    profile=page_text,
                )
            # LinkedIn propagates the accepted state asynchronously; an
            # immediate re-read can still render the old top card and
            # would report send_failed for a successful accept (observed
            # live 2026-06-11). Verify with one settle retry.
            verified_text = ""
            verified_state = None
            for attempt in range(2):
                if attempt:
                    await asyncio.sleep(3.0)
                verified = await self._read_main_profile(username)
                verified_text = verified.get("sections", {}).get("main_profile", "")
                verified_signals = await self._read_action_signals(username)
                verified_state = connection.detect_connection_state(verified_signals)
                if verified_state == "already_connected":
                    break
            if verified_state != "already_connected":
                return _connection_result(
                    url,
                    "send_failed",
                    "Accepted, but the profile did not transition to 1st-degree.",
                    profile=verified_text or page_text,
                )
            return _connection_result(
                url,
                "accepted",
                "Connection request accepted.",
                profile=verified_text,
            )

        # Follow-only profiles may have Connect hidden under the More menu
        # (high-follower / creator-mode profiles). Try opening it and
        # re-reading signals; if the vanityName invite anchor surfaces in
        # the menu, we can proceed with the deeplink. (The
        # has_invite_anchor=False guard is implicit: detect_connection_state
        # only returns "follow_only" after the has_invite_anchor branch
        # has already failed, so reaching this branch already implies it.)
        if state == "follow_only":
            opened = await self._open_more_menu()
            if opened:
                signals = await self._read_action_signals(username)
                # Close the menu before any subsequent navigation so it
                # doesn't intercept the upcoming page transition.
                try:
                    await self._press_escape()
                except OffLinkedInLandingError:
                    raise
                except Exception:
                    logger.debug("Escape after More-menu reread failed", exc_info=True)
                logger.info("Post-More signals for %s: signals=%s", username, signals)

        # Write-gate.
        #
        # Historically this required the vanityName invite anchor and returned
        # connect_unavailable without it. LinkedIn deleted that anchor from the
        # profile DOM on/around 2026-08-26, so the gate began rejecting 100% of
        # sends -- every profile, including ones with a visible Connect button.
        # The anchor is now treated as a fast path when present, and its
        # absence falls back to the deeplink probe rather than failing closed.
        #
        # This is NOT a loosening of the safety property. already_connected and
        # pending both return earlier, and the deeplink URL carries the
        # vanityName, so LinkedIn still resolves the invite to exactly the
        # requested person. See _send_via_custom_invite_deeplink for why a
        # relaxed DOM selector would be the unsafe alternative.
        if not signals.has_invite_anchor:
            logger.info(
                "No visible invite anchor for %s; falling back to the "
                "custom-invite deeplink probe",
                username,
            )

        deeplink_result = await self._send_via_custom_invite_deeplink(
            url, username, note, page_text
        )
        if deeplink_result is not None:
            return deeplink_result

        # LinkedIn opened no invite dialog for this vanityName. When a note was
        # requested, surface a Premium note-quota block if that is the reason
        # (the deeplink is already the current page, so no re-navigation).
        if note:
            note_limit_message = await self._probe_invite_note_limit()
            if note_limit_message is not None:
                return _connection_result(
                    url,
                    "custom_note_limit_reached",
                    note_limit_message,
                    note_sent=False,
                    profile=page_text,
                )
        return _connection_result(
            url,
            "connect_unavailable",
            "LinkedIn did not expose a usable Connect action for this profile.",
            profile=page_text,
        )
