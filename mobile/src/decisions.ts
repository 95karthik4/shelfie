/**
 * What the human decided about one detected spine.
 *
 * This lives above the item cards, in App, because the scan results tree
 * unmounts when the user switches to the Library tab. Decision state held
 * inside a card would be lost on the way back -- a confirmed book would offer
 * its Confirm button again, and a discard (which is frontend-only, with no
 * server record) would disappear entirely.
 *
 * Only the decision is lifted. Draft text, busy flags and validation errors
 * stay local to the card; losing a half-typed correction on a tab switch is
 * an acceptable cost, losing a decision is not.
 */

import { ConfirmedBook } from './api';

export type Decision =
  /** Persisted server-side. book is null when the server said 409 (already there). */
  | { kind: 'confirmed'; book: ConfirmedBook | null }
  /** Frontend-only: no request was made and no ConfirmedBook exists. */
  | { kind: 'discarded' };

export type DecisionMap = Record<number, Decision>;
