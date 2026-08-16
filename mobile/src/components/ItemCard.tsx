/**
 * One detected spine, and the human decision attached to it.
 *
 * The assignment's three outcomes are all reachable here: confirm, correct, or
 * discard. Rules this component exists to enforce:
 *
 *  1. Nothing is added to the library without an explicit tap -- including
 *     high-confidence "auto" items. The backend agrees: it only writes a
 *     ConfirmedBook when the confirm endpoint is called.
 *  2. Discarding is frontend-only and explicit. It makes no request, creates
 *     no ConfirmedBook, and does not remove the spine from the screen -- the
 *     card stays, labelled, with Undo available.
 *  3. Deciding never erases what the models produced. The original read,
 *     suggestion, confidence and reasons stay on screen afterwards, so a user
 *     can always see what the AI said versus what the human chose.
 *
 * The decision itself is owned by App (see src/decisions.ts); this component
 * only reports it upward and renders it.
 */

import React, { useState } from 'react';
import { StyleSheet, Text, TextInput, View } from 'react-native';

import { ApiError, ScanItem, confirmScanItem } from '../api';
import { Decision } from '../decisions';
import { colors, radius, spacing, statusPalette } from '../theme';
import { Button, Card, ErrorNotice, Label, Pill } from './ui';

interface Props {
  item: ScanItem;
  decision: Decision | undefined;
  onDecision: (itemId: number, decision: Decision | null) => void;
}

export function ItemCard({ item, decision, onDecision }: Props) {
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(item.matched_title ?? item.raw_title ?? '');
  const [author, setAuthor] = useState(item.matched_author ?? item.raw_author ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const palette = statusPalette[item.status];
  // An item the server already knows is confirmed (e.g. results reloaded) is
  // treated as decided even without a local decision.
  const effective: Decision | undefined =
    decision ?? (item.confirmed ? { kind: 'confirmed', book: null } : undefined);

  async function submit(payload: Parameters<typeof confirmScanItem>[1]) {
    setBusy(true);
    setError(null);
    try {
      const book = await confirmScanItem(item.id, payload);
      onDecision(item.id, { kind: 'confirmed', book });
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 409) {
        // Already in the library -- an outcome, not a failure to fix.
        onDecision(item.id, { kind: 'confirmed', book: null });
      } else if (caught instanceof ApiError && caught.status === 404) {
        setError('This scan result no longer exists on the server. Run the scan again.');
      } else if (caught instanceof ApiError) {
        setError(caught.message);
      } else {
        setError('Something went wrong confirming this book.');
      }
    } finally {
      setBusy(false);
    }
  }

  function submitManual() {
    const cleanTitle = title.trim();
    if (!cleanTitle) {
      setError('A title is required to add this book.');
      return;
    }
    const cleanAuthor = author.trim();
    void submit(cleanAuthor ? { title: cleanTitle, author: cleanAuthor } : { title: cleanTitle });
  }

  function discard() {
    // Purely local: no request, no ConfirmedBook, nothing removed from view.
    setError(null);
    setEditing(false);
    onDecision(item.id, { kind: 'discarded' });
  }

  const accent = effective?.kind === 'discarded' ? colors.unmatched : palette.fg;

  return (
    <Card style={StyleSheet.flatten([styles.card, { borderLeftColor: accent }])}>
      <View style={styles.header}>
        <Text style={styles.spine}>Spine {item.index + 1}</Text>
        <Pill text={palette.label} fg={palette.fg} bg={palette.bg} />
      </View>

      <Label>Read from the spine</Label>
      {item.legible ? (
        <>
          <Text style={styles.readTitle}>{item.raw_title ?? '(no title read)'}</Text>
          <Text style={styles.readAuthor}>{item.raw_author ?? 'Author not readable'}</Text>
        </>
      ) : (
        <Text style={styles.readAuthor}>Nothing could be read from this spine.</Text>
      )}

      {item.catalog_id ? (
        <View style={styles.section}>
          <Label>Suggested catalog match</Label>
          <Text style={styles.matchTitle}>{item.matched_title}</Text>
          <Text style={styles.readAuthor}>{item.matched_author ?? 'Unknown author'}</Text>
          <Text style={styles.meta}>
            Confidence {(item.confidence * 100).toFixed(0)}% · catalog #{item.catalog_id}
          </Text>
        </View>
      ) : null}

      {item.reasons.length > 0 ? (
        <Text style={styles.reasons}>{item.reasons.map(prettyReason).join(' · ')}</Text>
      ) : null}

      {error ? <ErrorNotice message={error} style={styles.errorBox} /> : null}

      {effective?.kind === 'confirmed' ? (
        <View style={styles.doneBox}>
          <Text style={styles.doneText}>
            {effective.book
              ? `Added to library: ${effective.book.title}${
                  effective.book.author ? ` — ${effective.book.author}` : ''
                }`
              : 'Already in your library.'}
          </Text>
        </View>
      ) : effective?.kind === 'discarded' ? (
        <View style={styles.discardBox}>
          <Text style={styles.discardText}>Discarded — not added to library</Text>
          <Button
            label="Undo"
            variant="secondary"
            onPress={() => onDecision(item.id, null)}
            style={styles.spaced}
          />
        </View>
      ) : (
        <View style={styles.actions}>
          {editing || item.status === 'unmatched' || !item.catalog_id ? (
            <View>
              <Label style={styles.fieldLabel}>Title</Label>
              <TextInput
                value={title}
                onChangeText={setTitle}
                placeholder="Book title"
                placeholderTextColor={colors.textMuted}
                style={styles.input}
                autoCapitalize="words"
              />
              <Label style={styles.fieldLabel}>Author (optional)</Label>
              <TextInput
                value={author}
                onChangeText={setAuthor}
                placeholder="Author"
                placeholderTextColor={colors.textMuted}
                style={styles.input}
                autoCapitalize="words"
              />
              <Button label="Add this book" onPress={submitManual} busy={busy} />
              {item.catalog_id ? (
                <Button
                  label="Back to suggestion"
                  variant="ghost"
                  onPress={() => {
                    setEditing(false);
                    setError(null);
                  }}
                  style={styles.spaced}
                />
              ) : null}
              <Button
                label="Discard this spine"
                variant="ghost"
                onPress={discard}
                disabled={busy}
                style={styles.spaced}
              />
            </View>
          ) : (
            <View>
              <Button
                label={item.status === 'auto' ? 'Confirm this book' : 'Accept suggestion'}
                onPress={() => void submit({ catalog_id: item.catalog_id as string })}
                busy={busy}
              />
              <Button
                label="Correct it instead"
                variant="secondary"
                onPress={() => setEditing(true)}
                style={styles.spaced}
              />
              <Button
                label="Discard this spine"
                variant="ghost"
                onPress={discard}
                disabled={busy}
                style={styles.spaced}
              />
            </View>
          )}
        </View>
      )}
    </Card>
  );
}

/** Matcher reason codes are shouty constants; soften them for display. */
function prettyReason(reason: string): string {
  const map: Record<string, string> = {
    NOT_LEGIBLE: 'Spine unreadable',
    INVALID_CROP: 'Crop unusable',
    AUTHOR_UNREADABLE: 'Author not readable',
    TITLE_ONLY_CONFIDENCE_CAP: 'Title-only match',
    SUBSTRING_PENALTY_APPLIED: 'Title overlaps another book',
    DIFFERENT_WORK_AMBIGUITY: 'Could be a different book',
    OMNIBUS_AMBIGUITY: 'Could be a collected edition',
    EDITION_AMBIGUITY: 'Multiple editions match',
    LOW_SIMILARITY: 'Weak match',
    NO_TITLE_TEXT: 'No title text',
    EMPTY_CATALOG: 'Catalog empty',
  };
  return map[reason] ?? reason.toLowerCase().replace(/_/g, ' ');
}

const styles = StyleSheet.create({
  card: { marginBottom: spacing.md, borderLeftWidth: 4 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  spine: { fontSize: 13, fontWeight: '700', color: colors.textMuted },
  readTitle: { fontSize: 17, fontWeight: '600', color: colors.text, marginTop: 2 },
  matchTitle: { fontSize: 16, fontWeight: '600', color: colors.text, marginTop: 2 },
  readAuthor: { fontSize: 14, color: colors.textMuted, marginTop: 2 },
  meta: { fontSize: 12, color: colors.textMuted, marginTop: spacing.xs },
  section: {
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  reasons: { fontSize: 12, color: colors.textMuted, marginTop: spacing.sm, fontStyle: 'italic' },
  actions: { marginTop: spacing.lg },
  spaced: { marginTop: spacing.sm },
  fieldLabel: { marginBottom: spacing.xs },
  input: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: 16,
    color: colors.text,
    backgroundColor: colors.background,
    marginBottom: spacing.md,
  },
  errorBox: { marginTop: spacing.md },
  doneBox: {
    marginTop: spacing.lg,
    backgroundColor: colors.autoSoft,
    borderRadius: radius.md,
    padding: spacing.md,
  },
  doneText: { color: colors.auto, fontSize: 14, fontWeight: '600' },
  discardBox: {
    marginTop: spacing.lg,
    backgroundColor: colors.unmatchedSoft,
    borderRadius: radius.md,
    padding: spacing.md,
  },
  discardText: { color: colors.textMuted, fontSize: 14, fontWeight: '600' },
});
