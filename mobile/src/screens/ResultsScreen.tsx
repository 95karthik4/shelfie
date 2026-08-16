/**
 * Every detected spine, and the decision the user still has to make about it.
 *
 * All items are rendered, in index order, whatever their status or decision --
 * a spine that matched nothing is as visible as one that matched perfectly,
 * and a discarded spine stays on screen labelled rather than vanishing.
 * Nothing is filtered, collapsed or auto-accepted.
 *
 * Decisions are owned by App and passed in, so they survive a trip to the
 * Library tab and back.
 */

import React, { useMemo } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { ScanResponse } from '../api';
import { Decision, DecisionMap } from '../decisions';
import { colors, spacing, statusPalette } from '../theme';
import { Button, Card, Pill } from '../components/ui';
import { ItemCard } from '../components/ItemCard';

interface Props {
  scan: ScanResponse;
  decisions: DecisionMap;
  onDecision: (itemId: number, decision: Decision | null) => void;
  onDone: () => void;
  onScanAnother: () => void;
}

export function ResultsScreen({ scan, decisions, onDecision, onDone, onScanAnother }: Props) {
  const counts = useMemo(() => {
    return scan.items.reduce(
      (totals, item) => ({ ...totals, [item.status]: totals[item.status] + 1 }),
      { auto: 0, review: 0, unmatched: 0 } as Record<'auto' | 'review' | 'unmatched', number>
    );
  }, [scan.items]);

  const decided = scan.items.filter(
    (item) => decisions[item.id] !== undefined || item.confirmed
  ).length;
  const outstanding = scan.items.length - decided;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>{scan.items.length} spines found</Text>
      <Text style={styles.subtitle}>
        Nothing is saved until you confirm it — including high-confidence matches. You can also
        correct a match or discard a spine.
      </Text>

      <Card style={styles.summary}>
        <View style={styles.pills}>
          <Pill
            text={`${counts.auto} high confidence`}
            fg={statusPalette.auto.fg}
            bg={statusPalette.auto.bg}
          />
          <Pill
            text={`${counts.review} to review`}
            fg={statusPalette.review.fg}
            bg={statusPalette.review.bg}
          />
          <Pill
            text={`${counts.unmatched} no match`}
            fg={statusPalette.unmatched.fg}
            bg={statusPalette.unmatched.bg}
          />
        </View>
        <Text style={styles.meta}>
          Detector: {scan.detector.source}
          {scan.detector.used_fallback ? ' (fallback)' : ''} · quality{' '}
          {scan.detector.quality.toFixed(2)}
          {scan.vlm.latency_ms !== null
            ? ` · model ${(scan.vlm.latency_ms / 1000).toFixed(1)}s`
            : ''}
        </Text>
        <Text style={styles.progress}>
          {outstanding === 0
            ? 'All books handled.'
            : `${outstanding} still waiting for your decision.`}
        </Text>
      </Card>

      {scan.items.length === 0 ? (
        <Card style={styles.empty}>
          <Text style={styles.emptyTitle}>No books detected</Text>
          <Text style={styles.emptyBody}>
            The scan worked, but no spines were found in this photo. Try getting closer, filling the
            frame with a single shelf, or improving the lighting.
          </Text>
          <Button label="Try another photo" onPress={onScanAnother} style={styles.button} />
        </Card>
      ) : (
        scan.items.map((item) => (
          <ItemCard
            key={item.id}
            item={item}
            decision={decisions[item.id]}
            onDecision={onDecision}
          />
        ))
      )}

      {scan.items.length > 0 ? (
        <View style={styles.footer}>
          <Button label="Scan another shelf" onPress={onScanAnother} />
          <Button label="Done" variant="secondary" onPress={onDone} style={styles.button} />
        </View>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: spacing.lg, paddingTop: spacing.xxl },
  title: { fontSize: 26, fontWeight: '800', color: colors.text },
  subtitle: { fontSize: 14, color: colors.textMuted, marginTop: spacing.xs, lineHeight: 20 },
  summary: { marginTop: spacing.lg, marginBottom: spacing.lg },
  pills: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  meta: { fontSize: 12, color: colors.textMuted, marginTop: spacing.md },
  progress: { fontSize: 13, color: colors.text, marginTop: spacing.sm, fontWeight: '600' },
  empty: { alignItems: 'flex-start' },
  emptyTitle: { fontSize: 18, fontWeight: '700', color: colors.text },
  emptyBody: { fontSize: 14, color: colors.textMuted, lineHeight: 20, marginTop: spacing.xs },
  footer: { marginTop: spacing.lg, marginBottom: spacing.xxl },
  button: { marginTop: spacing.md },
});
