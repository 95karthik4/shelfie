/**
 * The wait. A measured scan took 21 seconds, so this screen has to make a long
 * pause feel intentional rather than broken: it names the stages that are
 * actually running server-side.
 */

import React from 'react';
import { ActivityIndicator, Image, StyleSheet, Text, View } from 'react-native';

import { colors, radius, spacing } from '../theme';

export function AnalyzingScreen({ uri }: { uri: string }) {
  return (
    <View style={styles.container}>
      <Image source={{ uri }} style={styles.thumb} resizeMode="cover" />
      <ActivityIndicator size="large" color={colors.primary} style={styles.spinner} />
      <Text style={styles.title}>Reading your shelf…</Text>
      <Text style={styles.body}>
        Finding the spines, reading the titles, and matching them against the catalog. This usually
        takes about 20 seconds — please keep the app open.
      </Text>
      <View style={styles.steps}>
        <Text style={styles.step}>1. Detecting book spines on this device&apos;s server</Text>
        <Text style={styles.step}>2. Reading titles and authors with a vision model</Text>
        <Text style={styles.step}>3. Matching each read against the catalog</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xl,
  },
  thumb: {
    width: 120,
    height: 120,
    borderRadius: radius.lg,
    marginBottom: spacing.xl,
    backgroundColor: colors.border,
  },
  spinner: { marginBottom: spacing.lg },
  title: { fontSize: 20, fontWeight: '700', color: colors.text },
  body: {
    fontSize: 14,
    color: colors.textMuted,
    lineHeight: 21,
    textAlign: 'center',
    marginTop: spacing.sm,
    maxWidth: 320,
  },
  steps: { marginTop: spacing.xl, alignSelf: 'stretch', paddingHorizontal: spacing.lg },
  step: { fontSize: 13, color: colors.textMuted, marginBottom: spacing.sm },
});
