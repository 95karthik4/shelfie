/** Confirm the photo before spending ~20 seconds and one hosted API call on it. */

import React from 'react';
import { Image, ScrollView, StyleSheet, Text } from 'react-native';

import { colors, radius, spacing } from '../theme';
import { Button, ErrorNotice } from '../components/ui';

interface Props {
  uri: string;
  error: string | null;
  onRetake: () => void;
  onAnalyze: () => void;
}

export function PreviewScreen({ uri, error, onRetake, onAnalyze }: Props) {
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Check the photo</Text>
      <Text style={styles.subtitle}>
        Spines should be readable and roughly upright. A clearer photo means fewer books to review.
      </Text>

      <Image source={{ uri }} style={styles.image} resizeMode="cover" />

      {error ? <ErrorNotice message={error} style={styles.error} /> : null}

      <Button label="Analyze Shelf" onPress={onAnalyze} style={styles.button} />
      <Button
        label="Retake / Choose Another"
        variant="secondary"
        onPress={onRetake}
        style={styles.button}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: spacing.xl, paddingTop: spacing.xxl },
  title: { fontSize: 26, fontWeight: '800', color: colors.text },
  subtitle: {
    fontSize: 14,
    color: colors.textMuted,
    lineHeight: 20,
    marginTop: spacing.xs,
    marginBottom: spacing.lg,
  },
  image: {
    width: '100%',
    height: 380,
    borderRadius: radius.lg,
    backgroundColor: colors.border,
  },
  error: { marginTop: spacing.lg },
  button: { marginTop: spacing.md },
});
