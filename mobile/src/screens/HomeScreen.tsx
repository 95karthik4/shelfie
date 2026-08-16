/** Entry point: capture (primary) or pick from the library (secondary). */

import * as ImagePicker from 'expo-image-picker';
import React, { useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { API_BASE_URL } from '../api';
import { colors, spacing } from '../theme';
import { Button, Card, ErrorNotice } from '../components/ui';

interface Props {
  onOpenCamera: () => void;
  onPicked: (uri: string) => void;
}

export function HomeScreen({ onOpenCamera, onPicked }: Props) {
  const [error, setError] = useState<string | null>(null);

  async function pickFromLibrary() {
    setError(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError('Photo library access was declined. You can still take a photo with the camera.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.9,
    });
    if (!result.canceled && result.assets.length > 0) {
      onPicked(result.assets[0].uri);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Shelfie</Text>
      <Text style={styles.subtitle}>
        Photograph a bookshelf and turn it into a structured library. Every book is checked by you
        before it is saved.
      </Text>

      <Card style={styles.card}>
        <Text style={styles.cardTitle}>Scan a shelf</Text>
        <Text style={styles.cardBody}>
          Take a photo now, or choose one you already have. One shelf at a time works best.
        </Text>
        <Button label="Scan Shelf" onPress={onOpenCamera} style={styles.button} />
        <Button
          label="Choose Photo"
          variant="secondary"
          onPress={() => void pickFromLibrary()}
          style={styles.button}
        />
      </Card>

      {error ? <ErrorNotice message={error} style={styles.error} /> : null}

      <View style={styles.footer}>
        <Text style={styles.footerText}>Server: {API_BASE_URL}</Text>
        <Text style={styles.footerHint}>
          Set EXPO_PUBLIC_API_BASE_URL in mobile/.env to your computer&apos;s LAN IP when running on
          a phone.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: spacing.xl, paddingTop: spacing.xxl, flexGrow: 1 },
  title: { fontSize: 34, fontWeight: '800', color: colors.text },
  subtitle: {
    fontSize: 15,
    color: colors.textMuted,
    lineHeight: 22,
    marginTop: spacing.sm,
    marginBottom: spacing.xl,
  },
  card: { marginBottom: spacing.lg },
  cardTitle: { fontSize: 18, fontWeight: '700', color: colors.text },
  cardBody: { fontSize: 14, color: colors.textMuted, lineHeight: 20, marginTop: spacing.xs },
  button: { marginTop: spacing.md },
  error: { marginBottom: spacing.lg },
  footer: { marginTop: 'auto', paddingTop: spacing.xl },
  footerText: { fontSize: 12, color: colors.textMuted, fontWeight: '600' },
  footerHint: { fontSize: 12, color: colors.textMuted, marginTop: spacing.xs, lineHeight: 18 },
});
